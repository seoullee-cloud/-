import pickle, json, os, zipfile
import numpy as np, trimesh, ezdxf
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
import furniture_fdm as F
FLIP = {'dining_table', 'desk'}   # printed upside down (top on the bed)

SCALE = 1 / 50
items, groups = pickle.load(open('groups.pkl', 'rb'))
doc = ezdxf.readfile('plan.dxf')
walls = []
for e in doc.modelspace():
    if e.dxftype() == 'POLYLINE' and e.dxf.layer.startswith(('A-WALL', 'A-GLAZ', 'A-COLS')):
        p = [(v.x, v.y) for v in e.points()]
        if e.is_closed: p.append(p[0])
        walls.append(LineString(p))
WALL = unary_union(walls)

def bb_of(idx):
    P = np.vstack([items[i]['pts'] for i in idx])
    return [P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()]
def gbb(g): return groups[g]['bb']
def ebb(i): return list(items[i]['bb'])
def unit_of(bb):
    x = bb[0]
    return 'M84-A' if x < 208000 else ('M84-B' if x < 235000 else 'M59')

SIDES = ('+y', '-y', '+x', '-x')
def side_pts(bb, s, n=7):
    x0, y0, x1, y1 = bb
    t = np.linspace(0.15, 0.85, n)
    if s == '+y': return [(x0 + (x1 - x0) * k, y1 + 2) for k in t]
    if s == '-y': return [(x0 + (x1 - x0) * k, y0 - 2) for k in t]
    if s == '+x': return [(x1 + 2, y0 + (y1 - y0) * k) for k in t]
    return [(x0 - 2, y0 + (y1 - y0) * k) for k in t]
def wall_side(bb, allowed=SIDES):
    d = {s: np.mean([WALL.distance(Point(p)) for p in side_pts(bb, s)]) for s in allowed}
    return min(d, key=d.get)
def long_sides(bb):   # sides parallel to the long axis
    return ('+y', '-y') if (bb[2] - bb[0]) >= (bb[3] - bb[1]) else ('+x', '-x')
def end_sides(bb):
    return ('+x', '-x') if (bb[2] - bb[0]) >= (bb[3] - bb[1]) else ('+y', '-y')
def toward(bb, target_bb):
    """side of bb that faces target (front) -> back is the opposite"""
    c = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]); t = np.array([(target_bb[0] + target_bb[2]) / 2, (target_bb[1] + target_bb[3]) / 2])
    v = t - c
    front = ('+x' if v[0] > 0 else '-x') if abs(v[0]) > abs(v[1]) else ('+y' if v[1] > 0 else '-y')
    return {'+x': '-x', '-x': '+x', '+y': '-y', '-y': '+y'}[front]
def band_side(bb, i):
    """sofa/armchair: side where the thin full-width back band sits"""
    b = ebb(i); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    X0, Y0, X1, Y1 = bb
    d = {'+y': Y1 - cy, '-y': cy - Y0, '+x': X1 - cx, '-x': cx - X0}
    return min(d, key=d.get)

# ---- frame mapping (local: width x, depth y, back at +y) ----
def dims(bb, back):
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    return (w, h) if back in ('+y', '-y') else (h, w)
def to_local(bb, back, px, py):
    x0, y0, x1, y1 = bb
    if back == '+y': return px - x0, py - y0
    if back == '-y': return x1 - px, y1 - py
    if back == '+x': return y1 - py, px - x0
    return py - y0, x1 - px
def rect_local(bb, back, r):
    a = to_local(bb, back, r[0], r[1]); b = to_local(bb, back, r[2], r[3])
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
def place_matrix(bb, back):
    x0, y0, x1, y1 = bb
    T = np.eye(4)
    if back == '+y': R, t = np.eye(2), (x0, y0)
    elif back == '-y': R, t = -np.eye(2), (x1, y1)
    elif back == '+x': R, t = np.array([[0, 1], [-1, 0]]), (x0, y1)
    else: R, t = np.array([[0, -1], [1, 0]]), (x1, y0)
    T[:2, :2] = R; T[:2, 3] = t
    return T

specs = []   # (unit, type, bb, back, builder)
def add(kind, bb, back, fn):
    specs.append(dict(kind=kind, bb=bb, back=back, fn=fn, unit=unit_of(bb)))

def center_rect(i): return ebb(i)
def circ(i):
    b = ebb(i); return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) / 2)

# ---------------- M84-A (left plan) ----------------
bb = gbb(0); add('sofa', bb, band_side(bb, 235), lambda W, D, l: F.sofa(W, D, seats=2, arm=200, back=220))
add('dining_table', gbb(3), '+y', lambda W, D, l: F.dining_table(W, D))
for g in (1, 2, 4, 5):
    bb = gbb(g); add('dining_chair', bb, toward(bb, gbb(3)), lambda W, D, l: F.dining_chair(W, D))
# kitchen run (group 6): counter 253, fridge 244, sink 251, tap 250, hob 249, burners 245-248
kbb = ebb(253); kback = wall_side(kbb, long_sides(kbb))
def kitchen_fn(kbb, kback, sink, tap, hob, burners):
    def fn(W, D, l):
        s = rect_local(kbb, kback, sink); h = rect_local(kbb, kback, hob)
        t = to_local(kbb, kback, *tap)
        bs = [(*to_local(kbb, kback, x, y), r) for x, y, r in burners]
        return F.kitchen(W, D, sink=s, tap=t, hob=h, burners=bs)
    return fn
def tapc(i): c = circ(i); return (c[0], c[1])
add('kitchen_counter', kbb, kback, kitchen_fn(kbb, kback, ebb(251), tapc(250), ebb(249), [circ(i) for i in (245, 246, 247, 248)]))
add('fridge', ebb(244), kback, lambda W, D, l: F.fridge(W, D))
bb = gbb(7); add('wardrobe', bb, wall_side(bb, long_sides(bb)), lambda W, D, l: F.wardrobe(W, D))
def mon_w(i):
    b = ebb(i); return max(b[2] - b[0], b[3] - b[1])
def desk_fn(bb_, back_, mon_i):
    def fn(W, D, l):
        m = rect_local(bb_, back_, ebb(mon_i))
        return F.desk(W, D, mon=m)
    return fn
bb = ebb(223); back = wall_side(bb); add('desk', bb, back, desk_fn(bb, back, 220)); specs[-1]['mon_w'] = mon_w(220)
bb = gbb(8); add('desk_chair', bb, toward(bb, ebb(223)), lambda W, D, l: F.desk_chair(W, D))
bb = gbb(10); add('armchair', bb, band_side(bb, 191), lambda W, D, l: F.armchair(W, D, arm=150, back=180))
bb = gbb(11); add('bed_double', bb, wall_side(bb, end_sides(bb)), lambda W, D, l: F.bed(W, D))
bb = ebb(231); back = wall_side(bb); add('desk', bb, back, desk_fn(bb, back, 228)); specs[-1]['mon_w'] = mon_w(228)
bb = gbb(13); add('desk_chair', bb, toward(bb, ebb(231)), lambda W, D, l: F.desk_chair(W, D))
bb = gbb(14); add('wardrobe', bb, wall_side(bb, long_sides(bb)), lambda W, D, l: F.wardrobe(W, D))
bb = ebb(268); add('armchair', bb, band_side(bb, 267), lambda W, D, l: F.armchair(W, D, arm=150, back=200))
bb = ebb(274); add('bed_double', bb, wall_side(bb, end_sides(bb)), lambda W, D, l: F.bed(W, D))
bb = gbb(16); add('tall_cabinet', bb, wall_side(bb, long_sides(bb)), lambda W, D, l: F.cabinet(W, D, 2400, n_doors=2, tall=True))
bb = gbb(17); add('shoe_cabinet', bb, wall_side(bb, long_sides(bb)), lambda W, D, l: F.shoe_cabinet(W, D))
def vanity_fn(bb_, back_, bowls, taps):
    def fn(W, D, l):
        return F.vanity(W, D, [rect_local(bb_, back_, ebb(i)) for i in bowls], [to_local(bb_, back_, *tapc(i)) for i in taps])
    return fn
bb = gbb(18); back = wall_side(bb, long_sides(bb)); add('vanity_double', bb, back, vanity_fn(bb, back, (196, 198), (195, 197)))
bb = gbb(19); add('toilet', bb, wall_side(bb, end_sides(bb)), lambda W, D, l: F.toilet(W, D))
bb = gbb(20); back = wall_side(bb, long_sides(bb))
add('bathtub', bb, back, (lambda bb_, back_: (lambda W, D, l: F.bathtub(W, D, tap=to_local(bb_, back_, *tapc(201)))))(bb, back))

# ---------------- M84-B (middle plan) ----------------
for dg, cg in ((21, 23), (33, 34)):
    d_bb = ebb([i for i in groups[dg]['idx'] if items[i]['bb'][2] - items[i]['bb'][0] > 1000 or items[i]['bb'][3] - items[i]['bb'][1] > 1000][0])
    m_i = [i for i in groups[dg]['idx'] if ebb(i) != d_bb][0]
    back = wall_side(d_bb); add('desk', d_bb, back, desk_fn(d_bb, back, m_i)); specs[-1]['mon_w'] = mon_w(m_i)
    bb = gbb(cg); add('desk_chair', bb, toward(bb, d_bb), lambda W, D, l: F.desk_chair(W, D))
for i, band in ((84, 85), (86, 87)):
    bb = ebb(i); bb = [bb[0], bb[1], bb[2], max(bb[3], ebb(band)[3])]
    add('lounge_chair', bb, band_side(bb, band), lambda W, D, l: F.lounge_chair(W, D))
add('side_table', ebb(88), '+y', lambda W, D, l: F.side_table(W, D))
for g in (24, 35, 45):
    bb = gbb(g); add('wardrobe', bb, wall_side(bb, long_sides(bb)), lambda W, D, l: F.wardrobe(W, D))
for g in (25, 36):
    bb = gbb(g); add('bed_double', bb, wall_side(bb, end_sides(bb)), lambda W, D, l: F.bed(W, D))
for g in (26, 27):
    bb = gbb(g); band = [i for i in groups[g]['idx'] if items[i]['bb'][3] - items[i]['bb'][1] < 250][0]
    add('armchair', bb, band_side(bb, band), lambda W, D, l: F.armchair(W, D, arm=150, back=180))
for g in (28, 37, 46):
    bb = gbb(g); add('shower_tray', bb, wall_side(bb), lambda W, D, l: F.shower(W, D))
for g in (29, 38, 49):
    bb = gbb(g); add('toilet', bb, wall_side(bb, end_sides(bb)), lambda W, D, l: F.toilet(W, D))
for g in (30, 39, 48):
    bb = gbb(g); back = wall_side(bb, long_sides(bb))
    bowl = [i for i in groups[g]['idx'] if items[i]['closed'] and len(items[i]['pts']) > 12][0]
    tap = [i for i in groups[g]['idx'] if items[i]['closed'] and len(items[i]['pts']) <= 8][0]
    add('vanity', bb, back, vanity_fn(bb, back, (bowl,), (tap,)))
for g in (31, 32):
    long_i = [i for i in groups[g]['idx'] if items[i]['bb'][2] - items[i]['bb'][0] > 2500 and len(items[i]['pts']) == 4][0]
    short_i = [i for i in groups[g]['idx'] if items[i]['closed']][0]
    bl = ebb(long_i); bs = ebb(short_i)
    bs = [bs[0], bs[1], bs[2], bs[3]]
    bl = [bl[0], bl[1], bl[2], min(bl[3], bs[1])]
    add('wardrobe', bl, wall_side(bl, long_sides(bl)), lambda W, D, l: F.wardrobe(W, D))
    add('wardrobe', bs, wall_side(bs, long_sides(bs)), lambda W, D, l: F.wardrobe(W, D))

# ---------------- M59 (right plan) ----------------
bb = ebb([i for i in groups[40]['idx'] if items[i]['bb'][2] - items[i]['bb'][0] > 1000][0])
m_i = [i for i in groups[40]['idx'] if items[i]['bb'][2] - items[i]['bb'][0] < 1000][0]
back = wall_side(bb); add('desk', bb, back, desk_fn(bb, back, m_i)); specs[-1]['mon_w'] = mon_w(m_i)
cbb = gbb(42); add('desk_chair', cbb, toward(cbb, bb), lambda W, D, l: F.desk_chair(W, D))
kbb = ebb(21); kback = wall_side(kbb, long_sides(kbb))
add('kitchen_counter', kbb, kback, kitchen_fn(kbb, kback, ebb(19), tapc(18), ebb(17), [circ(i) for i in (13, 14, 15, 16)]))
add('fridge', ebb(22), kback, lambda W, D, l: F.fridge(W, D))
bb = gbb(43); add('tv_console', bb, wall_side(bb, long_sides(bb)), lambda W, D, l: F.tv_console(W, D))
bb = ebb(1); add('armchair', bb, band_side(bb, 0), lambda W, D, l: F.armchair(W, D, arm=150, back=180))
bb = ebb(4); add('bed_double', bb, wall_side(bb, end_sides(bb)), lambda W, D, l: F.bed(W, D))
for i in (32, 29):
    bb = ebb(i); add('washer' if i == 32 else 'dryer', bb, wall_side(bb, ('+y', '-y')), lambda W, D, l: F.washer(W, D))
bb = gbb(50); add('shoe_cabinet', bb, wall_side(bb, long_sides(bb)), lambda W, D, l: F.shoe_cabinet(W, D))
bb = gbb(51); add('tall_cabinet', bb, wall_side(bb, long_sides(bb)), lambda W, D, l: F.cabinet(W, D, 2400, n_doors=2, tall=True))

# ---------------- build + export ----------------
OUT = 'out_fdm'
os.makedirs(OUT, exist_ok=True)
counts, report, layout = {}, [], {}
for s in specs:
    W, D = dims(s['bb'], s['back'])
    man = s['fn'](W, D, None)
    mesh = man.to_mesh()
    tm = trimesh.Trimesh(np.array(mesh.vert_properties)[:, :3], np.array(mesh.tri_verts), process=True)
    key = f"{s['unit']}_{s['kind']}"
    counts[key] = counts.get(key, 0) + 1
    name = f"{key}_{counts[key]:02d}"
    # plan-placed copy for the layout check
    placed = tm.copy(); placed.apply_transform(place_matrix(s['bb'], s['back']))
    layout.setdefault(s['unit'], []).append(placed)
    # printable copy: 1:50, sitting on z=0, centred
    p = tm.copy(); p.apply_scale(SCALE)
    if s['kind'] in FLIP:
        p.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    p.apply_translation([-p.bounds[0][0] - p.extents[0] / 2, -p.bounds[0][1] - p.extents[1] / 2, -p.bounds[0][2]])
    os.makedirs(f"{OUT}/{s['unit']}", exist_ok=True)
    p.export(f"{OUT}/{s['unit']}/{name}.stl")
    report.append(dict(name=name, unit=s['unit'], kind=s['kind'], back=s['back'],
                       real_mm=[round(W), round(D), round(tm.extents[2])],
                       print_mm=[round(v, 1) for v in p.extents], watertight=bool(p.is_watertight),
                       volume_mm3=round(float(p.volume), 1)))
    print(name, s['back'], [round(v, 1) for v in p.extents], 'watertight' if p.is_watertight else 'NOT WATERTIGHT')
# separate monitors for every desk (width from the plan symbol)
for s in specs:
    if s['kind'] == 'desk' and s.get('mon_w'):
        mm = F.monitor(s['mon_w']).to_mesh()
        t = trimesh.Trimesh(np.array(mm.vert_properties)[:, :3], np.array(mm.tri_verts), process=True)
        t.apply_scale(SCALE); t.apply_translation([-t.bounds[0][0] - t.extents[0] / 2, -t.bounds[0][1] - t.extents[1] / 2, -t.bounds[0][2]])
        key = f"{s['unit']}_monitor"; counts[key] = counts.get(key, 0) + 1
        name = f"{key}_{counts[key]:02d}"; t.export(f"{OUT}/{s['unit']}/{name}.stl")
        report.append(dict(name=name, unit=s['unit'], kind='monitor', back='+y', real_mm=[round(v / SCALE) for v in t.extents],
                           print_mm=[round(v, 1) for v in t.extents], watertight=bool(t.is_watertight), volume_mm3=round(float(t.volume), 1)))
json.dump(report, open(OUT + '/report.json', 'w'), ensure_ascii=False, indent=1)
for u, ms in layout.items():
    m = trimesh.util.concatenate(ms); m.apply_scale(SCALE)
    m.apply_translation(-m.bounds[0] * [1, 1, 1])
    m.export(f'{OUT}/_layout_{u}_check.stl')
pickle.dump({u: [(x.vertices, x.faces) for x in ms] for u, ms in layout.items()}, open('layout_fdm.pkl', 'wb'))
print(len(specs), 'pieces')
