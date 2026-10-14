"""광역 대지 분석 다이어그램 (understanding the node) — 893.2 x 1080 SVG, Illustrator 편집용.
단색(마룬) 계열. 그룹 id = Illustrator 그룹 이름."""
import base64, io, json, math, pickle, sys
import numpy as np, cv2
from PIL import Image, ImageChops
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

REG = '../regional/'
W, H = 893.2, 1080
OUT = sys.argv[1] if len(sys.argv) > 1 else 'precinct.svg'

# ---------------- palette (monochrome maroon; edit freely in Illustrator) ----------------
C = dict(ink='#2B1A1C', ink2='#6E5A5D', acc='#8E2C3A', acc_d='#5A1A24', acc_m='#B9707A', acc_l='#E7CED1',
         bg='#EEEAEA', bld='#D5CDCE', road='#FBFAFA', green='#E3DCDC', water='#D6C6C9', paper='#FFFFFF')
FONT = "Pretendard, 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"

# ---------------- data ----------------
geo = pickle.load(open(REG + 'geo.pkl', 'rb'))
N = json.load(open(REG + 'nodes_area.json'))
M = np.load('masks.npz')
X0, Y0, X1, Y1 = M['meta']
SITE_TRI = [(214869.5, 559042.0), (214933.4, 558829.5), (215088.8, 558866.0)]
SITE_C = (214960.0, 558915.0)
GONGNEUNG_STN = (214858.0, 559095.0)          # 공릉역 — 동일로 x 경춘선숲길 교차부 추정 (확인 필요)
SAHMYOOK = (217949.0, 560381.0)               # 삼육대 (지도 밖, 동측)
JN = np.array([(213398, 562500), (213421, 561821), (213489, 561143), (213557, 560690), (213648, 560238),
               (213807, 559786), (214011, 559424), (214193, 559197), (214329, 558971), (214510, 558745),
               (214692, 558519), (214828, 558202), (214896, 557750), (214919, 557071)], float)

S = W / (X1 - X0)                              # px per metre (1 px = 5.0 m)
def T(x, y): return (x - X0) * S, (Y1 - y) * S
MAP_BOTTOM = H

def f(v): return f'{v:.1f}'.rstrip('0').rstrip('.')
def pts_d(P, close=False):
    s = 'M' + ' L'.join(f'{f(x)} {f(y)}' for x, y in P)
    return s + ('Z' if close else '')

# ---------------- vectorising helpers ----------------
def mask_paths(mask, step, eps, tf, min_area=4):
    """binary mask (1 m/px, row 0 = Y1) -> compound path d (with holes) in the given transform."""
    m = mask[::step, ::step].astype(np.uint8) * 255
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for i, c in enumerate(cs):
        if cv2.contourArea(c) < min_area: continue
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) < 3: continue
        P = [tf(X0 + p[0][0] * step, Y1 - p[0][1] * step) for p in c]
        out.append(pts_d(P, True))
    return ' '.join(out)

def bld_path(tf, simp=0.6, min_area=12, clip=None):
    out = []
    for k, v in geo.items():
        if not k.startswith('B001'): continue
        for p, closed in v:
            if len(p) < 3: continue
            g = Polygon(p)
            if not g.is_valid or g.area < min_area: continue
            if clip is not None and not clip(g): continue
            g = g.simplify(simp)
            if g.is_empty or g.geom_type != 'Polygon': continue
            out.append(pts_d([tf(x, y) for x, y in g.exterior.coords[:-1]], True))
    return ' '.join(out)

def lines_path(prefixes, tf, simp=0.8):
    out = []
    for k, v in geo.items():
        if not k.startswith(prefixes): continue
        for p, closed in v:
            if len(p) < 2: continue
            g = LineString(p).simplify(simp)
            out.append(pts_d([tf(x, y) for x, y in g.coords], False))
    return ' '.join(out)

def img_uri(path, box=None, size=None, trim=True):
    im = Image.open(path).convert('RGB')
    if trim:   # strip white borders / rounded white corners
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bb = ImageChops.difference(im, bg).convert('L').point(lambda v: 255 if v > 18 else 0).getbbox()
        if bb: im = im.crop(bb)
        w, h = im.size; im = im.crop((int(w * .02), int(h * .02), int(w * .98), int(h * .98)))
    if box:    # cover-crop to aspect
        bw, bh = box; w, h = im.size; r = bw / bh
        if w / h > r: nw = int(h * r); im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else: nh = int(w / r); im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    if size: im = im.resize(size, Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'JPEG', quality=86)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()

def aerial_crop(name, cx, cy, half, size):
    """crop a georeferenced aerial (img/*.png) around data coords."""
    gr = json.load(open(REG + 'georef.json'))
    x0, x1, y0, y1 = gr[name]
    im = Image.open(REG + 'img/' + name).convert('RGB'); w, h = im.size
    px = lambda x: (x - x0) / (x1 - x0) * w; py = lambda y: (y1 - y) / (y1 - y0) * h
    im = im.crop((int(px(cx - half * size[0] / size[1])), int(py(cy + half)), int(px(cx + half * size[0] / size[1])), int(py(cy - half))))
    im = im.resize(size, Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'JPEG', quality=86)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()

# ---------------- svg builder ----------------
el = []
def add(s): el.append(s)
def text(x, y, s, size=7, color=None, weight=400, anchor='start', style='normal', ls=1.32, opacity=1, spacing=0, gid=None):
    lines = s.split('\n'); color = color or C['ink']
    t = (f'<text x="{f(x)}" y="{f(y)}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" '
         f'text-anchor="{anchor}" font-style="{style}" letter-spacing="{spacing}" opacity="{opacity}"' + (f' id="{gid}"' if gid else '') + '>')
    for i, ln in enumerate(lines):
        t += f'<tspan x="{f(x)}" dy="{0 if i == 0 else f(size * ls)}">{ln}</tspan>'
    return t + '</text>'
def label_box(x, y, s, size=7.2, anchor='start', dark=False):
    w = sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s) + 6; h = size + 5
    bx = x if anchor == 'start' else (x - w / 2 if anchor == 'middle' else x - w)
    return (f'<g><rect x="{f(bx)}" y="{f(y - h + 3)}" width="{f(w)}" height="{f(h)}" fill="{C["acc_d"] if dark else C["acc"]}"/>'
            + text(bx + 3, y, s, size, '#FFFFFF', 600) + '</g>')

add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add(f'<defs><clipPath id="clipMap"><rect x="0" y="0" width="{W}" height="{f(MAP_BOTTOM)}"/></clipPath>')
# focus square
FD = (213430.6, 560503.0, 3294.0)               # focus square in map coords (x0, y_top, side m)
FX0, FY0 = T(FD[0], FD[1]); FS = FD[2] * S
add(f'<clipPath id="clipFocus"><rect x="{FX0}" y="{FY0}" width="{FS}" height="{FS}"/></clipPath></defs>')
add(f'<rect id="paper" width="{W}" height="{H}" fill="{C["paper"]}"/>')

# ======== MAIN MAP ========
add('<g id="MAP" clip-path="url(#clipMap)">')
ax0, ay0 = T(X0, Y1); ax1, ay1 = T(X1, Y0)
add(f'<rect id="map_bg" x="{f(ax0)}" y="{f(ay0)}" width="{f(ax1 - ax0)}" height="{f(ay1 - ay0)}" fill="{C["bg"]}"/>')
add(f'<path id="map_green" d="{mask_paths(M["green"], 3, 1.2, T, 6)}" fill="{C["green"]}" fill-rule="evenodd"/>')
add(f'<path id="map_water" d="{mask_paths(M["water"], 2, 1.0, T, 4)}" fill="{C["water"]}" fill-rule="evenodd"/>')
add(f'<path id="map_roads" d="{mask_paths(M["road"], 1, 1.4, T, 30)}" fill="{C["road"]}" fill-rule="evenodd"/>')
BLD = bld_path(T)
add(f'<path id="map_buildings" d="{BLD}" fill="{C["bld"]}"/>')
add(f'<g id="map_rail"><path d="{lines_path(("A015",), T)}" fill="none" stroke="{C["ink2"]}" stroke-width="1.1"/>'
    f'<path d="{lines_path(("A015",), T)}" fill="none" stroke="#FFFFFF" stroke-width="0.5" stroke-dasharray="2 2"/></g>')
# focus area tint + darker buildings inside
add(f'<g id="focus_tint" clip-path="url(#clipFocus)"><rect x="{FX0}" y="{FY0}" width="{FS}" height="{FS}" fill="{C["acc"]}" opacity="0.26"/>'
    f'<path d="{BLD}" fill="{C["acc"]}" opacity="0.55"/></g>')
# grid (500 m)
g = []
for gx in np.arange(math.floor(X0 / 500) * 500, X1 + 1, 500):
    x, _ = T(gx, 0); g.append(f'M{f(x)} 0 V{f(MAP_BOTTOM)}')
for gy in np.arange(math.floor(Y0 / 500) * 500, Y1 + 1, 500):
    _, y = T(0, gy); g.append(f'M0 {f(y)} H{W}')
add(f'<path id="grid_500m" d="{" ".join(g)}" stroke="#FFFFFF" stroke-width="0.5" opacity="0.75" fill="none"/>')
add('</g>')

# ======== FOCUS FRAME / RINGS / SITE ========
add('<g id="FOCUS">')
add(f'<rect x="{FX0}" y="{FY0}" width="{FS}" height="{FS}" fill="none" stroke="{C["acc_d"]}" stroke-width="1" stroke-dasharray="4 3"/>')
for cx, cy in ((FX0, FY0), (FX0 + FS, FY0), (FX0, FY0 + FS), (FX0 + FS, FY0 + FS)):
    add(f'<path d="M{cx - 12} {cy} H{cx + 12} M{cx} {cy - 12} V{cy + 12}" stroke="{C["acc_d"]}" stroke-width="1.6"/>')
for s_, x, y in (('N', FX0 + FS / 2, FY0 - 10), ('S', FX0 + FS / 2, FY0 + FS - 8), ('W', FX0 + 8, FY0 + FS / 2 + 9), ('E', FX0 + FS + 8, FY0 + FS / 2 + 9)):
    add(text(x, y, s_, 26, C['acc'], 300, 'middle' if s_ in 'NS' else 'start', opacity=0.55))
sx, sy = T(*SITE_C)
for r_m, dash, w_ in ((500, '3 2.5', 1.1), (1000, '1.5 2.5', 0.7)):
    add(f'<circle cx="{f(sx)}" cy="{f(sy)}" r="{f(r_m * S)}" fill="none" stroke="{C["acc_d"]}" stroke-width="{w_}" stroke-dasharray="{dash}"/>')
add(text(sx + 500 * S * 0.71 + 3, sy - 500 * S * 0.71 - 2, '500 m', 7, C['acc_d'], 600))
add(text(sx + 1000 * S * 0.71 + 3, sy - 1000 * S * 0.71 - 2, '1 km', 7, C['acc_d'], 600))
add(f'<path id="site" d="{pts_d([T(*p) for p in SITE_TRI], True)}" fill="{C["acc_d"]}" stroke="#FFFFFF" stroke-width="0.8"/>')
add('</g>')

# ======== NODES, LABELS, PHOTOS ========
add('<g id="LABELS">')
def node_poly(k, fill_op=0.0):
    P = [T(*p) for p in N[k]['poly']]
    return f'<path d="{pts_d(P, True)}" fill="{C["acc_m"]}" fill-opacity="{fill_op}" stroke="{C["acc_d"]}" stroke-width="1" stroke-dasharray="3 1.8"/>'
for k in ('SEOULTECH', 'SWU', 'INDUK', 'KWANGWOON', 'S-DBC', 'KW-STATION'):
    add(node_poly(k, 0.18))
def photo(x, y, w, h, uri, gid):
    return (f'<g id="{gid}"><rect x="{f(x - 2)}" y="{f(y - 2)}" width="{f(w + 4)}" height="{f(h + 4)}" fill="#FFFFFF"/>'
            f'<image x="{f(x)}" y="{f(y)}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" xlink:href="{uri}"/></g>')
PW_, PH_ = 104, 68
def callout(gid, node_xy, px, py, name, file, anchor_pt='bottom', dark=False):
    """photo at (px,py) + label under it + leader from the label/photo to the node."""
    cx, cy = node_xy
    lx = px + PW_ / 2; ly = py + PH_ + 16 if cy > py + PH_ else py - 2
    if cy < py: ly = py - 2
    add(f'<path d="M{f(cx)} {f(cy)} L{f(lx)} {f(ly)}" stroke="{C["acc_d"]}" stroke-width="0.7"/>')
    add(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="2.2" fill="{C["acc_d"]}"/>')
    add(photo(px, py, PW_, PH_, img_uri(file, (PW_, PH_), (PW_ * 4, PH_ * 4)), 'ph_' + gid))
    add(label_box(px - 2, py + PH_ + 14, name, 8.2, dark=dark))
NC = {k: T(*N[k]['center']) for k in N}
callout('SEOULTECH', NC['SEOULTECH'], 586, 452, '서울과학기술대학교', 'photos/과기대.webp')
callout('SWU', NC['SWU'], 772, 620, '서울여자대학교', 'photos/서울여대.jpg')
callout('INDUK', NC['INDUK'], 16, 548, '인덕대학교', 'photos/인덕대.jpg')
callout('KWANGWOON', NC['KWANGWOON'], 16, 868, '광운대학교', 'photos/광운대.jpg')
gx, gy = T(*GONGNEUNG_STN)
callout('GONGNEUNG', (gx, gy), 236, 536, '공릉역 (7호선)', 'photos/공릉역.jpg')
add(f'<g id="stn_gongneung"><circle cx="{f(gx)}" cy="{f(gy)}" r="4.8" fill="#FFFFFF" stroke="{C["acc_d"]}" stroke-width="1.3"/>'
    f'<text x="{f(gx)}" y="{f(gy + 2.6)}" font-family="{FONT}" font-size="6.5" font-weight="700" fill="{C["acc_d"]}" text-anchor="middle">7</text></g>')
for k, s_, dx, dy in (('S-DBC', 'S-DBC  창동 개발 거점', 34, 8), ('KW-STATION', '광운대역세권 개발', 30, 30)):
    cx, cy = NC[k]
    add(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="2.2" fill="{C["acc_d"]}"/>')
    add(f'<path d="M{f(cx)} {f(cy)} L{f(cx + dx)} {f(cy + dy)}" stroke="{C["acc_d"]}" stroke-width="0.7"/>')
    add(label_box(cx + dx, cy + dy + 5, s_, 8.6, dark=True))
sx, sy = T(*SITE_C)
add(label_box(sx + 12, sy + 26, 'SITE  공릉동 714', 9, dark=True))
jx, jy = T(214080, 559900)
add(text(jx, jy, '중\n랑\n천', 8.5, C['acc_d'], 600, opacity=0.85, ls=1.15))
fx_, fy_ = T(215300, 558690)
add(text(fx_, fy_, '경춘선숲길', 7.5, C['acc_d'], 600, opacity=0.9))
dx_, dy_ = T(214790, 558330)
add(text(dx_, dy_, '동일로', 7.5, C['acc_d'], 600, opacity=0.9))
ay = T(0, SAHMYOOK[1])[1]
add(f'<path d="M{f(W - 12)} {f(ay)} l-9 -4.5 v9 Z" fill="{C["acc_d"]}"/>')
add(label_box(W - 24, ay + 4, '삼육대학교 1.5 km', 7.6, anchor='end', dark=True))
add('</g>')

# ======== street photos (top) ========
add('<g id="STREET_PHOTOS">')
PH = [('01  남측 도로', 'photos/남측도로.jpg', ((SITE_TRI[1][0] + SITE_TRI[2][0]) / 2, (SITE_TRI[1][1] + SITE_TRI[2][1]) / 2 - 12)),
      ('02  경춘선숲길', 'photos/경춘선숲길.jpg', ((SITE_TRI[2][0] + SITE_TRI[0][0]) / 2 + 10, (SITE_TRI[2][1] + SITE_TRI[0][1]) / 2 + 8)),
      ('03  동일로', 'photos/동일로.jpg', ((SITE_TRI[0][0] + SITE_TRI[1][0]) / 2 - 12, (SITE_TRI[0][1] + SITE_TRI[1][1]) / 2))]
pw, ph_, gap = 168, 112, 10
px0 = W - 16 - 3 * pw - 2 * gap
for i, (cap, fn, loc) in enumerate(PH):
    x = px0 + i * (pw + gap); y = 34
    add(label_box(x - 2, y - 6, cap, 8.2))
    add(photo(x, y, pw, ph_, img_uri(fn, (pw, ph_), (pw * 4, ph_ * 4), trim=False), f'street_{i + 1}'))
    tx, ty = T(*loc)
    add(f'<path d="M{f(x + pw / 2)} {f(y + ph_ + 2)} L{f(tx)} {f(ty)}" stroke="{C["acc_d"]}" stroke-width="0.6" opacity="0.85"/>'
        f'<circle cx="{f(tx)}" cy="{f(ty)}" r="2.4" fill="#FFFFFF" stroke="{C["acc_d"]}" stroke-width="0.9"/>')
add('</g>')

# ======== scale + north ========
add('<g id="SCALE_NORTH">')
bx, by = W - 262, H - 34
add(f'<rect x="{f(bx)}" y="{f(by)}" width="248" height="24" fill="#FFFFFF" opacity="0.9"/>')
sbx, sby = bx + 10, by + 17
add(f'<rect x="{f(sbx)}" y="{f(sby - 2)}" width="{f(500 * S)}" height="2" fill="{C["ink"]}"/>')
add(f'<path d="M{f(sbx)} {f(sby)} H{f(sbx + 1000 * S)} M{f(sbx)} {f(sby - 4)} V{f(sby)} M{f(sbx + 500 * S)} {f(sby - 4)} V{f(sby)} M{f(sbx + 1000 * S)} {f(sby - 4)} V{f(sby)}" stroke="{C["ink"]}" stroke-width="0.8" fill="none"/>')
add(text(sbx, sby - 6, '0', 6, C['ink'])); add(text(sbx + 500 * S, sby - 6, '500', 6, C['ink'], anchor='middle')); add(text(sbx + 1000 * S + 4, sby - 6, '1 km', 6, C['ink']))
nx_ = bx + 236
add(f'<path d="M{f(nx_)} {f(by + 3)} l5 14 l-5 -3.5 l-5 3.5 Z" fill="{C["ink"]}"/>' + text(nx_, by + 23, 'N', 6, C['ink'], 700, 'middle'))
add('</g>')
add('</svg>')
open(OUT, 'w', encoding='utf-8').write('\n'.join(el))
print('saved', OUT, round(sum(len(e) for e in el) / 1e6, 2), 'MB')
