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

S = 0.17                                       # px per metre (1 px = 5.9 m)
SX, SY = 480.0, 600.0                          # canvas position of the site centre
def T(x, y): return SX + (x - SITE_C[0]) * S, SY - (y - SITE_C[1]) * S
MAP_BOTTOM = T(0, Y0)[1]                       # ~913

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
FX0, FY0, FS = 220.0, 330.0, 560.0
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
for s_, x, y in (('N', FX0 + FS / 2, FY0 - 10), ('S', FX0 + FS / 2, FY0 + FS - 8), ('W', FX0 + 8, FY0 + FS / 2 + 8), ('E', FX0 + FS + 7, FY0 + FS / 2 + 8)):
    add(text(x, y, s_, 22, C['acc'], 300, 'middle' if s_ in 'NS' else 'start', opacity=0.55))
sx, sy = T(*SITE_C)
for r_m, dash, w_ in ((500, '3 2.5', 1.1), (1000, '1.5 2.5', 0.7)):
    add(f'<circle cx="{f(sx)}" cy="{f(sy)}" r="{f(r_m * S)}" fill="none" stroke="{C["acc_d"]}" stroke-width="{w_}" stroke-dasharray="{dash}"/>')
add(text(sx + 500 * S * 0.71 + 3, sy - 500 * S * 0.71 - 2, '500 m', 6, C['acc_d'], 600))
add(text(sx + 1000 * S * 0.71 + 3, sy - 1000 * S * 0.71 - 2, '1 km', 6, C['acc_d'], 600))
add(f'<path id="site" d="{pts_d([T(*p) for p in SITE_TRI], True)}" fill="{C["acc_d"]}" stroke="#FFFFFF" stroke-width="0.8"/>')
add('</g>')

# ======== NODES, LABELS, PHOTOS ========
add('<g id="LABELS">')
def node_poly(k, fill_op=0.0):
    P = [T(*p) for p in N[k]['poly']]
    return f'<path d="{pts_d(P, True)}" fill="{C["acc_m"]}" fill-opacity="{fill_op}" stroke="{C["acc_d"]}" stroke-width="0.9" stroke-dasharray="2.5 1.5"/>'
for k in ('SEOULTECH', 'SWU', 'INDUK', 'KWANGWOON', 'S-DBC', 'KW-STATION'):
    add(node_poly(k, 0.18))
def photo(x, y, w, h, uri, gid):
    return (f'<g id="{gid}"><rect x="{f(x - 1.5)}" y="{f(y - 1.5)}" width="{f(w + 3)}" height="{f(h + 3)}" fill="#FFFFFF"/>'
            f'<image x="{f(x)}" y="{f(y)}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" xlink:href="{uri}"/></g>')
LBL = {  # key: (text, dx, dy, photo file, photo side)
    'SEOULTECH': ('서울과학기술대학교', 26, -48, 'photos/과기대.webp'),
    'SWU': ('서울여자대학교', -60, -40, 'photos/서울여대.jpg'),
    'INDUK': ('인덕대학교', -8, -44, 'photos/인덕대.jpg'),
    'KWANGWOON': ('광운대학교', 14, 30, 'photos/광운대.jpg'),
}
for k, (s_, dx, dy, ph) in LBL.items():
    cx, cy = T(*N[k]['center'])
    lx, ly = cx + dx, cy + dy
    add(f'<path d="M{f(cx)} {f(cy)} L{f(lx)} {f(ly)}" stroke="{C["acc_d"]}" stroke-width="0.6"/>')
    add(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="1.8" fill="{C["acc_d"]}"/>')
    add(photo(lx, ly - 36, 52, 32, img_uri(ph, (52, 32), (208, 128)), 'ph_' + k))
    add(label_box(lx, ly + 7, s_))
for k, s_, dx, dy in (('S-DBC', 'S-DBC  창동 개발 거점', 30, 6), ('KW-STATION', '광운대역세권 개발', -18, 40)):
    cx, cy = T(*N[k]['center'])
    add(f'<circle cx="{f(cx)}" cy="{f(cy)}" r="1.8" fill="{C["acc_d"]}"/>')
    add(f'<path d="M{f(cx)} {f(cy)} L{f(cx + dx)} {f(cy + dy)}" stroke="{C["acc_d"]}" stroke-width="0.6"/>')
    add(label_box(cx + dx, cy + dy + 4, s_, dark=True))
# 공릉역 (7호선) — position estimated
gx, gy = T(*GONGNEUNG_STN)
add(f'<g id="stn_gongneung"><circle cx="{f(gx)}" cy="{f(gy)}" r="4" fill="#FFFFFF" stroke="{C["acc_d"]}" stroke-width="1.2"/>'
    f'<text x="{f(gx)}" y="{f(gy + 2.3)}" font-family="{FONT}" font-size="5.5" font-weight="700" fill="{C["acc_d"]}" text-anchor="middle">7</text></g>')
add(f'<path d="M{f(gx - 4)} {f(gy - 3)} L{f(gx - 40)} {f(gy - 46)}" stroke="{C["acc_d"]}" stroke-width="0.6"/>')
add(photo(gx - 92, gy - 86, 52, 32, img_uri('photos/공릉역.jpg', (52, 32), (208, 128)), 'ph_GONGNEUNG'))
add(label_box(gx - 92, gy - 44, '공릉역 (7호선)'))
add(label_box(sx + 10, sy + 22, 'SITE  공릉동 714', 7.6, dark=True))
# place names (light)
jx, jy = T(214080, 559900)
add(text(jx, jy, '중\n랑\n천', 7.5, C['acc_d'], 600, opacity=0.8, ls=1.15))
fx_, fy_ = T(215300, 558690)
add(text(fx_, fy_, '경춘선숲길', 6.5, C['acc_d'], 600, opacity=0.9))
dx_, dy_ = T(214790, 558350)
add(text(dx_, dy_, '동일로', 6.5, C['acc_d'], 600, opacity=0.9))
# Sahmyook, off map
add(f'<path d="M{W - 14} 300 l-8 -4 v8 Z" fill="{C["acc_d"]}"/>')
add(text(W - 26, 303, '삼육대학교 1.5 km →', 6.2, C['acc_d'], 600, 'end'))
add('</g>')

# ======== TOP: title + street photos ========
add('<g id="HEADER">')
add(f'<rect x="0" y="0" width="214" height="{f(MAP_BOTTOM)}" fill="#FFFFFF" opacity="0.92"/>')
add(text(14, 26, 'understanding the node', 13, C['acc'], 500, spacing=-0.2))
add(text(14, 40, '2050 · S-DBC와 대학 클러스터 사이의 결절점', 6.8, C['ink2'], 500))
PH = [('남측 도로', 'photos/남측도로.jpg', ((SITE_TRI[1][0] + SITE_TRI[2][0]) / 2, (SITE_TRI[1][1] + SITE_TRI[2][1]) / 2 - 12)),
      ('경춘선숲길', 'photos/경춘선숲길.jpg', ((SITE_TRI[2][0] + SITE_TRI[0][0]) / 2 + 10, (SITE_TRI[2][1] + SITE_TRI[0][1]) / 2 + 8)),
      ('동일로', 'photos/동일로.jpg', ((SITE_TRI[0][0] + SITE_TRI[1][0]) / 2 - 12, (SITE_TRI[0][1] + SITE_TRI[1][1]) / 2))]
pw, ph_, gap = 108, 74, 7
px0 = W - 14 - 3 * pw - 2 * gap
for i, (cap, fn, loc) in enumerate(PH):
    x = px0 + i * (pw + gap); y = 26
    add(text(x, y - 6, cap, 6.8, C['ink'], 600, style='italic'))
    add(photo(x, y, pw, ph_, img_uri(fn, (pw, ph_), (pw * 4, ph_ * 4), trim=False), f'street_{i + 1}'))
    tx, ty = T(*loc)
    add(f'<path d="M{f(x + pw / 2)} {f(y + ph_ + 1.5)} L{f(tx)} {f(ty)}" stroke="{C["acc_d"]}" stroke-width="0.55" opacity="0.85"/>'
        f'<circle cx="{f(tx)}" cy="{f(ty)}" r="2.2" fill="#FFFFFF" stroke="{C["acc_d"]}" stroke-width="0.8"/>')
    add(text(x + pw / 2, y + ph_ + 10, f'0{i + 1}', 6, C['acc_d'], 700, 'middle'))
add('</g>')

# ======== LEFT COLUMN: text + mini diagrams ========
add('<g id="LEFT_PANELS">')
LX, LW = 14, 186
add(text(LX, 64, '대상지는 노원구 공릉동 714 일대,\n동일로·경춘선숲길·남측 도로로 둘러싸인\n삼각형 필지(약 2.1 ha)다. 반경 2 km 안에\n대학 4곳과 두 개의 개발 거점(S-DBC,\n광운대역세권)이 놓여 있다.', 6.6, C['ink'], 400, ls=1.42))

def mini(x, y, w, h, title, body, kind):
    s2 = min(w / (X1 - X0), h / (Y1 - Y0))
    ox = x + (w - (X1 - X0) * s2) / 2; oy = y
    tf = lambda X, Y: (ox + (X - X0) * s2, oy + (Y1 - Y) * s2)
    hh = (Y1 - Y0) * s2
    o = [f'<g id="mini_{kind}"><rect x="{f(ox)}" y="{f(oy)}" width="{f((X1 - X0) * s2)}" height="{f(hh)}" fill="{C["bg"]}"/>']
    if kind == 'connect':
        o.append(f'<path d="{lines_path(("인도",), tf, 3)}" fill="none" stroke="{C["ink2"]}" stroke-width="0.25"/>')
        o.append(f'<path d="{lines_path(("A015",), tf, 4)}" fill="none" stroke="{C["acc"]}" stroke-width="1.4"/>')
        a, b = tf(*SITE_C); o.append(f'<circle cx="{f(a)}" cy="{f(b)}" r="2.6" fill="{C["acc_d"]}"/>')
    elif kind == 'water':
        o.append(f'<path d="{mask_paths(M["green"], 8, 1.0, tf, 6)}" fill="{C["acc_m"]}" fill-opacity="0.55" fill-rule="evenodd"/>')
        o.append(f'<path d="{mask_paths(M["water"], 4, 1.0, tf, 3)}" fill="{C["acc_d"]}" fill-rule="evenodd"/>')
        a, b = tf(*SITE_C); o.append(f'<circle cx="{f(a)}" cy="{f(b)}" r="2.6" fill="none" stroke="{C["ink"]}" stroke-width="1"/>')
    elif kind == 'hotspot':
        o.append(f'<path d="{lines_path(("인도",), tf, 3)}" fill="none" stroke="#FFFFFF" stroke-width="0.35"/>')
        cs_ = {k: tf(*N[k]['center']) for k in N}
        for a_, b_ in (('SITE', 'S-DBC'), ('SITE', 'KW-STATION'), ('SITE', 'SEOULTECH'), ('SITE', 'SWU'), ('SITE', 'INDUK'), ('SITE', 'KWANGWOON'),
                       ('INDUK', 'SEOULTECH'), ('SEOULTECH', 'SWU'), ('SWU', 'KWANGWOON'), ('KWANGWOON', 'INDUK')):
            dash = '' if 'SITE' in (a_, b_) else ' stroke-dasharray="1.5 1.5"'
            o.append(f'<path d="M{f(cs_[a_][0])} {f(cs_[a_][1])} L{f(cs_[b_][0])} {f(cs_[b_][1])}" stroke="{C["acc_d"]}" stroke-width="0.7"{dash}/>')
        jn = [tf(*p) for p in JN]
        o.append(f'<path d="{pts_d(jn)}" fill="none" stroke="{C["acc"]}" stroke-width="1.6" stroke-dasharray="3 2"/>')
        for k in N:
            r_ = math.sqrt(N[k]['area'] / math.pi) * s2
            new = k in ('S-DBC', 'KW-STATION', 'SITE')
            o.append(f'<circle cx="{f(cs_[k][0])}" cy="{f(cs_[k][1])}" r="{f(max(r_, 2.2))}" fill="{C["acc_d"] if new else C["acc_m"]}" fill-opacity="{0.85 if new else 0.6}"/>')
    elif kind == 'locator':
        o.append(f'<path d="{lines_path(("인도",), tf, 3)}" fill="none" stroke="{C["ink2"]}" stroke-width="0.25" opacity="0.8"/>')
        a, b = tf(FX0 / 1, 0)  # placeholder (unused)
        fx0, fy0 = tf(SITE_C[0] + (FX0 - SX) / S, SITE_C[1] - (FY0 - SY) / S)
        o.append(f'<rect x="{f(fx0)}" y="{f(fy0)}" width="{f(FS / S * s2)}" height="{f(FS / S * s2)}" fill="{C["acc"]}" fill-opacity="0.25" stroke="{C["acc_d"]}" stroke-width="0.6" stroke-dasharray="2 1.5"/>')
        a, b = tf(*SITE_C)
        o.append(f'<path d="M{f(a - 7)} {f(b)} H{f(a + 7)} M{f(a)} {f(b - 7)} V{f(b + 7)}" stroke="{C["acc_d"]}" stroke-width="1.6"/>')
    o.append('</g>')
    o.append(text(x, y + hh + 11, title, 6.9, C['ink'], 700))
    o.append(text(x, y + hh + 21, body, 5.7, C['ink2'], 400, ls=1.4))
    return ''.join(o)
mw, mh = 88, 106
add(mini(LX, 118, mw, mh, '대상지 위치', '노원구 수치지형도 범위\n(4.5 × 5.4 km)', 'locator'))
add(mini(LX + mw + 10, 118, mw, mh, '교통 연결망', '동일로(7호선) · 1호선 ·\n경춘선 철도', 'connect'))
add(mini(LX, 272, mw, mh, '수계 · 녹지', '중랑천 · 당현천과\n동측 불암산 · 태릉 녹지', 'water'))
add(mini(LX + mw + 10, 272, mw, mh, '2050 거점 네트워크', '원 면적 = 부지 면적(추정)\n점선 = 중랑천 UAM 간선(가정)', 'hotspot'))
add(text(LX, 440, '대학 클러스터와 개발 거점', 7.6, C['acc'], 700))
add(text(LX, 453, '기존 거점: 서울과기대 · 서울여대 · 인덕대 · 광운대\n개발 거점(2050): S-DBC(창동) · 광운대역세권\n\n대상지는 동일로와 경춘선숲길이 만나는 자리에서\n두 거점군을 잇는 결절점이 된다. 대학의 교육·연구\n인구와 S-DBC의 일자리 인구가 이곳에서 일하고,\n살고, 머문다.', 6.3, C['ink'], 400, ls=1.45))
add(text(LX, 576, '대상지에서 거점까지', 7.6, C['acc'], 700))
DST = [('SEOULTECH', '서울과학기술대학교'), ('KW-STATION', '광운대역세권'), ('INDUK', '인덕대학교'), ('KWANGWOON', '광운대학교'),
       ('SWU', '서울여자대학교'), ('S-DBC', 'S-DBC (창동)')]
rows = sorted(((math.dist(SITE_C, N[k]['center']) / 1000, n_) for k, n_ in DST))
for i, (d_, n_) in enumerate(rows):
    y = 592 + i * 15
    add(f'<path d="M{LX} {y + 4} H{LX + LW}" stroke="{C["acc_l"]}" stroke-width="0.6"/>')
    add(text(LX, y, n_, 6.4, C['ink'], 500))
    bw = d_ / 3.2 * 70
    add(f'<rect x="{f(LX + LW - 104)}" y="{y - 5}" width="{f(bw)}" height="5" fill="{C["acc_m"]}"/>')
    add(text(LX + LW, y, f'{d_:.1f} km', 6.4, C['acc_d'], 700, 'end'))
add(text(LX, 592 + 6 * 15 + 4, '직선거리, 부지 중심 기준', 5.4, C['ink2']))
add('</g>')

# ======== BOTTOM BAND: challenges + vision + legend ========
add('<g id="BOTTOM">')
BY = MAP_BOTTOM
add(f'<rect x="0" y="{f(BY)}" width="{W}" height="{f(H - BY)}" fill="#FFFFFF"/>')
add(f'<path d="M14 {f(BY + 0.5)} H{W - 14}" stroke="{C["acc"]}" stroke-width="0.8"/>')
add(text(14, BY + 20, '+challenges?', 11, C['ink'], 700))
CH = [('닫힌 캠퍼스, 흩어진 교류', '대학 네 곳이 각자의 담장 안에 머물러\n학생 · 연구 인구가 지역과 섞일 곳이 적다.', aerial_crop('과기대.png', 215650, 559150, 230, (300, 200))),
      ('일상 공유 공간의 부족', '주거 밀집지에 함께 일하고 먹고 쉬는\n공용 공간이 부족하다.', aerial_crop('사이트.png', 214990, 558915, 140, (300, 200))),
      ('지나치는 가로, 머물지 않는 숲길', '동일로와 경춘선숲길이 대상지를\n통과만 할 뿐 머무는 장소가 없다.', img_uri('photos/경춘선숲길.jpg', (3, 2), (300, 200), trim=False))]
cw = 140
for i, (t_, b_, uri) in enumerate(CH):
    x = 14 + i * (cw + 8); y = BY + 30
    add(photo(x, y, cw, 62, uri, f'challenge_{i + 1}'))
    add(text(x, y + 74, t_, 7, C['acc_d'], 700, style='italic'))
    add(text(x, y + 85, b_, 5.8, C['ink2'], 400, ls=1.4))
VX = 14 + 3 * (cw + 8) + 6
add(text(VX, BY + 20, '+vision', 11, C['ink'], 700))
def icon(kind, cx, cy):
    a = C['acc']; d = C['acc_d']
    if kind == 'node':
        s = ''.join(f'<ellipse cx="{cx}" cy="{cy}" rx="{r}" ry="{r * .42}" fill="none" stroke="{a}" stroke-width="0.8" opacity="{1 - r / 30}"/>' for r in (6, 11, 16))
        s += ''.join(f'<path d="M{f(cx + 18 * math.cos(t))} {f(cy + 7.5 * math.sin(t))} L{f(cx + 25 * math.cos(t))} {f(cy + 10.5 * math.sin(t))}" stroke="{d}" stroke-width="0.9"/>' for t in np.linspace(0, 2 * math.pi, 9)[:-1])
        return s + f'<ellipse cx="{cx}" cy="{cy}" rx="3" ry="1.3" fill="{d}"/>'
    if kind == 'stack':
        s = ''
        for j, (ox, oy) in enumerate(((-9, 6), (9, 6), (0, -4))):
            x, y = cx + ox, cy + oy
            s += (f'<path d="M{x} {y - 6} l7 3.5 l-7 3.5 l-7 -3.5 Z" fill="{C["acc_l"]}" stroke="{d}" stroke-width="0.6"/>'
                  f'<path d="M{x - 7} {y - 2.5} v7 l7 3.5 v-7 Z" fill="{C["acc_m"]}" stroke="{d}" stroke-width="0.6"/>'
                  f'<path d="M{x + 7} {y - 2.5} v7 l-7 3.5 v-7 Z" fill="{a}" stroke="{d}" stroke-width="0.6"/>')
        return s
    if kind == 'ground':
        return (f'<path d="M{cx - 24} {cy + 4} L{cx} {cy - 7} L{cx + 24} {cy + 4} L{cx} {cy + 15} Z" fill="{C["acc_l"]}" stroke="{d}" stroke-width="0.6"/>'
                f'<path d="M{cx - 14} {cy + 9} C{cx - 4} {cy + 1} {cx + 4} {cy + 8} {cx + 14} {cy - 1}" fill="none" stroke="{a}" stroke-width="1.4" stroke-dasharray="2 1.4"/>'
                + ''.join(f'<circle cx="{cx + dx}" cy="{cy + dy}" r="2.4" fill="{C["acc_m"]}"/>' for dx, dy in ((-10, 0), (8, 6), (2, -2))))
    if kind == 'hub':
        return (f'<ellipse cx="{cx}" cy="{cy + 6}" rx="20" ry="7" fill="none" stroke="{a}" stroke-width="0.8" stroke-dasharray="2 1.5"/>'
                f'<ellipse cx="{cx}" cy="{cy + 6}" rx="9" ry="3.2" fill="{C["acc_l"]}" stroke="{d}" stroke-width="0.7"/>'
                f'<path d="M{cx - 22} {cy - 10} Q{cx} {cy - 22} {cx + 22} {cy - 10}" fill="none" stroke="{d}" stroke-width="0.9" stroke-dasharray="3 2"/>'
                f'<path d="M{cx} {cy + 6} V{cy - 14}" stroke="{d}" stroke-width="0.7"/><circle cx="{cx}" cy="{cy - 15}" r="2" fill="{d}"/>')
VI = [('node', '대학 · 일자리 거점을\n잇는 결절점'), ('stack', '일하고 · 살고 · 머무는\n직주락(職住樂) 복합'),
      ('ground', '숲길과 이어진\n열린 저층부(어반커먼)'), ('hub', 'UAM · 대중교통\n환승 거점')]
vw = (W - 14 - VX) / 2
for i, (k, t_) in enumerate(VI):
    x = VX + (i % 2) * vw; y = BY + 30 + (i // 2) * 60
    add(f'<g id="vision_{i + 1}">' + icon(k, x + 26, y + 20) + '</g>')
    add(text(x + 58, y + 17, t_, 6.4, C['ink'], 500, ls=1.4))
# scale bar, north, legend, source
sbx, sby = 14, H - 10
add(f'<path d="M{sbx} {sby} H{f(sbx + 1000 * S)} M{sbx} {sby - 3} V{sby} M{f(sbx + 500 * S)} {sby - 3} V{sby} M{f(sbx + 1000 * S)} {sby - 3} V{sby}" stroke="{C["ink"]}" stroke-width="0.7" fill="none"/>')
add(f'<rect x="{sbx}" y="{sby - 1.6}" width="{f(500 * S)}" height="1.6" fill="{C["ink"]}"/>')
add(text(sbx, sby - 5, '0', 5.4, C['ink'])); add(text(sbx + 500 * S, sby - 5, '500', 5.4, C['ink'], anchor='middle')); add(text(sbx + 1000 * S + 3, sby - 5, '1 km', 5.4, C['ink']))
nx_ = sbx + 1000 * S + 32
add(f'<path d="M{f(nx_)} {sby - 14} l4 12 l-4 -3 l-4 3 Z" fill="{C["ink"]}"/>' + text(nx_, sby + 0.5, 'N', 5.4, C['ink'], 700, 'middle'))
lgx = nx_ + 20
for j, (sw, lab) in enumerate(((f'<path d="M{{x}} {{y}} l5 -4 l5 4 Z" fill="{C["acc_d"]}"/>', '대상지'),
                               (f'<rect x="{{x}}" y="{{y0}}" width="10" height="6" fill="{C["acc"]}" opacity="0.4"/>', '분석 범위 2.8 km'),
                               (f'<rect x="{{x}}" y="{{y0}}" width="10" height="6" fill="{C["acc_m"]}" fill-opacity="0.3" stroke="{C["acc_d"]}" stroke-width="0.6" stroke-dasharray="2 1"/>', '거점 부지(추정)'),
                               (f'<path d="M{{x}} {{y}} h10" stroke="{C["ink2"]}" stroke-width="1.2"/>', '철도'))):
    x = lgx + j * 80
    add(sw.format(x=f(x), y=f(sby - 1), y0=f(sby - 6)) + text(x + 14, sby, lab, 5.6, C['ink']))
add(text(W - 14, sby, '자료: 국토지리정보원 수치지형도 · 항공사진(출처 기입) · 거리 사진 팀 촬영(2026.10) · 대학 사진(출처 기입)', 4.8, C['ink2'], anchor='end'))
add('</g>')
add('</svg>')
open(OUT, 'w', encoding='utf-8').write('\n'.join(el))
print('saved', OUT, round(sum(len(e) for e in el) / 1e6, 2), 'MB')
