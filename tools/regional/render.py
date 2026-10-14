"""Regional node collage in the style of the reference (navy linework, stepped tiles, inverted aerial cells)."""
import json, pickle, sys
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else 'regional_collage.png'
SHOW_LINKS = '--no-links' not in sys.argv

# ---------- palette ----------
BG     = '#E9EDF5'   # pale blue-grey ground
PANEL  = '#FFFFFF'   # stepped white panel
NAVY   = '#2B3A78'   # linework
NAVY_L = '#5A67A0'   # secondary lines
GRIDC  = '#9AA4C8'
DARK   = np.array([20, 28, 66]) / 255.0     # duotone shadow
LIGHT  = np.array([214, 222, 244]) / 255.0  # duotone highlight
FONT   = 'Inter'

geo = pickle.load(open('geo.pkl', 'rb'))
X0, Y0, X1, Y1, R = np.load('meta.npy')
H, W = int(Y1 - Y0) + 1, int(X1 - X0) + 1
SITE = np.array([214960.0, 558915.0])          # site triangle centroid
CELL = 250.0
ox, oy = SITE - CELL / 2                       # grid aligned so the site sits in one cell

def cells_in(rect, frac=0.0):
    """cells (i,j) whose area overlaps rect by more than frac."""
    x0, x1, y0, y1 = rect
    out = []
    for i in range(int(np.floor((x0 - ox) / CELL)), int(np.ceil((x1 - ox) / CELL))):
        for j in range(int(np.floor((y0 - oy) / CELL)), int(np.ceil((y1 - oy) / CELL))):
            cx0, cy0 = ox + i * CELL, oy + j * CELL
            ov = max(0, min(x1, cx0 + CELL) - max(x0, cx0)) * max(0, min(y1, cy0 + CELL) - max(y0, cy0))
            if ov / CELL**2 > frac:
                out.append((i, j))
    return out

# ---------- nodes (map metres, EPSG:5186 with the file's x-offset) ----------
NODES = {
    'SITE':       dict(rect=(214870, 215090, 558830, 559042), label='SITE', sub='GONGNEUNG 714'),
    'SEOULTECH':  dict(rect=(215180, 215900, 559000, 559620), label='SEOULTECH', sub='Seoul Nat. Univ. of Sci. & Tech.'),
    'KWANGWOON':  dict(rect=(213640, 214110, 557560, 557940), label='KWANGWOON UNIV.', sub=''),
    'INDUK':      dict(rect=(213260, 213760, 558790, 559300), label='INDUK UNIV.', sub=''),
    'SWU':        dict(rect=(216660, 217070, 558480, 558870), label="SEOUL WOMEN'S UNIV.", sub=''),
    'S-DBC':      dict(rect=(213250, 213950, 561550, 562500), label='S-DBC', sub='Seoul Digital Bio City (planned)'),
}
PHOTO_NODES = ['SITE', 'SEOULTECH', 'KWANGWOON', 'INDUK', 'SWU']

# ---------- aerial mosaic in raster space (1 m/px) ----------
G = json.load(open('georef.json'))
A_site = np.array(json.load(open('affine.json'))['사이트.png'])
mosaic = np.zeros((H, W), np.float32)
have = np.zeros((H, W), bool)
for f in ['사이트_주변.png', '과기대.png', '인덕대.png', '광운대.png', '서울여대.png', '사이트.png']:
    im = cv2.imread('img/' + f, cv2.IMREAD_UNCHANGED)[:, :, :3]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    h, w = g.shape
    if f == '사이트.png':
        M = A_site
    else:
        x0, x1, y0, y1 = G[f]
        M = np.array([[(x1 - x0) / w, 0, x0 - X0], [0, (y1 - y0) / h, Y1 - y1]])
    warped = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_AREA)
    m = cv2.warpAffine(np.ones_like(g), M, (W, H), flags=cv2.INTER_NEAREST) > 0.5
    mosaic[m] = warped[m]
    if f != '사이트_주변.png':
        have |= m
# local contrast then invert -> dark ground, light figures (reference look)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
eq = clahe.apply((mosaic * 255).astype(np.uint8)).astype(np.float32) / 255
inv = 1 - eq
inv = np.clip((inv - 0.25) / 0.75, 0, 1) ** 1.6       # keep it mostly dark
duo = DARK[None, None, :] * (1 - inv[..., None]) + LIGHT[None, None, :] * inv[..., None]

# ---------- which cells ----------
PC = CELL / 2   # photo sub-cells (125 m) so the stepped photo edge follows the high-res coverage
def sub_cells(rect, frac):
    x0, x1, y0, y1 = rect; out = []
    for i in range(int(np.floor((x0 - ox) / PC)), int(np.ceil((x1 - ox) / PC))):
        for j in range(int(np.floor((y0 - oy) / PC)), int(np.ceil((y1 - oy) / PC))):
            cx0, cy0 = ox + i * PC, oy + j * PC
            ov = max(0, min(x1, cx0 + PC) - max(x0, cx0)) * max(0, min(y1, cy0 + PC) - max(y0, cy0))
            if ov / PC**2 > frac: out.append((i, j))
    return out
def cov(i, j, s):
    ya, yb = int(Y1 - (oy + (j + 1) * s)), int(Y1 - (oy + j * s))
    xa, xb = int(ox + i * s - X0), int(ox + (i + 1) * s - X0)
    if ya < 0 or xa < 0 or yb > H or xb > W: return 0
    return have[ya:yb, xa:xb].mean()
photo_sub = set()
for k in PHOTO_NODES:
    photo_sub |= {c for c in sub_cells(NODES[k]['rect'], 0.05) if cov(*c, PC) > 0.98}
sdbc_cells = set(cells_in(NODES['S-DBC']['rect'], 0.25))

alpha = np.zeros((H, W), np.float32)
for i, j in photo_sub:
    xa, xb = int(ox + i * PC - X0), int(ox + (i + 1) * PC - X0)
    ya, yb = int(Y1 - (oy + (j + 1) * PC)), int(Y1 - (oy + j * PC))
    alpha[ya:yb, xa:xb] = 1
rgba = np.dstack([duo, alpha])

# ---------- open-land stipple (no building / road / water within ~35 m) ----------
occ = np.zeros((H, W), np.uint8)
occ |= cv2.imread('bld.png', 0)
occ |= cv2.imread('road.png', 0)
def rp(p): return np.c_[p[:, 0] - X0, Y1 - p[:, 1]].astype(np.int32)
for k, v in geo.items():
    if k.startswith(('E', 'A015', 'C0536')):
        for p, _ in v:
            cv2.polylines(occ, [rp(p)], False, 255, 3)
dist = cv2.distanceTransform((occ == 0).astype(np.uint8), cv2.DIST_L2, 5)
open_ = cv2.morphologyEx((dist > 32).astype(np.uint8), cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
sp = 16
yy, xx = np.mgrid[sp // 2:H:sp, sp // 2:W:sp]
xx = xx + (np.arange(xx.shape[0])[:, None] % 2) * sp // 2   # staggered dots
xx = np.clip(xx, 0, W - 1)
sel = open_[yy, xx] > 0
dots = np.c_[X0 + xx[sel], Y1 - yy[sel]]

# ---------- figure ----------
DPI = 300
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# stepped white panel: cells within 2.0 km of the site
panel = []
for c in cells_in((X0, X1, Y0, Y1)):
    cx, cy = ox + (c[0] + .5) * CELL, oy + (c[1] + .5) * CELL
    if np.hypot(cx - SITE[0], cy - SITE[1]) < 2000:
        panel.append(c)
ax.add_collection(PolyCollection(
    [[(ox + i * CELL, oy + j * CELL), (ox + (i + 1) * CELL, oy + j * CELL),
      (ox + (i + 1) * CELL, oy + (j + 1) * CELL), (ox + i * CELL, oy + (j + 1) * CELL)] for i, j in panel],
    facecolor=PANEL, edgecolor='none', zorder=1))

# grid: thin every cell, slightly stronger every 1 km
for i in range(-40, 40):
    x = ox + i * CELL
    if X0 < x < X1:
        ax.plot([x, x], [Y0, Y1], color=GRIDC, lw=0.25 if i % 4 else 0.45, alpha=0.6 if i % 4 else 0.9, zorder=2)
for j in range(-40, 40):
    y = oy + j * CELL
    if Y0 < y < Y1:
        ax.plot([X0, X1], [y, y], color=GRIDC, lw=0.25 if j % 4 else 0.45, alpha=0.6 if j % 4 else 0.9, zorder=2)

# stipple
ax.scatter(dots[:, 0], dots[:, 1], s=0.9, c=NAVY, linewidths=0, alpha=0.6, zorder=3)

# linework
def lines(prefixes, lw, color, z, alpha=1.0):
    segs = [p for k, v in geo.items() if k.startswith(prefixes) for p, _ in v]
    ax.add_collection(LineCollection(segs, lw=lw, color=color, alpha=alpha, zorder=z, capstyle='round'))
lines(('B',), 0.12, NAVY, 4, 0.8)                       # buildings
lines(('C0536', 'C0393', 'C007', 'F004'), 0.12, NAVY_L, 4, 0.5)  # walls, retaining, misc
lines(('인도',), 0.22, NAVY, 5)                          # streets
lines(('A003', 'A009'), 0.22, NAVY, 5)                   # bridges / overpasses
lines(('E',), 0.45, NAVY, 5)                             # streams
lines(('A015',), 0.7, NAVY, 5)                           # rail

# photo cells (inverted aerial duotone)
ax.imshow(rgba, extent=(X0, X0 + W, Y1 - H, Y1), origin='upper', zorder=6, interpolation='lanczos')

# S-DBC cells: no aerial in the source set -> dense navy hatch block
for i, j in sdbc_cells:
    ax.add_patch(Rectangle((ox + i * CELL, oy + j * CELL), CELL, CELL, facecolor='#1F2A5E',
                           edgecolor='none', zorder=6))
    for t in np.arange(-CELL, CELL, 9):
        x0_, y0_ = ox + i * CELL, oy + j * CELL
        a = (max(0, t), max(0, -t)); b = (min(CELL, CELL + t), min(CELL, CELL - t))
        ax.plot([x0_ + a[0], x0_ + b[0]], [y0_ + a[1], y0_ + b[1]], color='#3C4A88', lw=0.25, zorder=6)

# light linework over the dark cells (clip to cells)
from matplotlib.path import Path
dark_cells = [(i, j, PC) for i, j in photo_sub] + [(i, j, CELL) for i, j in sdbc_cells]
verts, codes = [], []
for i, j, cs in dark_cells:
    x0_, y0_ = ox + i * cs, oy + j * cs
    verts += [(x0_, y0_), (x0_ + cs, y0_), (x0_ + cs, y0_ + cs), (x0_, y0_ + cs), (x0_, y0_)]
    codes += [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
clip = matplotlib.patches.PathPatch(Path(verts, codes), transform=ax.transData, facecolor='none', edgecolor='none')
ax.add_patch(clip)
for pre, lw, al in [(('인도',), 0.3, 0.75), (('E',), 0.45, 0.9), (('A015',), 0.55, 0.9), (('B',), 0.12, 0.35)]:
    segs = [p for k, v in geo.items() if k.startswith(pre) for p, _ in v]
    lc = LineCollection(segs, lw=lw, color='#EEF1FB', alpha=al, zorder=7)
    lc.set_clip_path(clip); ax.add_collection(lc)
# cell borders of dark cells
for i, j, cs in dark_cells:
    ax.add_patch(Rectangle((ox + i * cs, oy + j * cs), cs, cs, facecolor='none',
                           edgecolor='#C9D0EA', lw=0.3, zorder=7))

# site outline (white, crisp)
tri = np.array([[214869.5, 559042], [214933.4, 558829.5], [215088.8, 558866]])
ax.add_patch(matplotlib.patches.Polygon(tri, closed=True, facecolor='none', edgecolor='white', lw=1.1, zorder=9))

# concentric distance rings
for r, ls in [(1000, (0, (1, 3))), (2000, (0, (1, 3))), (3000, (0, (1, 4)))]:
    ax.add_patch(Circle(SITE, r, facecolor='none', edgecolor=NAVY, lw=0.6, ls=ls, alpha=1.0, zorder=8))
    ax.text(SITE[0] + r * 0.707 + 20, SITE[1] - r * 0.707 - 20, f'{r // 1000}km', fontsize=4.2,
            color=NAVY, family=FONT, zorder=10, ha='left', va='top')

# links: each node -> site (thin dashed), reads as convergence on the site
centers = {k: np.array([(v['rect'][0] + v['rect'][1]) / 2, (v['rect'][2] + v['rect'][3]) / 2]) for k, v in NODES.items()}
SAHM = np.array([217949.0, 560381.0])   # Sahmyook Univ. (outside the base map, east)
if SHOW_LINKS:
    for k, c in centers.items():
        if k == 'SITE':
            continue
        ax.plot([c[0], SITE[0]], [c[1], SITE[1]], color=NAVY, lw=0.55, ls=(0, (6, 3)), zorder=8.5, alpha=0.9)
    # off-map node
    d = (SAHM - SITE) / np.linalg.norm(SAHM - SITE)
    tedge = (X1 - 40 - SITE[0]) / d[0]
    end = SITE + d * tedge
    ax.plot([SITE[0], end[0]], [SITE[1], end[1]], color=NAVY, lw=0.55, ls=(0, (6, 3)), zorder=8.5, alpha=0.9)
    ax.add_patch(FancyArrowPatch(end - d * 60, end, arrowstyle='-|>', mutation_scale=6, color=NAVY, lw=0.6, zorder=9))
    dist_km = np.linalg.norm(SAHM - SITE) / 1000
    ax.text(end[0] - 30, end[1] + 40, f'SAHMYOOK UNIV.  {dist_km:.1f}km', fontsize=4.6, color=NAVY,
            family=FONT, ha='right', va='bottom', zorder=10)

# node markers + labels
def cross(x, y, s=55, color='white', lw=0.9):
    ax.plot([x - s, x + s], [y, y], color=color, lw=lw, zorder=10)
    ax.plot([x, x], [y - s, y + s], color=color, lw=lw, zorder=10)
for k, v in NODES.items():
    c = centers[k] if k != 'SITE' else SITE
    cross(*c)
    x1, y1 = v['rect'][1], v['rect'][3]
    big = k in ('SITE', 'S-DBC')
    ax.text(x1 + 25, y1 - 10, v['label'], fontsize=8 if big else 5.4, fontweight='bold' if big else 'semibold',
            color=NAVY, family=FONT, ha='left', va='top', zorder=10,
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    if v['sub']:
        ax.text(x1 + 25, y1 - (150 if big else 110), v['sub'], fontsize=3.8, color=NAVY, family=FONT,
                ha='left', va='top', zorder=10,
                bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

# compass letters at the frame edges (reference layout)
for lab, (x, y) in {'N': (SITE[0], Y1 - 140), 'S': (SITE[0], Y0 + 140), 'E': (X1 - 140, SITE[1]), 'W': (X0 + 140, SITE[1])}.items():
    ax.text(x, y, lab, fontsize=10, color=NAVY, family=FONT, ha='center', va='center', zorder=10,
            bbox=dict(boxstyle='square,pad=0.25', facecolor=BG, edgecolor='none'))

# scale bar (top-left)
sx, sy = X0 + 260, Y1 - 260
for i, m in enumerate([0, 250, 500, 750, 1000]):
    ax.plot([sx + m, sx + m], [sy, sy + 30], color=NAVY, lw=0.5, zorder=10)
    ax.text(sx + m, sy + 45, str(m), fontsize=3.8, color=NAVY, family=FONT, ha='center', va='bottom', zorder=10)
ax.plot([sx, sx + 1000], [sy, sy], color=NAVY, lw=0.5, zorder=10)
ax.text(sx + 1040, sy, 'm', fontsize=3.8, color=NAVY, family=FONT, va='center', zorder=10)

# caption (bottom-left)
ax.text(X0 + 260, Y0 + 240, '2050  ·  S-DBC × UNIVERSITY CLUSTER  —  THE SITE AS NODE',
        fontsize=6.2, fontweight='semibold', color=NAVY, family=FONT, zorder=10)
ax.text(X0 + 260, Y0 + 140, 'Base: digital topographic map  ·  Aerial: [source]  ·  Grid 250 m',
        fontsize=3.8, color=NAVY_L, family=FONT, zorder=10)

fig.savefig(OUT, dpi=DPI, facecolor=BG)
print('saved', OUT, len(photo_sub), 'photo sub-cells', len(sdbc_cells), 'sdbc cells')
