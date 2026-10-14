"""'Network between hot-spots' sheet: black linework, area-true circles, two networks, UAM corridor + branch."""
import json, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = sys.argv[1] if len(sys.argv) > 1 else 'network.png'
FONT = 'Inter'
INK, INK2 = '#222222', '#8A8A8A'
BEIGE, BLUEG = '#F2D6A2', '#8E95B8'
ORANGE, NAVY, GREEN = '#E07B39', '#17204A', '#7D9B3A'

geo = pickle.load(open('geo.pkl', 'rb'))
N = json.load(open('nodes_area.json'))
X0, Y0, X1, Y1, R = np.load('meta.npy')
N['SITE']['area'] = 21222.0                         # official site area (decision log)
N['SITE']['center'] = [214960.0, 558915.0]

EXISTING = {'SEOULTECH': 'SEOULTECH', 'KWANGWOON': 'KWANGWOON UNIV.', 'INDUK': 'INDUK UNIV.', 'SWU': "SEOUL WOMEN'S UNIV."}
NEW = {'S-DBC': 'S-DBC', 'KW-STATION': 'KWANGWOON STN.\nDISTRICT', 'SITE': 'SITE'}
C = {k: np.array(v['center']) for k, v in N.items()}
RAD = {k: np.sqrt(v['area'] / np.pi) for k, v in N.items()}   # circle area = site area

# ---------- page (A4 portrait @300dpi) ----------
PW, PH = 2480, 3508
fig = plt.figure(figsize=(PW / 300, PH / 300), dpi=300)
fig.patch.set_facecolor('white')
mw = 0.80; mh = mw * PW * (Y1 - Y0) / (X1 - X0) / PH
ax = fig.add_axes([(1 - mw) / 2, 0.215, mw, mh])
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect('equal'); ax.axis('off')

base = np.load('base_real.npy')
ax.imshow(base, extent=(X0, X0 + base.shape[1], Y1 - base.shape[0], Y1), origin='upper', zorder=0,
          interpolation='lanczos')
ax.add_patch(matplotlib.patches.Rectangle((X0, Y0), X1 - X0, Y1 - Y0, facecolor='none', edgecolor=INK, lw=0.6, zorder=9))

# ---------- circles ----------
for k in EXISTING:
    ax.add_patch(Circle(C[k], RAD[k], facecolor=BEIGE, edgecolor='none', alpha=0.62, zorder=3))
    ax.add_patch(Circle(C[k], RAD[k], facecolor='none', edgecolor='#C9963F', lw=0.6, zorder=3))
for k in NEW:
    ax.add_patch(Circle(C[k], RAD[k], facecolor=BLUEG, edgecolor='none', alpha=0.6, zorder=3))
    ax.add_patch(Circle(C[k], RAD[k], facecolor='none', edgecolor='#5B6390', lw=0.6, zorder=3))
ax.add_patch(Polygon(N['SITE']['poly'], closed=True, facecolor='none', edgecolor=NAVY, lw=0.6, zorder=6))

# ---------- networks ----------
def link(a, b, color, lw, z):
    ax.plot([C[a][0], C[b][0]], [C[a][1], C[b][1]], color=color, lw=lw, solid_capstyle='round', zorder=z)
for a, b in [('INDUK', 'SEOULTECH'), ('SEOULTECH', 'SWU'), ('SWU', 'KWANGWOON'), ('KWANGWOON', 'INDUK')]:
    link(a, b, ORANGE, 2.2, 4)
for a, b in [('SITE', 'S-DBC'), ('SITE', 'KW-STATION'), ('S-DBC', 'KW-STATION'),
             ('SITE', 'SEOULTECH'), ('SITE', 'SWU'), ('SITE', 'INDUK'), ('SITE', 'KWANGWOON')]:
    link(a, b, NAVY, 0.9, 5)

# ---------- UAM corridor along Jungnangcheon (river centreline) + branch to the SITE ----------
# Jungnangcheon centreline: waypoints read off the base map (main channel, north -> south)
JN = np.array([(213421, 561821), (213489, 561143), (213557, 560690), (213648, 560238), (213807, 559786),
               (214011, 559424), (214193, 559197), (214329, 558971), (214510, 558745)], float)
def smooth(P, n=400):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])]
cl = smooth(JN)
k = 41
for c in (0, 1):
    cl[:, c] = np.convolve(np.pad(cl[:, c], k // 2, mode='edge'), np.ones(k) / k, mode='valid')
cl = cl[::-1]   # south -> north, matching trunk order below
trunk = np.vstack([C['KW-STATION'], cl, C['S-DBC']])
ax.plot(trunk[:, 0], trunk[:, 1], color=GREEN, lw=2.6, ls=(0, (2.2, 1.4)), zorder=6, solid_capstyle='butt')

# branch: leave the corridor at the point nearest the site, arc into the rooftop vertiport
j = np.argmin(np.linalg.norm(cl - C['SITE'], axis=1))
p0, p2 = cl[j], C['SITE']
mid = (p0 + p2) / 2; nrm = np.array([-(p2 - p0)[1], (p2 - p0)[0]]); nrm /= np.linalg.norm(nrm)
p1 = mid + nrm * 260
t = np.linspace(0, 1, 60)[:, None]
bez = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
ax.plot(bez[:, 0], bez[:, 1], color=GREEN, lw=1.6, ls=(0, (2.2, 1.4)), zorder=6)
for hub in ('S-DBC', 'KW-STATION'):
    ax.plot(*C[hub], marker='o', ms=4.5, mfc='white', mec=GREEN, mew=1.2, zorder=7)
ax.plot(*C['SITE'], marker='o', ms=4.5, mfc='white', mec=GREEN, mew=1.2, zorder=7)

# ---------- labels ----------
def label(k, text, dx, dy, ha='left', size=5.6, weight='semibold'):
    ax.text(C[k][0] + dx, C[k][1] + dy, text, fontsize=size, color=INK, family=FONT, fontweight=weight,
            ha=ha, va='center', zorder=8, linespacing=1.1,
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))
label('SEOULTECH', EXISTING['SEOULTECH'], RAD['SEOULTECH'] * 0.75, RAD['SEOULTECH'] * 0.75)
label('KWANGWOON', EXISTING['KWANGWOON'], -RAD['KWANGWOON'] - 60, -160, ha='right')
label('INDUK', EXISTING['INDUK'], -RAD['INDUK'] - 40, 120, ha='right')
label('SWU', EXISTING['SWU'], 0, -RAD['SWU'] - 120, ha='center')
label('S-DBC', NEW['S-DBC'], RAD['S-DBC'] + 40, 0, size=6.4, weight='bold')
label('KW-STATION', NEW['KW-STATION'], RAD['KW-STATION'] + 50, -120)
label('SITE', NEW['SITE'], 130, -170, size=6.4, weight='bold')

# off-map node: Sahmyook Univ. (east)
SAHM = np.array([217949.0, 560381.0])
d = (SAHM - C['SEOULTECH']) / np.linalg.norm(SAHM - C['SEOULTECH'])
end = C['SEOULTECH'] + d * (X1 - 60 - C['SEOULTECH'][0]) / d[0]
ax.plot([C['SEOULTECH'][0], end[0]], [C['SEOULTECH'][1], end[1]], color=ORANGE, lw=2.2, ls=(0, (1, 1.2)), zorder=4)
ax.add_patch(FancyArrowPatch(end - d * 80, end, arrowstyle='-|>', mutation_scale=9, color=ORANGE, lw=1, zorder=5))
ax.text(end[0] - 20, end[1] + 70, 'SAHMYOOK UNIV.', fontsize=5, color=INK, family=FONT, ha='right', zorder=8)

# scale bar + north
sx, sy = X1 - 1300, Y0 + 180
ax.plot([sx, sx + 1000], [sy, sy], color=INK, lw=0.6, zorder=8)
for m in (0, 500, 1000):
    ax.plot([sx + m, sx + m], [sy, sy + 40], color=INK, lw=0.6, zorder=8)
    ax.text(sx + m, sy + 70, f'{m}', fontsize=4.2, family=FONT, color=INK, ha='center', va='bottom', zorder=8)
ax.text(sx + 1060, sy, 'm', fontsize=4.2, family=FONT, color=INK, va='center', zorder=8)
ax.annotate('', xy=(X1 - 120, Y0 + 520), xytext=(X1 - 120, Y0 + 300),
            arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.7), zorder=8)
ax.text(X1 - 120, Y0 + 570, 'N', fontsize=6, family=FONT, color=INK, ha='center', zorder=8)

# ---------- title ----------
fig.text(0.5, 0.955, 'Network between Hot-spots', fontsize=15, family=FONT, fontweight='light', color=INK, ha='center')
fig.text(0.5, 0.935, '2050  ·  S-DBC × University Cluster  ·  The site as node', fontsize=7, family=FONT,
         color=INK2, ha='center')

# ---------- legend ----------
lx1, lx2, ly = 0.17, 0.53, 0.165
def leg_circle(x, y, color, alpha, text):
    fig.patches.append(matplotlib.patches.Ellipse((x, y), 0.022, 0.022 * PW / PH, transform=fig.transFigure, facecolor=color, alpha=alpha, edgecolor='none'))
    fig.text(x + 0.03, y, text, fontsize=6.4, family=FONT, color=INK, va='center')
def leg_line(x, y, color, lw, ls, text):
    fig.lines.append(Line2D([x - 0.012, x + 0.05], [y, y], transform=fig.transFigure, color=color, lw=lw, ls=ls))
    fig.text(x + 0.065, y, text, fontsize=6.4, family=FONT, color=INK, va='center')
leg_circle(lx1, ly, BEIGE, 0.9, 'Existing hot-spots  (universities)')
leg_circle(lx1, ly - 0.035, BLUEG, 0.8, 'Hot-spots to be developed  (2050)')
fig.text(lx1 - 0.011, ly - 0.07, 'Circle area = site area (estimated, ±20%)', fontsize=5.4, family=FONT, color=INK2, va='center')
leg_line(lx2, ly, ORANGE, 2.2, '-', 'Network between existing hot-spots')
leg_line(lx2, ly - 0.035, NAVY, 0.9, '-', 'Network through the site')
leg_line(lx2, ly - 0.07, GREEN, 2.6, (0, (2.2, 1.4)), 'UAM corridor along Jungnangcheon  (assumed)')
leg_line(lx2, ly - 0.105, GREEN, 1.6, (0, (2.2, 1.4)), 'UAM branch to the site vertiport  (assumed)')

fig.savefig(OUT, dpi=300, facecolor='white')
print('saved', OUT)
