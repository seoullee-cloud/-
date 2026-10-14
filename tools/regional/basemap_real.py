"""Realistic-ish colored base raster (1 m/px) from the vector base map: paper, greenery, water, buildings with shadow."""
import pickle
import numpy as np, cv2

rng = np.random.default_rng(7)
geo = pickle.load(open('geo.pkl', 'rb'))
X0, Y0, X1, Y1, R = np.load('meta.npy')
H, W = int(Y1 - Y0) + 1, int(X1 - X0) + 1

def rp(p): return np.c_[p[:, 0] - X0, Y1 - p[:, 1]].astype(np.int32)
def noise(scale, octaves=4):
    out = np.zeros((H, W), np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        s = max(2, int(scale / 2**o))
        n = rng.random((H // s + 2, W // s + 2)).astype(np.float32)
        out += amp * cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)[:H, :W]
        tot += amp; amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)
def col(h): h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255
def paint(img, mask, color, alpha=1.0):
    a = (mask.astype(np.float32) * alpha)[..., None]
    img[:] = img * (1 - a) + color * a

bld = cv2.imread('bld.png', 0) > 0
road = cv2.imread('road.png', 0) > 0

# --- masks -------------------------------------------------------------------
water = np.zeros((H, W), np.uint8)
JN = np.array([(213398, 562500), (213421, 561821), (213489, 561143), (213557, 560690), (213648, 560238),
               (213807, 559786), (214011, 559424), (214193, 559197), (214329, 558971), (214510, 558745),
               (214692, 558519), (214828, 558202), (214896, 557750), (214919, 557071)], float)
cv2.polylines(water, [rp(JN)], False, 255, 46, lineType=cv2.LINE_AA)        # Jungnangcheon channel
for k, v in geo.items():
    if k.startswith('E'):
        for p, _ in v:
            cv2.polylines(water, [rp(p)], False, 255, 7, lineType=cv2.LINE_AA)  # tributaries
water = cv2.GaussianBlur(water, (0, 0), 1.2) > 100
bank = (cv2.dilate(water.astype(np.uint8), np.ones((45, 45), np.uint8)) > 0) & ~water   # riverside parks

occ = (bld | road).astype(np.uint8) * 255
for k, v in geo.items():
    if k.startswith(('A015', 'C0536')):
        for p, _ in v:
            cv2.polylines(occ, [rp(p)], False, 255, 3)
dist = cv2.distanceTransform((occ == 0).astype(np.uint8), cv2.DIST_L2, 5)
green = cv2.morphologyEx((dist > 22).astype(np.uint8), cv2.MORPH_OPEN, np.ones((21, 21), np.uint8)) > 0
forest = cv2.morphologyEx((dist > 45).astype(np.uint8), cv2.MORPH_OPEN, np.ones((41, 41), np.uint8)) > 0
green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8)) > 0
forest = cv2.morphologyEx(forest.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((81, 81), np.uint8)) > 0
green &= ~bld & ~road; forest &= ~bld & ~road
green |= bank & ~bld & ~road
green &= ~water

rail = np.zeros((H, W), np.uint8)
for k, v in geo.items():
    if k.startswith('A015'):
        for p, _ in v:
            cv2.polylines(rail, [rp(p)], False, 255, 2, lineType=cv2.LINE_AA)
yard = cv2.morphologyEx(rail, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0   # rail yards as ballast fields

# building size classes (big footprints = apartments / campus)
n, lab, stats, _ = cv2.connectedComponentsWithStats(bld.astype(np.uint8), 4)
big = np.isin(lab, np.where(stats[:, cv2.CC_STAT_AREA] > 900)[0][1:])

# --- paint -------------------------------------------------------------------
img = np.ones((H, W, 3), np.float32) * col('#F3F0E8')
grain = noise(6, 2)
img *= (0.97 + 0.05 * grain)[..., None]                                   # paper grain

paint(img, yard, col('#D9D4CA'), 0.9)
gn = noise(220, 5); leaf = noise(10, 2)
g_lo, g_hi = col('#C7D3AE'), col('#A9BE8A')
gcol = g_lo * (1 - gn[..., None]) + g_hi * gn[..., None]
img[green] = gcol[green] * (0.94 + 0.10 * leaf[green])[:, None]
f_lo, f_hi = col('#9DB27F'), col('#7E9764')
fcol = f_lo * (1 - gn[..., None]) + f_hi * gn[..., None]
canopy = (leaf > 0.55)
img[forest] = (fcol[forest] * (0.90 + 0.18 * leaf[forest])[:, None])
img[forest & canopy] *= 0.93                                              # tree-crown speckle

wn = noise(60, 3)
img[water] = (col('#9FBFCF') * (0.96 + 0.08 * wn[water])[:, None])
edge = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
img[edge] = col('#7FA2B5')

# street surfaces: a light band where the sidewalk lines are
img[road] = img[road] * 0.55 + col('#FAF8F3') * 0.45
img[cv2.dilate(rail, np.ones((2, 2), np.uint8)) > 0] = col('#6F6A63')

# building shadows (sun from the north-west -> shadow to the south-east)
sh = np.roll(np.roll(bld.astype(np.float32), 4, axis=0), 4, axis=1)
sh = cv2.GaussianBlur(sh, (0, 0), 1.5) * (~bld)
img *= (1 - 0.28 * sh)[..., None]
img[bld] = col('#DCD5C9')
img[big] = col('#CFC6B6')
bedge = cv2.morphologyEx(bld.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8)) > 0
img[bedge] = img[bedge] * 0.55 + col('#7C7468') * 0.45

np.save('base_real.npy', (np.clip(img, 0, 1) * 255).astype(np.uint8))
cv2.imwrite('base_real_preview.jpg', cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)[::3, ::3])
print('ok')
