import json, numpy as np, trimesh, sys, collections
OUT = sys.argv[1]
rep = json.load(open(f'{OUT}/report.json'))
res = {}
for r in rep:
    m = trimesh.load(f"{OUT}/{r['unit']}/{r['name']}.stl")
    n = m.face_normals; c = m.triangles_center; a = m.area_faces
    off_bed = c[:, 2] > 0.25
    steep = (n[:, 2] < -0.72) & (n[:, 2] > -0.985) & off_bed
    flat = (n[:, 2] <= -0.985) & off_bed
    # group flat faces into patches and measure the short side (bridge span)
    spans = []
    if flat.any():
        sub = m.submesh([np.where(flat)[0]], append=True)
        for comp in sub.split(only_watertight=False):
            e = comp.extents; spans.append((round(min(e[0], e[1]), 2), round(max(e[0], e[1]), 2), round(comp.bounds[0][2], 2)))
    steep_area = a[steep].sum()
    worst = max([s[0] for s in spans], default=0)
    res[r['kind']] = res.get(r['kind']) or dict(name=r['name'], steep_mm2=round(steep_area, 2), max_bridge_mm=worst, n_bridges=len(spans))
for k, v in res.items():
    print(f"{k:16s} steep {v['steep_mm2']:6.2f} mm2   bridges {v['n_bridges']:3d}  widest span {v['max_bridge_mm']:5.2f} mm   ({v['name']})")
