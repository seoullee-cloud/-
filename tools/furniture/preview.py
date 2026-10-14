import json, numpy as np, trimesh, matplotlib, sys; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
rep=json.load(open('out/report.json'))
seen={}; 
for r in rep:
    seen.setdefault(r['kind'],r)
names=list(seen.values())
cols=5; rows=(len(names)+cols-1)//cols
fig=plt.figure(figsize=(cols*4,rows*4))
light=np.array([0.4,-0.6,0.7]); light/=np.linalg.norm(light)
for k,r in enumerate(names):
    m=trimesh.load(f"out/{r['unit']}/{r['name']}.stl")
    # view from the front: front is local -y only for back=+y pieces; rotate so the back faces away
    rot=0  # exported pieces are already in the local frame (back = +Y)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rot),[0,0,1]))
    ax=fig.add_subplot(rows,cols,k+1,projection='3d')
    tri=m.vertices[m.faces]; n=m.face_normals
    sh=np.clip(n@light,0,1)*0.75+0.25
    pc=Poly3DCollection(tri,facecolors=plt.cm.Greys(0.15+0*sh)[:, :3]*sh[:,None],edgecolor='none')
    ax.add_collection3d(pc)
    e=m.extents.max(); c=m.bounds.mean(0)
    ax.set_xlim(c[0]-e/2,c[0]+e/2); ax.set_ylim(c[1]-e/2,c[1]+e/2); ax.set_zlim(0,e)
    ax.view_init(elev=28,azim=-60); ax.set_box_aspect((1,1,1)); ax.axis('off')
    ax.set_title(f"{r['kind']}\n{r['print_mm'][0]}×{r['print_mm'][1]}×{r['print_mm'][2]} mm",fontsize=9)
plt.tight_layout(); plt.savefig(sys.argv[1] if len(sys.argv)>1 else 'preview.png',dpi=90)
