import ezdxf,sys,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ezdxf import path as P
fn,out=sys.argv[1],sys.argv[2]; win=[float(x) for x in sys.argv[3:7]] if len(sys.argv)>3 else None
d=ezdxf.readfile(fn); m=d.modelspace()
cols={'21':'#f4c27a','22':'#a8d5a2','23':'#9cc3e6','24':'#e6a0c4','20':'#dddddd','36':'#cccccc'}
fig,ax=plt.subplots(figsize=(16,16),dpi=100)
for e in m.query('HATCH'):
    c=cols.get(e.dxf.layer[:2],'#555555' if e.dxf.layer.startswith('1') else '#eeeeee')
    for p in P.from_hatch(e):
        v=list(p.flattening(0.05))
        if len(v)>2: ax.fill([q.x for q in v],[q.y for q in v],color=c,lw=0,alpha=0.9 if e.dxf.layer.startswith('1') else 0.6)
lc={'40_DOORS':'red','43_GLAZING':'cyan','50_FURNITURE':'green','10_CUT_PROFILE':'black','42_CORE_FIXTURES':'purple','03_COLOR_REGION_EDGES':'#999999','01_FLOOR_OUTLINE':'blue','51_UNIT_FIXTURES':'magenta'}
for e in m.query('LWPOLYLINE'):
    if e.dxf.layer not in lc: continue
    v=list(P.make_path(e).flattening(0.05))
    ax.plot([q.x for q in v],[q.y for q in v],color=lc[e.dxf.layer],lw=0.6)
for e in m.query('TEXT'):
    if win and not (win[0]<e.dxf.insert.x<win[2] and win[1]<e.dxf.insert.y<win[3]): continue
    ax.text(e.dxf.insert.x,e.dxf.insert.y,e.dxf.text,fontsize=6)
ax.set_aspect('equal')
if win: ax.set_xlim(win[0],win[2]); ax.set_ylim(win[1],win[3])
fig.savefig(out,bbox_inches=None if win else 'tight')
