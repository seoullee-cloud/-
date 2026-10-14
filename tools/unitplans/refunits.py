import ezdxf,pickle,collections
d=ezdxf.readfile('src/유닛 치수선.dxf'); m=d.modelspace()
titles=[(e.dxf.insert.x,e.dxf.insert.y,e.text) for e in m.query('MTEXT')]
units={i:{'title':t,'ents':collections.defaultdict(list)} for i,t in enumerate(titles)}
def owner(minx,maxy):
    best=None
    for i,(tx,ty,_) in enumerate(titles):
        if tx-2500<=minx and ty>=maxy-200:
            s=(ty-maxy)+(minx-tx)
            if best is None or s<best[0]: best=(s,i)
    return best[1] if best else None
for e in m.query('POLYLINE'):
    pts=[(p[0],p[1]) for p in e.points()]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    i=owner(min(xs),max(ys))
    units[i]['ents'][e.dxf.layer].append((pts,e.is_closed))
for i,u in units.items():
    allp=[p for L,v in u['ents'].items() if L.startswith('A-WALL') for pts,_ in v for p in pts]
    xs=[p[0] for p in allp]; ys=[p[1] for p in allp]
    u['bbox']=(min(xs),min(ys),max(xs),max(ys))
    print(i,u['title'][2],{k:len(v) for k,v in u['ents'].items()},'size',round(max(xs)-min(xs)),round(max(ys)-min(ys)))
pickle.dump(units,open('refunits.pkl','wb'))
