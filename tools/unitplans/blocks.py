import pickle,sys
from shapely.geometry import LineString,Polygon,MultiLineString
from shapely.ops import unary_union
U=pickle.load(open('refunits.pkl','rb'))
def groups(ri,layers=('A-FURN','A-FIXT'),tol=30):
    items=[]
    for L in layers:
        for pts,cl in U[ri]['ents'].get(L,[]):
            items.append((L,pts,cl,LineString(pts+[pts[0]] if cl else pts) if len(pts)>1 else None))
    # union-find by proximity
    par=list(range(len(items)))
    def f(i):
        while par[i]!=i: par[i]=par[par[i]]; i=par[i]
        return i
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            if items[i][3].distance(items[j][3])<=tol: par[f(i)]=f(j)
    G={}
    for i,it in enumerate(items): G.setdefault(f(i),[]).append(it)
    out=[]
    for k,v in G.items():
        b=MultiLineString([x[3] for x in v]).bounds
        out.append(dict(items=[(x[0],x[1],x[2]) for x in v],bbox=b,layer=v[0][0]))
    out.sort(key=lambda g:(-g['bbox'][3],g['bbox'][0]))
    return out
if __name__=='__main__':
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    ri=int(sys.argv[1]); gs=groups(ri)
    fig,ax=plt.subplots(figsize=(14,14))
    from refwalls import wall_polys
    w,_=wall_polys(U[ri])
    for g in (w.geoms if hasattr(w,'geoms') else [w]): ax.fill(*g.exterior.xy,color='#888')
    for pts,cl in U[ri]['ents'].get('A-DOOR',[]): ax.plot([p[0] for p in pts],[p[1] for p in pts],'r',lw=.6)
    for pts,cl in U[ri]['ents'].get('A-GLAZ',[]): ax.plot([p[0] for p in pts],[p[1] for p in pts],'c',lw=.6)
    for i,g in enumerate(gs):
        for L,pts,cl in g['items']:
            p=pts+[pts[0]] if cl else pts
            ax.plot([q[0] for q in p],[q[1] for q in p],'g' if L=='A-FURN' else 'm',lw=.6)
        b=g['bbox']; ax.text((b[0]+b[2])/2,(b[1]+b[3])/2,str(i),fontsize=11,color='b',weight='bold')
        print(i,g['layer'],'%.0fx%.0f'%(b[2]-b[0],b[3]-b[1]),'at',round(b[0]),round(b[1]))
    ax.set_aspect('equal'); fig.savefig(f'png/blocks{ri}.png',dpi=60,bbox_inches='tight')
