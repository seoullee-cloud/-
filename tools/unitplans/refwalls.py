import pickle,math,numpy as np,cv2
from shapely.geometry import LineString,Point,Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
U=pickle.load(open('refunits.pkl','rb'))
R=10.0  # mm per px
WL=('A-WALL-INT','A-WALL-DEMISING','A-WALL-EXT','A-COLS')
def wall_polys(u):
    lines=[]
    for L in WL:
        for pts,cl in u['ents'].get(L,[]):
            p=pts+[pts[0]] if cl else pts
            if len(p)>1: lines.append(LineString(p))
    tree=STRtree(lines)
    ends=[]
    for i,l in enumerate(lines):
        if l.is_closed: continue
        for c in (l.coords[0],l.coords[-1]):
            P=Point(c); hit=False
            for j in tree.query(P.buffer(5)):
                if j==i: 
                    # self touch other than at this end?
                    continue
                if lines[j].distance(P)<5: hit=True;break
            if not hit: ends.append(c)
    pairs=sorted((math.dist(a,b),a,b) for i,a in enumerate(ends) for b in ends[i+1:] if math.dist(a,b)<=420)
    used=set(); jambs=[]
    for d,a,b in pairs:
        if a in used or b in used: continue
        used|={a,b}; jambs.append(LineString([a,b]))
    allL=lines+jambs
    x0,y0,x1,y1=unary_union(allL).bounds; x0-=200;y0-=200;x1+=200;y1+=200
    W=int((x1-x0)/R)+1;H=int((y1-y0)/R)+1
    img=np.zeros((H,W),np.uint8)
    for l in allL:
        pts=np.array([[(x-x0)/R,(y-y0)/R] for x,y in l.coords]).round().astype(np.int32)
        cv2.polylines(img,[pts],False,255,1)
    n,lab,st,_=cv2.connectedComponentsWithStats((img==0).astype(np.uint8),connectivity=4)
    mask=img.copy()
    for k in range(1,n):
        a=st[k,cv2.CC_STAT_AREA]
        comp=(lab==k).astype(np.uint8)
        per=cv2.arcLength(cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0][0],True)
        if st[k,0]==0 and st[k,1]==0: continue  # outside
        if 2*a/max(per,1)<=33: mask[lab==k]=255   # avg thickness <=330mm
    cs,hier=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    polys=[]
    for i,c in enumerate(cs):
        if hier[0][i][3]!=-1: continue
        c=cv2.approxPolyDP(c,0.8,True)
        if len(c)<3: continue
        shell=[(x0+p[0][0]*R,y0+p[0][1]*R) for p in c]
        holes=[]
        ch=hier[0][i][2]
        while ch!=-1:
            h=cv2.approxPolyDP(cs[ch],0.8,True)
            if len(h)>=3: holes.append([(x0+p[0][0]*R,y0+p[0][1]*R) for p in h])
            ch=hier[0][ch][0]
        g=Polygon(shell,holes).buffer(0)
        if g.area>2000: polys.append(g)
    return unary_union(polys),len(ends)-len(used)
if __name__=='__main__':
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,4,figsize=(24,18))
    out={}
    for i,ax in zip(sorted(U),axs.flat):
        w,nd=wall_polys(U[i]); out[i]=w
        for g in (w.geoms if hasattr(w,'geoms') else [w]):
            ax.fill(*g.exterior.xy,color='k')
            for h in g.interiors: ax.fill(*h.xy,color='w')
        for L,c in (('A-DOOR','r'),('A-GLAZ','c'),('A-FURN','g'),('A-FIXT','m')):
            for pts,cl in U[i]['ents'].get(L,[]):
                ax.plot([p[0] for p in pts],[p[1] for p in pts],c,lw=.4)
        ax.set_title(f"{i} {U[i]['title'][2]} dangling-left {nd}");ax.set_aspect('equal')
    fig.savefig('png/refwalls.png',dpi=70,bbox_inches='tight')
    pickle.dump(out,open('refwalls.pkl','wb'))
