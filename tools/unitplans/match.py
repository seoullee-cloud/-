import ezdxf,pickle,numpy as np,cv2,sys,json
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from shapely import affinity
from seg import units,hatch_polys
RU=pickle.load(open('refunits.pkl','rb'))
RW=pickle.load(open('refwalls.pkl','rb')); RU=pickle.load(open('refunits.pkl','rb'))
S=0.25 # paper units per px (=50mm)
MAP={'84 분리형':[9],'84 분리형 S81':[9],'84':[9],'59':[7],'49':[3],'49 / R':[0],'49 / C':[4],'59 / R63':[1,5],'59 / R54':[8],'59 / La':[10],'59 / Lb':[11]}
ORI=[(r,f) for f in (False,True) for r in (0,90,180,270)]
SCALE_LABELS={'84 분리형','84 분리형 S81'}
def xf(g,r,f,sx=1.0):
    g=affinity.scale(g,sx/200,1/200,origin=(0,0))
    if f: g=affinity.scale(g,-1,1,origin=(0,0))
    return affinity.rotate(g,r,origin=(0,0))
def rast(geom,x0,y0,W,H):
    img=np.zeros((H,W),np.uint8)
    for g in (geom.geoms if hasattr(geom,'geoms') else [geom]):
        if g.is_empty: continue
        ext=np.array([[(x-x0)/S,(y-y0)/S] for x,y in g.exterior.coords]).round().astype(np.int32)
        cv2.fillPoly(img,[ext],255)
        for h in g.interiors:
            cv2.fillPoly(img,[np.array([[(x-x0)/S,(y-y0)/S] for x,y in h.coords]).round().astype(np.int32)],0)
    return img
def match_file(fn):
    m=ezdxf.readfile(fn).modelspace()
    cs,_=units(m)
    tw=unary_union([g for _,g in hatch_polys(m,('12','13'))])
    ts=unary_union([g for _,g in hatch_polys(m,('13',))])
    labs=[(e.dxf.text,Point(e.dxf.insert.x,e.dxf.insert.y)) for e in m.query('TEXT') if e.dxf.layer=='90_LABELS']
    res=[]
    for c in cs:
        lab=[t for t,p in labs if c.buffer(0.5).contains(p)][0]
        if lab not in MAP: res.append(dict(label=lab,poly=c,ref=None)); continue
        x0,y0,x1,y1=c.bounds; M=12
        X0,Y0=x0-M,y0-M; W=int((x1-x0+2*M)/S); H=int((y1-y0+2*M)/S)
        timg=rast(tw.intersection(c.buffer(M).envelope),X0,Y0,W,H)
        cands=[]
        for ri in MAP[lab]:
            best=None
            sxs=[round(1+0.005*k,3) for k in range(-12,13)] if lab in SCALE_LABELS else [1.0]
            for sx in sxs:
              for r,f in ORI:
                g=xf(RW[ri],r,f,sx); gx0,gy0,gx1,gy1=g.bounds
                tpl=rast(g,gx0,gy0,int((gx1-gx0)/S)+2,int((gy1-gy0)/S)+2)
                if tpl.shape[0]>=H or tpl.shape[1]>=W: continue
                sc=cv2.matchTemplate(timg,tpl,cv2.TM_CCORR_NORMED)
                _,mx,_,loc=cv2.minMaxLoc(sc)
                if best is None or mx>best[0]:
                    best=(mx,ri,r,f,X0+loc[0]*S-gx0,Y0+loc[1]*S-gy0,sx)
            mx,_,r,f,tx,ty,sx=best
            cs_=0
            for pts,cl in RU[ri]['ents'].get('A-COLS',[]):
                cp=affinity.translate(xf(Polygon(pts),r,f,sx),tx,ty)
                cs_+=1 if cp.intersects(ts) else -1
            cands.append(((cs_,mx),best))
        mx,ri,r,f,tx,ty,sx=max(cands)[1]
        res.append(dict(label=lab,poly=c,ref=ri,rot=r,flip=f,tx=tx,ty=ty,score=mx,sx=sx))
    return m,res,tw
if __name__=='__main__':
    out={}
    for fn in sys.argv[1:]:
        _,res,_=match_file(fn)
        out[fn]=res
        print('==',fn.split('/')[-1])
        for d in res:
            if d['ref'] is None: print('   ',d['label'],'SKIP'); continue
            print('   %-14s ref%-2d rot%3d flip%d sx%.3f score %.3f t=(%.2f,%.2f)'%(d['label'],d['ref'],d['rot'],d['flip'],d['sx'],d['score'],d['tx'],d['ty']))
    pickle.dump(out,open('match.pkl','wb'))
