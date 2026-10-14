import ezdxf,sys
from ezdxf import path as P
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
def hatch_polys(m,prefixes):
    out=[]
    for e in m.query('HATCH'):
        if e.dxf.layer[:2] in prefixes:
            for p in P.from_hatch(e):
                v=[(q.x,q.y) for q in p.flattening(0.02)]
                if len(v)>2:
                    g=Polygon(v).buffer(0)
                    if g.area>1e-4: out.append((e.dxf.layer,g))
    return out
def units(m,b=0.15):
    hp=hatch_polys(m,('21','22','23','24'))
    U=unary_union([g.buffer(b) for _,g in hp])
    comps=list(U.geoms) if U.geom_type=='MultiPolygon' else [U]
    return [c.buffer(-b) for c in comps],hp
if __name__=='__main__':
    for fn in sys.argv[1:]:
        m=ezdxf.readfile(fn).modelspace()
        cs,_=units(m)
        labs=[(e.dxf.text,Point(e.dxf.insert.x,e.dxf.insert.y)) for e in m.query('TEXT') if e.dxf.layer=='90_LABELS']
        print('==',fn.split('/')[-1],len(cs))
        for c in cs:
            x0,y0,x1,y1=c.bounds
            print('   area %.1f m2  %dx%d mm  at (%.1f,%.1f)'%(c.area*0.04,(x1-x0)*200,(y1-y0)*200,x0,y0),[t for t,p in labs if c.buffer(0.5).contains(p)])
