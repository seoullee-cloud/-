import ezdxf,pickle,math,sys,os
import numpy as np
from ezdxf import path as P
from shapely.geometry import LineString,Polygon,MultiPolygon,Point
from shapely.ops import unary_union
from shapely import affinity
from match import match_file,xf,RW
RU=pickle.load(open('refunits.pkl','rb'))
T_ENTRY,T_INT=50.0,40.0   # door leaf thickness mm (현관문 / 실내문)
def tpt(p,d):
    x,y=p[0]/200,p[1]/200
    if d['flip']: x=-x
    a=math.radians(d['rot']); c,s=math.cos(a),math.sin(a)
    return (c*x-s*y+d['tx'], s*x+c*y+d['ty'])
def geoms(g): return [x for x in (g.geoms if hasattr(g,'geoms') else [g]) if not x.is_empty]
def ent_geom(e):
    v=[(q.x,q.y) for q in P.make_path(e).flattening(0.01)]
    return LineString(v) if len(v)>1 else Point(v[0])
def door_parts(pts):
    """ref door polyline -> (hinge, open_end, closed_end, arc_pts) in ref mm"""
    if math.dist(pts[0],pts[1])>300: h,o,arc=pts[0],pts[1],pts[1:]
    else: h,o,arc=pts[-1],pts[-2],pts[:-1][::-1]
    return h,o,arc[-1],arc
def leaf_rect(h,o,c,t):
    ux,uy=(o[0]-h[0]),(o[1]-h[1]); L=math.hypot(ux,uy); ux/=L;uy/=L
    nx,ny=-uy,ux
    # thickness on the swing side (toward the closed position), like a standard door block
    if (c[0]-h[0])*nx+(c[1]-h[1])*ny<0: nx,ny=-nx,-ny
    return [h,o,(o[0]+nx*t,o[1]+ny*t),(h[0]+nx*t,h[1]+ny*t)]
def _extend(r):
    c=list(r.exterior.coords)[:4]
    e=[(math.dist(c[i],c[(i+1)%4]),i) for i in range(4)]
    L,i=max(e)
    a,b=c[i],c[(i+1)%4]; ux,uy=(b[0]-a[0])/L,(b[1]-a[1])/L
    k=8.0
    return Polygon([(p[0]-ux*k,p[1]-uy*k) if j in(i,(i+3)%4) else (p[0]+ux*k,p[1]+uy*k) for j,p in enumerate(c)]).convex_hull
def run(fn,outfn):
    doc=ezdxf.readfile(fn); m=doc.modelspace()
    _,res,_=match_file(fn)
    done=[d for d in res if d['ref'] is not None]
    skipped=[d['label'] for d in res if d['ref'] is None]
    if not done: return None,skipped
    # --- regions
    unitU=unary_union([d['poly'] for d in done])
    structP=[]
    for e in m.query('HATCH'):
        if e.dxf.layer=='13_POCHE_STRUCTURE':
            for p in P.from_hatch(e):
                v=[(q.x,q.y) for q in p.flattening(0.01)]
                if len(v)>2: structP.append(Polygon(v).buffer(0))
    sU=unary_union(structP)
    refW=[];env=[]
    clipU=unary_union([d['poly'] for d in done]).buffer(0.35,join_style=2)
    for d in done:
        w=affinity.translate(xf(RW[d['ref']],d['rot'],d['flip']),d['tx'],d['ty'])
        d['phantom']=[]
        for pts,cl in RU[d['ref']]['ents'].get('A-COLS',[]):
            cp=Polygon([tpt(p,d) for p in pts])
            if cp.intersects(sU): continue
            d['phantom'].append(cp)
            wm=w.difference(cp); strips=[]
            for g in geoms(wm.intersection(cp.buffer(1.5))):
                if g.geom_type!='Polygon' or g.area<0.01: continue
                r=g.minimum_rotated_rectangle
                strips.append(affinity.scale(r,6,6,origin=r.centroid).intersection(cp) if False else _extend(r).intersection(cp).intersection(clipU))
            w=unary_union([wm]+strips)
        refW.append(w)
        env.append(unary_union([d['poly'],w]).buffer(0.8).buffer(-0.8))
    envU=unary_union(env)
    # --- old wall poche
    oldW=[];old_hatch=None
    for e in list(m.query('HATCH')):
        if e.dxf.layer=='12_POCHE_WALLS':
            old_hatch=old_hatch or e
            for p in P.from_hatch(e):
                v=[(q.x,q.y) for q in p.flattening(0.01)]
                if len(v)>2: oldW.append(Polygon(v).buffer(0))
            m.delete_entity(e)
    oldU=unary_union(oldW)
    keep=oldU.difference(envU.buffer(0.05))
    keep=unary_union([g for g in geoms(keep) if g.area>0.05])
    struct=[]
    for e in m.query('HATCH'):
        if e.dxf.layer in('11_POCHE_CORE','13_POCHE_STRUCTURE'):
            for p in P.from_hatch(e):
                v=[(q.x,q.y) for q in p.flattening(0.01)]
                if len(v)>2: struct.append(Polygon(v).buffer(0))
    structU=unary_union(struct)
    newW=unary_union([keep]+refW).buffer(0.06,join_style=2).buffer(-0.06,join_style=2).difference(structU)
    newW=[Polygon(g.exterior,[r for r in g.interiors if Polygon(r).area>0.3]).simplify(0.005) for g in geoms(newW) if g.area>0.02]
    # cut profiles: drop those belonging to old walls, redraw from new walls
    oldB=oldU.boundary.buffer(0.03)
    for e in list(m.query('LWPOLYLINE[layer=="10_CUT_PROFILE"]')):
        if oldB.contains(ent_geom(e)): m.delete_entity(e)
    for g in newW:
        h=m.add_hatch(dxfattribs={'layer':'12_POCHE_WALLS','color':7})
        h.set_solid_fill(color=7,rgb=ezdxf.colors.int2rgb(1383199))
        h.paths.add_polyline_path(list(g.exterior.coords)[:-1],is_closed=True,flags=1)
        for r in g.interiors: h.paths.add_polyline_path(list(r.coords)[:-1],is_closed=True)
        for ring in [g.exterior]+list(g.interiors):
            m.add_lwpolyline(list(ring.coords)[:-1],close=True,dxfattribs={'layer':'10_CUT_PROFILE'})
    WU=unary_union(newW)
    # move new hatches behind lines: put at start of draw order
    # --- glazing: clip old ring out of units, add reference glass
    cut=envU.buffer(0.05)
    for e in list(m.query('LWPOLYLINE[layer=="43_GLAZING"]')):
        g=ent_geom(e)
        if g.intersects(cut):
            rest=g.difference(cut)
            for r in geoms(rest):
                if r.geom_type=='LineString' and r.length>0.05:
                    m.add_lwpolyline(list(r.coords),dxfattribs={'layer':'43_GLAZING'})
            m.delete_entity(e)
    # --- doors / furniture inside units: remove
    db=unitU.buffer(0.3)
    for e in list(m.query('LWPOLYLINE[layer=="40_DOORS"]')):
        if ent_geom(e).intersects(db): m.delete_entity(e)
    for e in list(m.query('LWPOLYLINE[layer=="50_FURNITURE"]')):
        if ent_geom(e).intersects(unitU): m.delete_entity(e)
    if '51_UNIT_FIXTURES' not in doc.layers:
        src=doc.layers.get('50_FURNITURE')
        L=doc.layers.add('51_UNIT_FIXTURES',color=src.dxf.color)
        L.dxf.lineweight=src.dxf.lineweight; L.dxf.true_color=src.dxf.true_color
    nd=[0,0]
    for d in done:
        u=RU[d['ref']]['ents']
        for lay,tl in (('A-FURN','50_FURNITURE'),('A-FIXT','51_UNIT_FIXTURES'),('A-GLAZ','43_GLAZING')):
            for pts,cl in u.get(lay,[]):
                m.add_lwpolyline([tpt(p,d) for p in pts],close=cl,dxfattribs={'layer':tl})
        for cp in d['phantom']:
            for pts,cl in u.get('A-GLAZ',[]):
                q=[tpt(p,d) for p in pts]
                for a,b in ((q[0],q[1]),(q[-1],q[-2])):
                    if cp.exterior.distance(Point(a))<0.15 or cp.contains(Point(a)):
                        L=math.dist(a,b); ux,uy=(a[0]-b[0])/L,(a[1]-b[1])/L
                        seg=LineString([a,(a[0]+ux*8,a[1]+uy*8)]).intersection(cp.buffer(0.01))
                        if seg.length>0.05:
                            for g in geoms(LineString([a,seg.coords[-1]]).difference(WU)):
                                if g.length>0.05: m.add_lwpolyline(list(g.coords),dxfattribs={'layer':'43_GLAZING'})
        for pts,cl in u.get('A-DOOR',[]):
            h,o,c,arc=door_parts(pts)
            entry=math.dist(h,o)>=950
            t=T_ENTRY if entry else T_INT; nd[0 if entry else 1]+=1
            m.add_lwpolyline([tpt(p,d) for p in leaf_rect(h,o,c,t)],close=True,dxfattribs={'layer':'40_DOORS'})
            m.add_lwpolyline([tpt(p,d) for p in arc],dxfattribs={'layer':'40_DOORS'})
    # hatches first in draw order so lines stay on top
    order={}
    for i,e in enumerate(m.query('HATCH')): order[e.dxf.handle]='%X'%(0x100+i)
    for i,e in enumerate(m.query('LWPOLYLINE TEXT MTEXT LINE')): order[e.dxf.handle]='%X'%(0x100000+i)
    m.set_redraw_order(order)
    doc.saveas(outfn)
    return dict(units=len(done),entry=nd[0],interior=nd[1]),skipped
if __name__=='__main__':
    os.makedirs('out',exist_ok=True)
    for fn in sys.argv[1:]:
        r,sk=run(fn,'out/'+os.path.basename(fn))
        print(os.path.basename(fn),r,'skipped:',len(sk))
