"""MAIN 04F 84 공유형 L/R: 기존 벽 유지, 기준 도면(M84·S84) 가구·위생기구 배치, 문 두께, 유리 위치 반영."""
import math,sys,os,ezdxf
from shapely.geometry import LineString,Polygon,box
from shapely.ops import unary_union
from shapely import affinity
from blocks import groups
from seg import units
L2,L9=groups(2),groups(9)
def pick(G,ids,native):
    it=[x for i in ids for x in G[i]['items']]
    xs=[p[0] for _,pts,_ in it for p in pts]; ys=[p[1] for _,pts,_ in it for p in pts]
    x0,y0=min(xs),min(ys)
    return dict(items=[(L,[((p[0]-x0)/200,(p[1]-y0)/200) for p in pts],cl) for L,pts,cl in it],back=native)
LIB={
 'SOFA':pick(L2,[0],'top'),
 'DINING':pick(L2,[6,4,5,9,10],'top'),
 'FRIDGE':pick(L2,[12],'bottom'),
 'BED':pick(L9,[8,10,12],'left'),
 'DESK':pick(L9,[0,2,6],'top'),
 'WARDROBE':pick(L9,[17,18],'left'),
 'SHELF':pick(L9,[22],'left'),
 'LOWCAB':pick(L9,[29],'bottom'),
 'BATH':pick(L9,[23,25,27,31],'bottom'),
}
def kitchen(Lk):
    sink=pick(L2,[13,16],'bottom'); cook=pick(L2,[14,15,17,18],'bottom')
    it=[('A-FURN',[(0,0),(Lk,0),(Lk,3.75),(0,3.75)],True)]
    def put(b,cx):
        xs=[p[0] for _,pts,_ in b['items'] for p in pts]; w=max(xs)
        for L,pts,cl in b['items']: it.append((L,[(p[0]+cx-w/2,p[1]+0.9) for p in pts],cl))
    put(sink,Lk*0.33); put(cook,Lk*0.75)
    return dict(items=it,back='bottom')
ANG={'bottom':0,'right':90,'top':180,'left':270}  # rotation (CCW) that moves a bottom-back to that side
def place(b,back,x0,y0,mirror=False):
    rot=(ANG[back]-ANG[b['back']])%360
    out=[]
    for L,pts,cl in b['items']:
        g=LineString(pts if len(pts)>1 else pts*2)
        if mirror: g=affinity.scale(g,-1 if back in('top','bottom') else 1,1 if back in('top','bottom') else -1,origin=(0,0))
        g=affinity.rotate(g,rot,origin=(0,0))
        out.append((L,g,cl))
    bx=unary_union([g for _,g,_ in out]).bounds
    lay={'A-FURN':'50_FURNITURE','A-FIXT':'51_UNIT_FIXTURES'}
    return [(lay[L],[(x-bx[0]+x0,y-bx[1]+y0) for x,y in g.coords],cl) for L,g,cl in out]
def size(b,back):
    it=place(b,back,0,0); xs=[p[0] for _,pts,_ in it for p in pts]; ys=[p[1] for _,pts,_ in it for p in pts]
    return max(xs),max(ys)
# ---- layouts (paper units, 1 = 200 mm) --------------------------------------------
def R_layout():
    I=[]
    I+=place(LIB['FRIDGE'],'top',13.3,54.25-size(LIB['FRIDGE'],'top')[1])
    I+=place(kitchen(13.75),'top',16.6,54.25-3.75)
    w,h=size(LIB['DINING'],'right'); I+=place(LIB['DINING'],'right',21.75-w/2,39.4)
    w,h=size(LIB['SOFA'],'right'); I+=place(LIB['SOFA'],'right',43.75-w,40.2)
    w,h=size(LIB['BED'],'top'); I+=place(LIB['BED'],'top',18.5,79-h)
    w,h=size(LIB['WARDROBE'],'right'); I+=place(LIB['WARDROBE'],'right',30.25-w,61.0)
    I+=place(LIB['DESK'],'left',13.3,59.0)
    w,h=size(LIB['BED'],'right'); I+=place(LIB['BED'],'right',64.5-w,43.6)
    I+=place(LIB['DESK'],'bottom',45.6,38.3)
    w,h=size(LIB['WARDROBE'],'top'); I+=place(LIB['WARDROBE'],'top',53.2,58.25-h)
    I+=place(LIB['SHELF'],'left',30.75,60.6)
    w,h=size(LIB['BATH'],'top'); I+=place(LIB['BATH'],'top',31.2,79-h)
    I+=place(LIB['BATH'],'bottom',43.2,67.88,mirror=True)
    w,h=size(LIB['SHELF'],'right'); I+=place(LIB['SHELF'],'right',64.5-w,60.0)
    D=[((62,80),(57,80),(62,85)),((30.25,59.5),(30.25,55.5),(26.25,59.5)),((41,67.88),(37,67.88),(41,71.88)),
       ((54.12,76.5),(54.12,72.5),(50.12,76.5)),((47.5,58.25),(51.5,58.25),(47.5,54.25))]
    return I,D
def L_layout():
    I=[]
    w,h=size(LIB['BED'],'left'); I+=place(LIB['BED'],'left',115.9,43.0)
    I+=place(LIB['DESK'],'bottom',128.6,38.3)
    w,h=size(LIB['WARDROBE'],'top'); I+=place(LIB['WARDROBE'],'top',116.2,56.05-h)
    w,h=size(LIB['BED'],'top'); I+=place(LIB['BED'],'top',171.0,64.3-h)
    w,h=size(LIB['DESK'],'right'); I+=place(LIB['DESK'],'right',180.15-0.3-w,40.0)
    I+=place(LIB['WARDROBE'],'left',164.75,39.0)
    w,h=size(LIB['FRIDGE'],'top'); I+=place(LIB['FRIDGE'],'top',139.2,64.3-h)
    I+=place(kitchen(17.0),'top',142.5,64.3-3.75)
    w,h=size(LIB['DINING'],'top'); I+=place(LIB['DINING'],'top',148.0-w/2,46.0)
    w,h=size(LIB['SOFA'],'right'); I+=place(LIB['SOFA'],'right',164.25-w,40.0)
    I+=place(LIB['SHELF'],'left',115.9,63.0)
    w,h=size(LIB['BATH'],'top'); I+=place(LIB['BATH'],'top',128.3,79-h); I+=place(LIB['BATH'],'top',128.3,70.53-h)
    w,h=size(LIB['LOWCAB'],'top'); I+=place(LIB['LOWCAB'],'top',125.0,61.93-h)
    D=[((117,80),(122,80),(117,85)),((123.78,77.5),(123.78,73.5),(127.78,77.5)),((123.78,68.5),(123.78,64.5),(127.78,68.5)),
       ((129.5,56.05),(125.5,56.05),(129.5,52.05)),((164.75,59.5),(164.75,55.0),(169.25,59.5))]
    return I,D
CEN=(96.5,121.5)
def rot180(p): return (2*CEN[0]-p[0],2*CEN[1]-p[1])
FAC=dict(x0=12.5,x1=180.5,y0=37.5,y1=205.5)
def glass(poly):
    out=[]
    b=poly.boundary
    for name,line,n in (('y0',LineString([(0,FAC['y0']),(300,FAC['y0'])]),(0,1)),('y1',LineString([(0,FAC['y1']),(300,FAC['y1'])]),(0,-1)),
                        ('x0',LineString([(FAC['x0'],0),(FAC['x0'],300)]),(1,0)),('x1',LineString([(FAC['x1'],0),(FAC['x1'],300)]),(-1,0))):
        seg=b.intersection(line.buffer(0.01))
        for g in (seg.geoms if hasattr(seg,'geoms') else [seg]):
            if g.is_empty or g.length<0.5: continue
            x0,y0,x1,y1=g.bounds
            for off in (0.35,0.65):
                if n[0]==0: out.append(LineString([(x0,y0+n[1]*off),(x1,y0+n[1]*off)]))
                else: out.append(LineString([(x0+n[0]*off,y0),(x0+n[0]*off,y1)]))
    return unary_union(out)
def build(fn):
    m=ezdxf.readfile(fn).modelspace()
    cs,_=units(m)
    labs=[(e.dxf.text,e.dxf.insert) for e in m.query('TEXT') if e.dxf.layer=='90_LABELS']
    def unit_at(x,y):
        for c in cs:
            if c.buffer(0.3).contains(Polygon([(x-.01,y-.01),(x+.01,y-.01),(x+.01,y+.01)])): return c
    custom=[]
    for lab,fnL,probe in (('84 공유형 R',R_layout,(20,45)),('84 공유형 L',L_layout,(150,45))):
        I,D=fnL()
        for twin in (False,True):
            T=(lambda p:rot180(p)) if twin else (lambda p:p)
            px,py=T(probe); poly=unit_at(px,py)
            items=[(L,[T(p) for p in pts],cl) for L,pts,cl in I]
            doors=[(T(H),T(C),T(O)) for H,C,O in D]
            custom.append(dict(label=lab,poly=poly,items=items,doors=doors,glass=glass(poly)))
    return custom
if __name__=='__main__':
    from apply import run
    fn='src/MAIN_04F_UNIT_84_SPLIT_SHARED.dxf'
    r,sk=run(fn,'out/'+os.path.basename(fn),custom=build(fn))
    print(r,'skipped',sk)
