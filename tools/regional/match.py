import numpy as np, cv2, sys, os
bld=cv2.imread('bld.png',0); road=cv2.imread('road.png',0)
X0,Y0,X1,Y1,R=np.load('meta.npy')
edges=cv2.Canny(bld,50,150)|road
ref=cv2.GaussianBlur(edges.astype(np.float32),(0,0),3)
def match(f, scales):
    im=cv2.imread('img/'+f,cv2.IMREAD_UNCHANGED)
    if im.shape[2]==4: im=im[:,:,:3]
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    best=None
    for s in scales:   # s = m per image px
        w,h=int(g.shape[1]*s/R),int(g.shape[0]*s/R)
        if w<80 or h<80 or w>=ref.shape[1] or h>=ref.shape[0]: continue
        gs=cv2.resize(g,(w,h),interpolation=cv2.INTER_AREA)
        e=cv2.Canny(cv2.GaussianBlur(gs,(3,3),0),60,160)
        t=cv2.GaussianBlur(e.astype(np.float32),(0,0),3)
        r=cv2.matchTemplate(ref,t,cv2.TM_CCOEFF_NORMED)
        dens=cv2.boxFilter(ref,-1,(w,h),normalize=True,anchor=(0,0))[:r.shape[0],:r.shape[1]]
        r[~np.isfinite(r)]=-1; r[dens<ref.mean()*0.5]=-1
        _,mv,_,ml=cv2.minMaxLoc(r)
        if best is None or mv>best[0]: best=(mv,s,ml,w,h)
    return best
if __name__=='__main__':
    f=sys.argv[1]; lo,hi,n=map(float,sys.argv[2:5])
    b=match(f,np.geomspace(lo,hi,int(n)))
    mv,s,(x,y),w,h=b
    print(f, 'score %.3f m/px %.3f'%(mv,s),'X %.0f-%.0f Y %.0f-%.0f'%(X0+x*R,X0+(x+w)*R,Y1-(y+h)*R,Y1-y*R))
