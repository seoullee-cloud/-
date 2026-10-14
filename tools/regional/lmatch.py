import numpy as np, cv2, sys
bld=cv2.imread('bld.png',0); road=cv2.imread('road.png',0)
X0,Y0,X1,Y1,R=np.load('meta.npy')
f=sys.argv[1]; cx,cy,win=map(float,sys.argv[2:5]); lo,hi=map(float,sys.argv[5:7])
ref_full=cv2.GaussianBlur((cv2.Canny(bld,50,150)|road).astype(np.float32),(0,0),2.5)
im=cv2.imread('img/'+f,cv2.IMREAD_UNCHANGED)[:,:,:3]; g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
best=None
for s in np.geomspace(lo,hi,30):
    w,h=int(g.shape[1]*s),int(g.shape[0]*s)
    x0=int(cx-X0-win-w/2); y0=int(Y1-cy-win-h/2)
    x0=max(x0,0); y0=max(y0,0)
    ref=ref_full[y0:y0+int(2*win+h), x0:x0+int(2*win+w)]
    if ref.shape[0]<=h or ref.shape[1]<=w: continue
    gs=cv2.resize(g,(w,h),interpolation=cv2.INTER_AREA)
    e=cv2.Canny(cv2.GaussianBlur(gs,(5,5),0),40,120)
    t=cv2.GaussianBlur(e.astype(np.float32),(0,0),2.5)
    r=cv2.matchTemplate(ref,t,cv2.TM_CCOEFF_NORMED); r[~np.isfinite(r)]=-1
    _,mv,_,ml=cv2.minMaxLoc(r)
    if best is None or mv>best[0]: best=(mv,s,(x0+ml[0],y0+ml[1]),w,h,gs)
mv,s,(x,y),w,h,gs=best
ext=(X0+x, X0+x+w, Y1-y-h, Y1-y)
print(f,'score %.3f m/px %.3f'%(mv,s),'X %.1f %.1f Y %.1f %.1f'%ext)
# overlay check
ov=cv2.cvtColor(gs,cv2.COLOR_GRAY2BGR)//2+60
m=road[y:y+h,x:x+w]>0; ov[m]=(0,0,255)
m=cv2.Canny(bld[y:y+h,x:x+w],50,150)>0; ov[m]=(255,255,0)
cv2.imwrite('ov_'+f, ov)
open('placements.txt','a').write('%s %.3f %.1f %.1f %.1f %.1f %.3f\n'%((f,s)+ext+(mv,)))
