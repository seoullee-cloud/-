import cv2,numpy as np,sys
def gray(f):
    im=cv2.imread('img/'+f,cv2.IMREAD_UNCHANGED)[:,:,:3]; return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
big=gray('사이트_주변.png').astype(np.float32)
def find(f,lo,hi,n=40):
    g=gray(f); best=None
    for k in np.geomspace(lo,hi,n):  # k = small px per big px ratio: template scaled by k
        w,h=int(g.shape[1]*k),int(g.shape[0]*k)
        if w<30 or h<30 or w>=big.shape[1] or h>=big.shape[0]: continue
        t=cv2.resize(g,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)
        r=cv2.matchTemplate(big,t,cv2.TM_CCOEFF_NORMED)
        _,mv,_,ml=cv2.minMaxLoc(r)
        if best is None or mv>best[0]: best=(mv,k,ml,w,h)
    return best
for f in sys.argv[1:]:
    print(f, find(f,0.1,0.8))
