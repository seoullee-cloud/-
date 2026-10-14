import cv2,numpy as np,sys
bld=cv2.imread('bld.png',0); road=cv2.imread('road.png',0)
X0,Y0,X1,Y1,R=np.load('meta.npy')
cx,cy,s=float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]); out=sys.argv[4]
x=int(cx-X0); y=int(Y1-cy)
c=np.full((2*s,2*s,3),255,np.uint8)
xa,ya=max(x-s,0),max(y-s,0)
b=bld[ya:y+s,xa:x+s]; r=road[ya:y+s,xa:x+s]
sub=c[ya-(y-s):ya-(y-s)+b.shape[0], xa-(x-s):xa-(x-s)+b.shape[1]]
sub[b>0]=(170,170,170); sub[r>0]=(0,0,0)
cv2.circle(c,(s,s),8,(0,0,255),3)
for i in range(0,2*s,100): cv2.line(c,(i,0),(i,2*s),(0,170,0),1); cv2.line(c,(0,i),(2*s,i),(0,170,0),1)
cv2.imwrite(out,cv2.resize(c,(700,700)))
