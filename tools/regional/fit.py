import numpy as np, cv2, sys, json
bld=cv2.imread('bld.png',0); road=cv2.imread('road.png',0)
X0,Y0,X1,Y1,R=np.load('meta.npy')
def fit(f, img_pts, map_pts, out):
    A=cv2.estimateAffinePartial2D(np.float32(img_pts), np.float32([(x-X0, Y1-y) for x,y in map_pts]))[0]
    im=cv2.imread('img/'+f)[:,:,:3]; h,w=im.shape[:2]
    s=np.hypot(A[0,0],A[1,0]); rot=np.degrees(np.arctan2(A[1,0],A[0,0]))
    # overlay: warp map edges into image space
    Ai=cv2.invertAffineTransform(A)
    e=cv2.Canny(bld,50,150)|road
    we=cv2.warpAffine(e,Ai,(w,h))
    ov=im.copy(); ov[we>0]=(0,0,255)
    cv2.imwrite(out,ov)
    print(f,'scale %.3f m/px rot %.2f deg'%(s,rot))
    return A.tolist()
if __name__=='__main__':
    f=sys.argv[1]; d=json.loads(sys.argv[2])
    A=fit(f,d['img'],d['map'],'fit_'+f)
    P=json.load(open('affine.json')) if __import__('os').path.exists('affine.json') else {}
    P[f]=A; json.dump(P,open('affine.json','w'))
