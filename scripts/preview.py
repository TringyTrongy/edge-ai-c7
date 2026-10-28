import cv2, glob, os, sys, numpy as np
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data'
files=sorted(glob.glob(root+'/images/*/*.jpg'))[:int(sys.argv[1]) if len(sys.argv)>1 else 12]
names=['sct013','robot'];col=[(0,200,255),(255,120,0)]
tiles=[]
for f in files:
    im=cv2.imread(f);h,w=im.shape[:2]
    lf=f.replace('/images/','/labels/').replace('.jpg','.txt')
    for l in open(lf):
        c,x,y,bw,bh=l.split();c=int(c);x,y,bw,bh=[float(v) for v in (x,y,bw,bh)]
        p1=(int((x-bw/2)*w),int((y-bh/2)*h));p2=(int((x+bw/2)*w),int((y+bh/2)*h))
        cv2.rectangle(im,p1,p2,col[c],2);cv2.putText(im,names[c],(p1[0],max(12,p1[1]-4)),0,.5,col[c],1)
    tiles.append(cv2.resize(im,(320,240)))
while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]
cv2.imwrite(sys.argv[2] if len(sys.argv)>2 else 'preview.jpg',np.vstack(rows))
