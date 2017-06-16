import numpy as np
import cv2
from psnr import psnr

# img = cv2.imread('dataset/test/crop/070_im.png')
mask = cv2.imread('mask.png',0)
p = 0
l2 = 0
l1 = 0

for i in range(1,101):
    img = cv2.imread('dataset/val/paris_eval_gt/%03d_im.png'%i)
    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)
    p = p+psnr(img[32:32+64,32:32+64,:].astype(np.float32),dst[32:32+64,32:32+64,:].astype(np.float32))
    cv2.imwrite('result/test/old_all/%03d_im.png'%i,dst)
    t = img[32:32+64,32:32+64,:].astype(np.float32)/127.5-dst[32:32+64,32:32+64,:].astype(np.float32)/127.5
    l2 = l2 + np.mean(np.square(t))
    t = np.abs(t)
    l1 = l1 + np.mean(t)

print(l1/100.0)
print(l2/100.0)
print(p/100.0)
