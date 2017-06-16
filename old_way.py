import numpy as np
import cv2

img = cv2.imread('dataset/test/crop/065_im.png')
mask = cv2.imread('mask.png',0)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

cv2.imwrite('out.png',dst)