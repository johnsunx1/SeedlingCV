'''
Copyright (c) 2022  https://gitee.com/l1233   https://gitee.com/zhao-chuanfei  All rights reserved.

Reference source:

https://github.com/ZhihengCV/Bayesian-Crowd-Counting

https://github.com/dyh/unbox_yolov5_deepsort_counting

https://github.com/ultralytics/yolov5

'''


import os
import cv2
import imutils
import numpy as np

img_dir = 'picture'

names = os.listdir(img_dir)
images = []
for name in names:
    img_path = os.path.join(img_dir, name)
    image = cv2.imread(img_path)
    images.append(image)
# print(imutils.is_cv3())
stitcher = cv2.createStitcher(cv2.Stitcher_SCANS) if imutils.is_cv3() else cv2.Stitcher_create(mode=1)
status, stitched = stitcher.stitch(images)
stitched = np.array(stitched)


stitched_copy = stitched.copy()


# stitched = np.rot90(stitched, -1)
stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)
mask = np.zeros(thresh.shape, dtype="uint8")
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
minRect = mask.copy()
sub = mask.copy()
while cv2.countNonZero(sub) > 0:
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)
cnts, hierarchy = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)
stitched = stitched[y:y + h, x:x + w]

cv2.imwrite('final-1.jpg',stitched_copy)
cv2.imwrite('final-2.jpg', stitched)
