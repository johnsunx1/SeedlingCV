########################################################################
# This software is Copyright 2022 The Regents of Shandong Agricultural University.
# All Rights Reserved.
#
# Permission to copy, modify, and distribute this software and its
# documentation for educational, research and non-profit purposes, without fee,
# and without a written agreement is hereby granted, provided that the above
# copyright notice, this paragraph and the following two paragraphs appear
# in all copies.
#
# This software program and documentation are copyrighted by The Regents of
# Shandong Agricultural University. The software program and documentation are supplied
# "as is", without any accompanying services from The Regents. The Regents does
# not warrant that the operation of the program will be uninterrupted or
# error-free. The end-user understands that the program was developed for
# research purposes and is advised not to rely exclusively on the program for
# any reason.
#

#Author: Qikun Zhao, Chuanfeng Zhao (https://github.com/14385423,https://github.com/ZCFzhaochuanfei)
#For commermical usage, please contact corresponding author (johnsunx1@yahoo.com)
########################################################################




import os
import cv2
import imutils
import numpy as np

img_dir = 'picture'

names = os.listdir(img_dir)
names.sort(key=lambda x: int(x[:-4]))
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
