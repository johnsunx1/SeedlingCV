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



import cv2
cap = cv2.VideoCapture('./wheat_8s.mp4')
total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(total)
print(fps)
print(int(total / fps))

step = 15
lis_img = []
for i in range(1, int(total), int(step)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    cap.grab()
    ret, frame = cap.read()
    cv2.imwrite('picture'+'/'+str(i) + '.jpg', frame)
