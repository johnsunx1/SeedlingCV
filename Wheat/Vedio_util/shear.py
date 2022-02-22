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
import numpy as np

def crop(img, save_dir, direction='w', sub=2000):
    crop_list = []
    img = np.array(img)
    w, h, channel = img.shape
    if (direction == 'h'):
        w = h
    for i in range(0, w - 1, sub):
        if (w - i - 1 <= 1.5 * sub and i + 2 * sub > w - 1):
            crop_list.append([i, w - 1])
            break
        crop_list.append([i, i + sub - 1])
    for i in crop_list:
        if (direction == 'h'):
            tmp = img[:, i[0]:i[1]]
        elif (direction == 'w'):
            tmp = img[i[0]:i[1]]
        cv2.imwrite(save_dir + '/' + str(i[0]) + '-' + str(i[1]) + '.jpg', tmp)

if __name__ == '__main__':

    src = 'final-2.jpg'
    img = cv2.imread(src)
    save_path = './sheared'
    crop(img, save_dir=save_path, direction='w', sub=1000)
