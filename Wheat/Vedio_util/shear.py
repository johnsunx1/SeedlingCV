'''
Copyright (c) 2022  https://gitee.com/l1233   https://gitee.com/zhao-chuanfei  All rights reserved.

Reference source:

https://github.com/ZhihengCV/Bayesian-Crowd-Counting

https://github.com/dyh/unbox_yolov5_deepsort_counting

https://github.com/ultralytics/yolov5

'''

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
