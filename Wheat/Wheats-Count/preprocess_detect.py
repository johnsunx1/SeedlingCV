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



from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
from utils import getdir


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
    return Image.fromarray(im)


def parse_args():
    parser = argparse.ArgumentParser(description='Detect ')

    parser.add_argument('--origin-dir', default='Detect-orig/orig',
                        help='original data directory')

    parser.add_argument('--data-dir', default='Detect-processed/detect',
                        help='processed data directory')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    orig_dir = args.origin_dir
    save_dir = args.data_dir

    save_dir = getdir.increment_path(save_dir)
    save_dir = str(save_dir)
    min_size = 512
    max_size = 2048

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    im_list = glob(os.path.join(orig_dir, '*jpg'))

    for im_path in im_list:
        name = os.path.basename(im_path)
        im = generate_data(im_path)
        im_save_path = os.path.join(save_dir, name)
        im.save(im_save_path)
