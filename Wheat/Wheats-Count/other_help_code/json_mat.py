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
from scipy.io import savemat
import json
import argparse
from glob import glob

def json_to_mat(filename):
    fin = open(filename, encoding='UTF-8')
    content = json.load(fin)
    data = dict()
    content_out = []
    for k, v in content.items():
        if (k == 'points'):
            for i in v:
                content_out.append([i['x'], i['y']])
    data['annPoints'] = content_out
    fin.close()
    return data


args = None


def parse_args():
    parser = argparse.ArgumentParser(description='json to mat')

    parser.add_argument('--data-dir', default='./json',
                        help='the json dir')

    parser.add_argument('--mat-dir', default='./mat_dir', help='save mat dir')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    src_dir = args.data_dir
    save_dir = args.mat_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    im_list = glob(os.path.join(src_dir, '*json'))

    for im_path in im_list:
        mat = json_to_mat(im_path)
        name = os.path.basename(im_path)
        mat_name = name.replace('.json', '_ann.mat')
        savemat(save_dir + '/' + mat_name, mat)
