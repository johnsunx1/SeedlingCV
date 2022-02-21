'''
Copyright (c) 2022  https://github.com/14385423  https://github.com/ZCFzhaochuanfei  All rights reserved.

Reference source:

https://github.com/ZhihengCV/Bayesian-Crowd-Counting

https://github.com/dyh/unbox_yolov5_deepsort_counting

https://github.com/ultralytics/yolov5

'''

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
