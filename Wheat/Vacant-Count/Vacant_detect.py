'''
Copyright (c) 2022  https://github.com/14385423  https://github.com/ZCFzhaochuanfei  All rights reserved.

Reference source:

https://github.com/ZhihengCV/Bayesian-Crowd-Counting

https://github.com/dyh/unbox_yolov5_deepsort_counting

https://github.com/ultralytics/yolov5

'''

import argparse
import numpy as np
from detector import Detector
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import glob
import re
from pathlib import Path


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)

    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")

    dir = path if path.suffix == '' else path.parent

    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)

    return path

def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def filter_len(data, keep, scale, intervals):
    def check_len(lenn):
        flag = False
        for i in intervals:
            if (lenn >= i[0] and lenn <= i[1]):
                flag = True
                break
        return flag

    keep_o = []
    for i in keep:
        lenn = abs(data[i][1] - data[i][3]) * scale
        if (check_len(lenn) == True):
            keep_o.append(i)
    return keep_o


def view_xyz(result, z_name, save_path):
    result = np.mat(result)
    x = result[:, 0]
    y = result[:, 1]
    if (z_name == 'len'):
        z = result[:, 2]
    elif (z_name == 'area'):
        z = result[:, 3]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    if (z_name == 'len'):
        z_name = z_name + '(cm)'
    elif (z_name == 'area'):
        z_name = z_name + '(cm^2)'
    ax.set_zlabel(z_name, fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('y(cm)', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('x(cm)', fontdict={'size': 15, 'color': 'red'})
    plt.savefig(save_path + '/' + 'x-y-' + z_name)
    plt.close()


def view_xy(result, y_name, save_path):
    result = np.array(result)
    name_list = []
    for i in range(0, len(result)):
        name_list.append(str(i))
    if (y_name == 'len'):
        len_list = list(result[:, 2])

        plt.figure(figsize=(24, 8))

        plt.bar(range(len(len_list)), len_list, tick_label=name_list)
        plt.xlabel('id')
        plt.ylabel('len(cm)')
        plt.savefig(save_path + '/' + 'id_len')
        plt.close()
    elif (y_name == 'area'):
        area_list = list(result[:, 3])

        plt.figure(figsize=(24, 8))

        plt.bar(range(len(area_list)), area_list, tick_label=name_list)
        plt.xlabel('id')
        plt.ylabel('area(cm^2)')
        plt.savefig(save_path + '/' + 'id_area')
        plt.close()


def get_LineThickness(im):
    return round(0.001 * (im.shape[0] + im.shape[1]) * 0.5) + 1


def draw_box(im, c1, c2, line_thickness, index):
    line_thickness = line_thickness or get_LineThickness(im)
    color = (255, 255, 0)
    index = str(index)
    cv2.rectangle(im, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    font_thickness = max(line_thickness - 1, 1)
    t_size = cv2.getTextSize(index, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(im, 'id:{}'.format(index), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description='Detect ')

    parser.add_argument('--origin-dir', default='img/orthophotoll.jpg',
                        help='original data directory')

    parser.add_argument('--save-dir', default='out/result',
                        help='save the result dir')

    parser.add_argument('--scale', default=25 / 150, help='the scale of field')

    parser.add_argument('--filter_len', default=None, help='the interval of len to filter')
    # parser.add_argument('--filter_len', default=[[20,80]], help='the interval of len to filter')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    im = cv2.imread(args.origin_dir)
    im = np.array(im)

    im_orig = im.copy()
    im_del_repetition_orig = im.copy()
    im_filter_line = im.copy()
    im_result = im.copy()

    lenn = 2000
    sub_len = int(lenn * (1 - 0.4))

    h, w, _ = im.shape
    detector = Detector()
    data = []

    save_dir = args.save_dir
    save_dir = increment_path(save_dir)
    os.makedirs(save_dir)
    save_dir = str(save_dir)

    for i in range(0, h - 1, sub_len):
        print(i, i + lenn)
        tmp_im = im[i:i + lenn]
        boxes = detector.detect(tmp_im)
        for x1, y1, x2, y2, kind, lv in boxes:
            lv = lv.data.cpu().numpy()
            lv = float(lv)
            cv2.rectangle(im_orig[i:i + lenn], (x1, y1), (x2, y2), color=(255, 255, 0), thickness=3,
                          lineType=cv2.LINE_AA)
            data.append([x1, y1 + i, x2, y2 + i, lv])
    cv2.imwrite(save_dir + '/' + 'orig.jpg', im_orig)
    data = np.array(data)

    keep = py_cpu_nms(data, thresh=0.3)

    for i in keep:
        cv2.rectangle(im_del_repetition_orig, (int(data[i][0]), int(data[i][1])), (int(data[i][2]), int(data[i][3])),
                      color=(255, 255, 0), thickness=3,
                      lineType=cv2.LINE_AA)
    cv2.imwrite(save_dir + '/' + 'del_repetition_orig.jpg', im_del_repetition_orig)

    all_len = 0
    all_area = 0
    result = []

    scale = args.scale
    scale = float(scale)

    if (args.filter_len != None):
        intervals = args.filter_len
        keep = filter_len(data, keep, intervals=intervals, scale=scale)

    id = 0
    new = []
    for i in keep:
        new.append(((data[i][0] + data[i][2]) / 2, (data[i][1] + data[i][3]) / 2, i))

    ll = 500
    new_label = []
    for i in range(0, h - 1, ll):
        temp = []
        if (h - i - 1 <= 1.5 * ll and i + 2 * ll > h - 1):
            for content in new:
                if (content[1] >= i and content[1] <= h - 1):
                    temp.append(content)
            if (len(temp) != 0):
                temp.sort(key=lambda x: x[0])
                for content in temp:
                    new_label.append(content[2])
            break
        for content in new:
            if (content[1] >= i and content[1] <= i + ll - 1):
                temp.append(content)
        if (len(temp) != 0):
            temp.sort(key=lambda x: x[0])
            for content in temp:
                new_label.append(content[2])
    keep = new_label

    with open(save_dir + '/' + 'result.txt', 'w') as f:
        for i in keep:
            draw_box(im_result, (int(data[i][0]), int(data[i][1])),
                     (int(data[i][2]), int(data[i][3])), index=id, line_thickness=None)

            all_len = all_len + abs(data[i][3] - data[i][1]) * scale

            all_area = all_area + abs(abs(data[i][2] - data[i][0]) * scale * abs(data[i][3] - data[i][1]) * scale)

            f.write('id: ' + str(id) + ' ' + 'x: ' +
                    str((data[i][0] + data[i][2]) / 2 * scale) + ' ' + 'y: ' + str(
                (data[i][1] + data[i][3]) / 2 * scale) + ' ' +
                    'len: ' + str(abs(data[i][3] - data[i][1]) * scale) + ' ' + 'area: ' +
                    str(abs(data[i][2] - data[i][0]) * scale * abs(data[i][3] - data[i][1]) * scale) + '\n'
                    )
            result.append([(data[i][0] + data[i][2]) / 2 * scale, (data[i][1] + data[i][3]) / 2 * scale,
                           abs(data[i][3] - data[i][1]) * scale,
                           abs(data[i][2] - data[i][0]) * scale * abs(data[i][3] - data[i][1]) * scale])
            id = id + 1
    num = len(keep)

    with open(save_dir + '/' + 'result.txt', 'a') as f:
        f.write('num: ' + str(num) + ' ' + 'all_len: ' + str(all_len) + ' ' + 'area: ' + str(all_area) + '\n')

    view_xyz(result, z_name='len', save_path=save_dir)
    view_xyz(result, z_name='area', save_path=save_dir)

    view_xy(result, 'len', save_path=save_dir)
    view_xy(result, 'area', save_path=save_dir)
    cv2.imwrite(save_dir + '/' + 'result.jpg', im_result)
