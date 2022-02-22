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



import torch
import os
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from utils import getdir

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Detect ')

    parser.add_argument('--data-dir', default='Detect-processed/detect',
                        help='processed data to detect')

    parser.add_argument('--weights-dir', default='weights/best_model.pth',
                        help='model directory')

    parser.add_argument('--device', default='0', help='assign device')

    parser.add_argument('--out-detect-dir', default='out_detect/result', help='the result dir')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    datasets = Crowd(args.data_dir, 512, 8, is_gray=False,
                     method='detect')

    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=1, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(args.weights_dir, device))

    result_dir = args.out_detect_dir

    result_dir = getdir.increment_path(result_dir)
    result_dir = str(result_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ans_sum = 0
    num = 0

    with open(result_dir + '/' + 'result.txt', 'w') as f:
        for inputs, name in dataloader:
            inputs = inputs.to(device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1'
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                predict = outputs.data.cpu().numpy()
                plt.figure(2)
                plt.imshow(predict[0][0], cmap=CM.jet)
                plt.savefig(result_dir + '/' + name[0])
                ans = torch.sum(outputs).item()
                print(name[0] + '.jpg', torch.sum(outputs).item())

                f.write(name[0] + '.jpg   ' + 'predict_num:   ' + str(ans) + '\n')

                ans_sum = ans_sum + ans
                num = num + 1

    with open(result_dir + '/' + 'result.txt', 'a') as f:
        f.write('img_num: ' + str(num) + '   ' + 'predict_sum:   ' + str(ans_sum) + '\n')
        f.close()
