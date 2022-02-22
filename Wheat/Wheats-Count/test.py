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
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from utils import getdir
from scipy.interpolate import make_interp_spline

args = None


def view_pred(m, save_path):
    plt.imshow(m, cmap=CM.jet)
    plt.savefig(save_path)
    plt.close()


def view_result(x, truth, pred, percents, save_path):
    plt.figure(figsize=(32, 8))
    fig, ax = plt.subplots(2, 1)
    x = np.array(x)
    truth = np.array(truth)
    pred = np.array(pred)
    percents = np.array(percents)
    mean_percent = np.mean(percents)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    truth_smooth = make_interp_spline(x, truth)(x_smooth)
    pred_smooth = make_interp_spline(x, pred)(x_smooth)
    percents_smooth = make_interp_spline(x, percents)(x_smooth)
    ax[0].plot(x_smooth, truth_smooth, label='truth', c='red')
    ax[0].plot(x_smooth, pred_smooth, label='pred', c='blue')
    # ax[0].set_xlabel("jpgs")
    ax[0].set_ylabel("wheat-num")
    ax[0].legend(loc='best')
    ax[0].set_title('mae:' + str(mae) + '   ' + 'mse:' + str(mse) + '\n' +
                    'mean_accuracy:' + str(mean_percent) + '%', c='brown')
    ax[1].plot(x_smooth, percents_smooth, label='accuracy', c='blue')
    ax[1].set_xlabel("jpgs")
    ax[1].set_ylabel("percent(%)")
    ax[1].legend(loc='best')
    plt.savefig(save_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')

    parser.add_argument('--data-dir', default='BayData/test',
                        help='the dir of  the data to test ')

    parser.add_argument('--weights-dir', default='weights/best_model.pth',
                        help='model directory')

    parser.add_argument('--device', default='0', help='assign device')

    parser.add_argument('--out-test-dir', default='out_test/result',
                        help='the result of test')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    datasets = Crowd(args.data_dir, 512, 8, is_gray=False, method='val')

    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=1, pin_memory=False)
    model = vgg19()

    device = torch.device('cuda')

    model.to(device)

    model.load_state_dict(torch.load(args.weights_dir, device))

    result_dir = args.out_test_dir
    result_dir = getdir.increment_path(result_dir)
    result_dir = str(result_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    epoch_minus = []
    num = 1
    x = []
    truth = []
    pred = []
    percents = []
    with open(result_dir + '/' + 'result.txt', 'w') as f:
        for inputs, count, name in dataloader:
            inputs = inputs.to(device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1'
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                ans = torch.sum(outputs).item()
                temp_minu = count[0].item() - ans
                predict = outputs.data.cpu().numpy()

                view_pred(predict[0][0], save_path=result_dir + '/' + name[0])
                x.append(num)
                num = num + 1
                truth.append(count[0].item())
                pred.append(ans)
                percent = (1 - abs(temp_minu) / count[0].item()) * 100
                percents.append(percent)

                print(name[0] + '.jpg', temp_minu, count[0].item(), torch.sum(outputs).item())

                f.write(name[0] + '.jpg   ' + 'truth - predict:  ' + str(temp_minu)
                        + '   truth :   ' + str(count[0].item()) + '   ' + 'predict:   ' + str(ans) + '\n')
                epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    view_result(x=x, truth=truth, pred=pred, percents=percents, save_path=result_dir + '/' + 'view_result')
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
    with open(result_dir + '/' + 'result.txt', 'a') as f:
        f.write(log_str + '\n')
    f.close()
