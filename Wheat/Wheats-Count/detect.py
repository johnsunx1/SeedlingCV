'''
Copyright (c) 2022  https://github.com/14385423  https://github.com/ZCFzhaochuanfei  All rights reserved.

Reference source:

https://github.com/ZhihengCV/Bayesian-Crowd-Counting

https://github.com/dyh/unbox_yolov5_deepsort_counting

https://github.com/ultralytics/yolov5

'''

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
