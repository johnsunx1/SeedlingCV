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


import argparse

import yaml

from wandb_utils import WandbLogger

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def create_dataset_artifact(opt):
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    logger = WandbLogger(opt, '', None, data, job_type='Dataset Creation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--project', type=str, default='YOLOv5', help='name of W&B Project')
    opt = parser.parse_args()
    opt.resume = False  # Explicitly disallow resume check for dataset upload job

    create_dataset_artifact(opt)
