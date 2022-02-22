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
import sys
from pathlib import Path

import torch
import yaml

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

port = 0  # --master_port
path = Path('').resolve()
for last in path.rglob('*/**/last.pt'):
    ckpt = torch.load(last)
    if ckpt['optimizer'] is None:
        continue

    # Load opt.yaml
    with open(last.parent.parent / 'opt.yaml') as f:
        opt = yaml.load(f, Loader=yaml.SafeLoader)

    # Get device count
    d = opt['device'].split(',')  # devices
    nd = len(d)  # number of devices
    ddp = nd > 1 or (nd == 0 and torch.cuda.device_count() > 1)  # distributed data parallel

    if ddp:  # multi-GPU
        port += 1
        cmd = f'python -m torch.distributed.launch --nproc_per_node {nd} --master_port {port} train.py --resume {last}'
    else:  # single-GPU
        cmd = f'python train.py --resume {last}'

    cmd += ' > /dev/null 2>&1 &'  # redirect output to dev/null and run in daemon thread
    print(cmd)
    os.system(cmd)
