'''
Copyright (c) 2022  https://gitee.com/l1233   https://gitee.com/zhao-chuanfei  All rights reserved.

Reference source:

https://github.com/ZhihengCV/Bayesian-Crowd-Counting

https://github.com/dyh/unbox_yolov5_deepsort_counting

https://github.com/ultralytics/yolov5

'''


from glob import glob
import re
from pathlib import Path


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)

    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")

    dir = path if path.suffix == '' else path.parent

    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)

    return path
