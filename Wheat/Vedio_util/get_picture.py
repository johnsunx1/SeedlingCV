'''
Copyright (c) 2022  https://github.com/14385423  https://github.com/ZCFzhaochuanfei  All rights reserved.

Reference source:

https://github.com/ZhihengCV/Bayesian-Crowd-Counting

https://github.com/dyh/unbox_yolov5_deepsort_counting

https://github.com/ultralytics/yolov5

'''


import cv2
cap = cv2.VideoCapture('./wheat_8s.mp4')
total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(total)
print(fps)
print(int(total / fps))

step = 15
lis_img = []
for i in range(1, int(total), int(step)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    cap.grab()
    ret, frame = cap.read()
    cv2.imwrite('picture'+'/'+str(i) + '.jpg', frame)
