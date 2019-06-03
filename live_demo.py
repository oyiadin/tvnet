# coding=utf-8
import cv2
import time
import numpy as np

import torch
from torchver.tvnet import TVNet


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

height, width = cap.read()[1].shape[:2]
height = int(height / 8)
width = int(width / 8)


def preprocess(frame):
    return cv2.flip(cv2.resize(frame, (width, height)), 1).transpose(2, 0, 1)


frame1 = preprocess(cap.read()[1])
frame2 = np.zeros_like(frame1)
frames = np.array([frame2, frame1])  # (N, C, H, W)


scale = 1
warp = 1
iteration = 1

tvnet = TVNet()
flow_mat = np.zeros((height, width, 3))

from_time = time.time()
number_of_frames = 0

while True:
    frames[0] = frames[1]
    frames[1] = preprocess(cap.read()[1])

    x1 = torch.Tensor(frames[0][None, ...], device=torch.device('cpu'))
    x2 = torch.Tensor(frames[1][None, ...], device=torch.device('cpu'))

    u1, u2, rho = tvnet.tvnet_flow(x1, x2,
                                   max_scales=scale,
                                   warps=warp,
                                   max_iterations=iteration)

    flow_mat[:, :, 0] = np.squeeze(u1.detach().numpy())
    flow_mat[:, :, 1] = np.squeeze(u2.detach().numpy())

    number_of_frames += 1
    fps = int(number_of_frames / (time.time() - from_time))

    raw_frame = frames[0].transpose(1, 2, 0).copy()
    cv2.putText(raw_frame,
                '{:>2} FPS'.format(fps),
                (10, 20),
                font,
                0.5,
                (0, 0, 0))
    final = np.hstack([raw_frame / 255, flow_mat])
    cv2.imshow('TVNet Demo', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
