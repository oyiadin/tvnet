import time
t = time.time()

import os
import random
import shutil

import cv2
import numpy as np
import tensorflow as tf
# import scipy.io as sio
from scipy.misc import imsave
from torchver.tvnet import TVNet
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable

# flags = tf.app.flags
# flags.DEFINE_integer("scale", 5, " TVNet scale [3]")
# flags.DEFINE_integer("warp", 5, " TVNet warp [1]")
# flags.DEFINE_integer("iteration", 50, " TVNet iteration [10]")
# flags.DEFINE_string("gpu", '0', " gpu to use [0]")
# FLAGS = flags.FLAGS
#
# scale = FLAGS.scale
# warp = FLAGS.warp
# iteration = FLAGS.iteration
# if int(FLAGS.gpu > -1):
#     os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
#
# print('TVNet Params:\n scale: %d\n warp: %d\n iteration: %d\nUsing gpu: %s' \
#       % (scale, warp, iteration, FLAGS.gpu))

scale = 1
warp = 1
iteration = 1


lines = open('/ds/hmdb_frames/mini_test.txt').readlines()
filename, label = random.choice(lines).split()
basedir = filename[:-len('.avi')]
path = os.path.join('/ds/hmdb_frames/', label, basedir)
frames = [f for f in os.listdir(path) \
          if os.path.isfile(os.path.join(path, f))]

images = []
for i in range(0, len(frames)):
    # load image
    images.append(cv2.imread(os.path.join(path, '%05d.jpg' % (i+1))))

images = np.array(images).transpose([0, 3, 1, 2])  # (N, C, H, W)

N = images.shape[0]
_ = cv2.imread(os.path.join(path, '00001.jpg'))
h, w, c = _.shape

# model construction
x1 = torch.Tensor(images[:N-1, ...], device=torch.device('cpu'))
x2 = torch.Tensor(images[1:, ...], device=torch.device('cpu'))
tvnet = TVNet()
u1, u2, rho = tvnet.tvnet_flow(x1,x2,max_scales=scale,
                     warps=warp,
                     max_iterations=iteration)

# run model
u1_np, u2_np = u1.detach().numpy(), u2.detach().numpy()

u1_np = np.squeeze(u1_np)
u2_np = np.squeeze(u2_np)

flow_mat = np.zeros([images.shape[0]-1, h, w, 2])
flow_mat[:, :, :, 0] = u1_np
flow_mat[:, :, :, 1] = u2_np

print(flow_mat.shape)

if os.path.exists('result'):
    shutil.rmtree('result')
os.mkdir('result')


flow_mat = np.concatenate((flow_mat, np.zeros((flow_mat.shape[0], flow_mat.shape[1], flow_mat.shape[2], 1))),
                   axis=3)

for i in range(flow_mat.shape[0]):
    print(i)
    res_path = os.path.join('result', '{}.png'.format(i))
    imsave(res_path, flow_mat[i])

print(path)

print((time.time() - t) / i)