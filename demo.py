import os
import random
import shutil

import cv2
import numpy as np
import tensorflow as tf
# import scipy.io as sio
from scipy.misc import imsave
from tvnet import TVNet
from torch.utils.data import DataLoader

flags = tf.app.flags
flags.DEFINE_integer("scale", 5, " TVNet scale [3]")
flags.DEFINE_integer("warp", 5, " TVNet warp [1]")
flags.DEFINE_integer("iteration", 50, " TVNet iteration [10]")
flags.DEFINE_string("gpu", '0', " gpu to use [0]")
FLAGS = flags.FLAGS

scale = FLAGS.scale
warp = FLAGS.warp
iteration = FLAGS.iteration
if int(FLAGS.gpu > -1):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

print 'TVNet Params:\n scale: %d\n warp: %d\n iteration: %d\nUsing gpu: %s' \
      % (scale, warp, iteration, FLAGS.gpu)


lines = open('/ds/hmdb_frames/split1_train.txt').readlines()
filename, label = random.choice(lines).split()
basedir = filename[:-len('.avi')]
path = os.path.join('/ds/hmdb_frames/', label, basedir)
frames = [f for f in os.listdir(path) \
          if os.path.isfile(os.path.join(path, f))]

_ = cv2.imread(os.path.join(path, '00001.jpg'))
h, w, c = _.shape

# model construction
x1 = tf.placeholder(shape=[None, h, w, 3], dtype=tf.float32)
x2 = tf.placeholder(shape=[None, h, w, 3], dtype=tf.float32)
tvnet = TVNet()
u1, u2, rho = tvnet.tvnet_flow(x1,x2,max_scales=scale,
                     warps=warp,
                     max_iterations=iteration)

# init
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
sess.run(tf.global_variables_initializer())


images = []
for i in range(0, len(frames)):
    # load image
    images.append(cv2.imread(os.path.join(path, '%05d.jpg' % (i+1))))

images = np.array(images)


# run model
u1_np, u2_np = sess.run([u1, u2], feed_dict={x1: images[:-1, ...], x2: images[1:, ...]})

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
