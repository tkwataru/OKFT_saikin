#!/usr/bin/env python
"""Semantic segmentation for Saikin. Inference Saikin video.
"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from scipy import ndimage
import csv
import cv2

import six
import six.moves.cPickle as pickle
from six.moves import queue

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from matplotlib.ticker import *
from chainer import serializers
from chainer import cuda
import chainercv

from saikinNet import SaikinNet

#
# Arguments.
#
parser = argparse.ArgumentParser(
    description='Saikin semantic segmentation')
parser.add_argument('video', help='Path to inference video file')
parser.add_argument('--out_root', '-R', default='./Segmentation', help='Root directory path of segmentation files')
parser.add_argument('--csv', '-c', default='Accuracy.csv', help='Path to Accuracy text file')
parser.add_argument('--model', '-m', default='model', help='Path to model file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--RotationFlip', '-rf', default=False, type=bool, help='Flag of image rotation and flip')
args = parser.parse_args()

#
# Preprocessing.
#
model = SaikinNet()  # Initialize a neural network model.

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

serializers.load_npz(args.model, model)  # Load trained parameters of the model.
try:
    os.mkdir(args.out_root)
except FileExistsError:
    pass

"""
## Visualize Filter ============================
# print(model.conv1.W.shape)
# print(model.conv1.W.data[0,0])

n1, n2, h, w = model.conv1.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv1.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
# plt.savefig(args.model+'_conv1.png',dsp=150)
# cv2.waitKey(0)

n1, n2, h, w = model.conv2.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv2.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# cv2.waitKey(0)

n1, n2, h, w = model.conv3.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv3.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

cv2.waitKey(0)
# ============================================
"""

# model.to_cpu()                     # Use CPU for chainer.
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()  # Set a GPU device number.
    model.to_gpu()  # Transfer the model to GPU

chainer.config.cudnn_deterministic = True  # Set deterministic mode for cuDNN.
chainer.config.train = False  # Set evaluation mode for Chainer.


# Convert label to RGB
def label2RGB(label):
    rgb = np.zeros((label.shape[0], label.shape[1], 3), np.uint8)

    rgb[:, :, 0] = 255 * (label == 0)
    rgb[:, :, 1] = 255 * (label == 1)
    rgb[:, :, 2] = 255 * (label == 2)

    return rgb


# Convert score to label
def PDF2label(pdf):
    label = np.argmax(pdf, axis=2).astype(np.int32)  # Find a class of maximum score

    return label


#
# Main loop.
#
if __name__ == '__main__':
    cap = cv2.VideoCapture(args.video)  # Open an input video file.

    dir, file = os.path.split(args.video)
    name, ext = os.path.splitext(file)
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Set output CODEC
    fps = cap.get(cv2.CAP_PROP_FPS)  # set output fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # set output video size
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # set output video size
    out = cv2.VideoWriter(args.out_root + '\\' + name + '_seg.avi', fourcc, fps, (width, height))   # Open an output video file.

    count = 0
    while (cap.isOpened()):  # frame loop.
        count += 1
        print(count, '/', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        ret, frame = cap.read()  # Read a frame.
        if not ret:  # Video end
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255  # Convert a color image to gray scale. Normalize pixel values to 0.0 - 1.0
        gray = gray[np.newaxis, np.newaxis, :]    # Convert dimension to chainer form.

        t0 = time.perf_counter() * 1000  # Start timer
        x = chainer.Variable(xp.asarray(gray))  # Transfer an image to GPU. Set an image to chainer input.
        hmap = chainer.cuda.to_cpu(model.predict(x).data[0, :, :])  # Calculate foward propagation. Transfer a score to CPU.
        t1 = time.perf_counter() * 1000  # Stop timer
        print('%f[msec]' % (t1 - t0))

        hmap = (hmap * 255).astype(np.uint8).transpose(1, 2, 0)  # Convert a score to RGB form
        b, r = hmap[:, :, 0],  hmap[:, :, 1]
        g = np.zeros((model.DSTY, model.DSTX), np.uint8)
        pdf = cv2.merge((b, g, r))  # Change Saikin color to red.
        seg_label = PDF2label(hmap)  # Convert a score to a label.
        seg = label2RGB(seg_label)  # Convert a label to RGB form.
        b, g, r = cv2.split(seg)
        seg = cv2.merge((b, r, g))  # Change Saikin color to red.

        # layer1 = pdf // 4  # Score image
        layer1 = seg // 4   # Segmentation image
        layer2 = frame // 4 * 3    # Input image

        cv2.imshow(args.video, layer1 + layer2)  # Display a superimposed image.

        # cv2.waitKey(0)
        # cv2.waitKey(1000)
        cv2.waitKey(1)

        out.write(layer1 + layer2)  # Output a superimposed image to the video file.

        print('')

    cap.release()   # Close the input video file.
    out.release()   # Close the output video file.

cv2.destroyAllWindows()  # Close all display windows.
