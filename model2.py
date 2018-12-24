import tensorflow.contrib.slim as slim
import scipy.misc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
import utils
import os
from config import config

# 解决Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Tensorflow GPU显存占满，而Util为0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"	# 这里指定GPU0


class MYSR(object):
    def __init__(self):
        print("Building MYSR...")
        self.imgsize = config.TRAIN.imgsize
        self.output_channels = config.TRAIN.output_channels
        self.scale = config.TRAIN.scale

        # Placeholder for image inputs
        self.input = x = tf.placeholder(tf.float32, [None, None, None, self.output_channels])
        # Placeholder for upscaled image ground-truth
        self.target = y = tf.placeholder(tf.float32, [None, None, None, self.output_channels])

        # 输入预处理
        # result = result / (255. / 2.)
        # TODO: 后边有relu层，注意将输入图像收缩至[-1, 1]区间是否合适
        # result = result - 1.
        image_input = x / (255. / 2.)
        image_input = image_input - 1
        image_target = y / (255. / 2.)
        image_target = image_target - 1

        # 貌似即使收缩至[-1, 1]区间，卷积层依旧可以有效适应，只是注意最后一层不能使用relu，b毕竟relu值域在[0, x]
        # ENCODER
        # 入口
        x = slim.conv2d(image_input, 64, [5, 5])   # 入口适当大点？
        conv_1 = x

        # ENCODER-resBlock-64
        scaling_factor = 1
        for i in range(3):
            x = utils.resBlock(x, 64, scale=scaling_factor)

        x = slim.conv2d(image_input, 128, [3, 3])

        # ENCODER-resBlock-128
        scaling_factor = 1
        for i in range(4):
            x = utils.resBlock(x, 128, scale=scaling_factor)

        x = slim.conv2d(image_input, 256, [3, 3])

        # ENCODER-resBlock-256
        scaling_factor = 1
        for i in range(5):
            x = utils.resBlock(x, 256, scale=scaling_factor)

        # Upsample output of the convolution
        x = utils.upsample(x, self.scale, 128, None)

        # DECODER-resBlock-64
        scaling_factor = 0.1
        for i in range(4):
            x = utils.resBlock(x, 64, scale=scaling_factor)

        # DECODER-resBlock-32
        scaling_factor = 0.1
        for i in range(3):
            x = utils.resBlock(x, 32, scale=scaling_factor)

        # DECODER-resBlock-16
        scaling_factor = 0.1
        for i in range(2):
            x = utils.resBlock(x, 16, scale=scaling_factor)

        # DECODER-resBlock-8
        scaling_factor = 0.1
        for i in range(1):
            x = utils.resBlock(x, 8, scale=scaling_factor)

        # DECODER-输出
        # TODO: 貌似这里直接使用conv会破坏逐步修复结构？反而会因为缺少feature_map而导致精细度降低？
        x = slim.conv2d(x, self.output_channels, [3, 3])

        output = x

        # 结果
        self.out = tf.clip_by_value((output+1)*(255. / 2.), 0.0, 255.0)
        self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(image_target, output))

        # Calculating Peak Signal-to-noise-ratio
        # Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        mse = tf.reduce_mean(tf.squared_difference(image_target, output))
        PSNR = tf.constant(255 ** 2, dtype=tf.float32) / mse
        PSNR = tf.constant(10, dtype=tf.float32) * utils.log10(PSNR)

        # Scalar to keep track for loss
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("PSNR", PSNR)
        # Image summaries for input, target, and output
        tf.summary.image("input_image", tf.cast(self.input, tf.uint8))
        tf.summary.image("target_image", tf.cast(self.target, tf.uint8))
        tf.summary.image("output_image", tf.cast(self.out, tf.uint8))

        # Tensorflow graph setup... session, saver, etc.
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        print("Done building!")



























