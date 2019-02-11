import tensorflow.contrib.slim as slim
import scipy.misc
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
import numpy as np
import shutil
import utils
import os
from config import config
import time
import math
import modellib
import random

# 解决Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Tensorflow GPU显存占满，而Util为0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 这里指定GPU0


# 添加了x和sk观察模块
class MYSR(object):
    def __init__(self):
        print("Building MYSR...")
        self.is_continued = False
        self.imgsize = config.TRAIN.imgsize
        self.output_channels = config.TRAIN.output_channels
        self.scale = config.TRAIN.scale
        self.epoch = config.TRAIN.n_epoch
        self.batch_size = config.TRAIN.batch_size
        self.save_model_dir = config.TRAIN.save_model_dir
        self.test_input = config.TEST.hr_img_path
        self.test_output = config.TEST.sr_img_path

        # 初始化log文件
        utils.log_message(config.VALID.log_file, "w+", "START")
        utils.log_message(config.TEST.log_file, "w+", "START")

        # Placeholder for image inputs
        # self.input = x = tf.placeholder(tf.float32, [None, self.imgsize, self.imgsize, self.output_channels])
        self.input = x = tf.placeholder(tf.float32, [None, None, None, self.output_channels])
        self.input_bicubic = tf.placeholder(tf.float32, [None, None, None, self.output_channels])
        # Placeholder for upscaled image ground-truth
        # self.target = y = tf.placeholder(tf.float32, [None, self.imgsize*self.scale, self.imgsize*self.scale, self.output_channels])
        self.target = y = tf.placeholder(tf.float32, [None, None, None, self.output_channels])

        # 输入预处理
        # TODO: 后边有relu层，注意将输入图像收缩至[-1, 1]区间是否合适
        image_input = x / (255. / 2.)
        image_input -= 1
        image_target = y / (255. / 2.)
        image_target -= 1
        image_input_bicubic = self.input_bicubic / (255. / 2.)
        image_input_bicubic -= 1

        # TODO:加载模型
        output, output_x, output_sk = modellib.MYSR_v5_Dense2_test(self, image_input, 64, 4, 16)

        # 结果 注意预处理的值
        self.out = tf.clip_by_value((output+1)*(255. / 2.), 0.0, 255.0)
        self.out_x = tf.clip_by_value((output_x + 1) * (255. / 2.), 0.0, 255.0)
        self.out_sk = tf.clip_by_value((output_sk + 1) * (255. / 2.), 0.0, 255.0)
        self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(image_target, output))

        # Calculating Peak Signal-to-noise-ratio
        # Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        mse = tf.reduce_mean(tf.squared_difference((image_target+1)*(255. / 2.), tf.clip_by_value((output+1)*(255. / 2.), 0.0, 255.0)))
        PSNR = tf.constant(255**2, dtype=tf.float32) / mse
        self.PSNR = PSNR = tf.constant(10, dtype=tf.float32) * utils.log10(PSNR)

        # Scalar to keep track for loss
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("PSNR", PSNR)
        # Image summaries for input, target, and output
        tf.summary.image("input_image", tf.cast(self.input, tf.uint8))
        tf.summary.image("target_image", tf.cast(self.target, tf.uint8))
        tf.summary.image("output_image", tf.cast(self.out, tf.uint8))
        tf.summary.image("output_x_image", tf.cast(self.out_x, tf.uint8))
        tf.summary.image("output_sk_image", tf.cast(self.out_sk, tf.uint8))

        # Tensorflow graph setup... session, saver, etc.
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_tf)
        self.saver = tf.train.Saver()
        print("Done building!")

    def evaluate(self):
        print("Begin loading data...")
        test_hr_img_list = sorted(
            tl.files.load_file_list(path=config.TEST.hr_img_path, regx='.*.png', printable=False))
        test_hr_imgs = tl.vis.read_images(test_hr_img_list, path=config.TEST.hr_img_path, n_threads=32)
        print("Done loading!")

        ## 初始化模型训练参数
        # Just a tf thing, to merge all summaries into one
        merged = tf.summary.merge_all()
        # Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer()
        # This is the train operation for our objective
        train_op = optimizer.minimize(self.loss)
        # Operation to initialize all variables
        # init = tf.global_variables_initializer()   # 已过时
        init = tf.initializers.global_variables()

        # GPU
        with self.sess as sess, tf.device('/gpu:0'):
            # Initialize all variables
            sess.run(init)
            # 恢复模型
            print("Restoring...")
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_model_dir))
            print("Restored!")

            # 测试集集专用TB writer
            test_writer = tf.summary.FileWriter(config.TRAIN.save_tensorboard_test_dir, sess.graph)

            # 验证集数据预处理
            b_test_hr_imgs = tl.prepro.threading_data(test_hr_imgs, fn=utils.return_fn)
            b_test_lr_imgs = tl.prepro.threading_data(b_test_hr_imgs, fn=utils.downsample_fn2)
            b_test_bicubic_imgs = tl.prepro.threading_data(b_test_lr_imgs, fn=utils.lr2bicubic_fn)

            # PSNR值
            ll_PSNR = []
            ss = 0.0
            for idx in tqdm(range(0, len(b_test_lr_imgs))):
                # run
                test_feed = {
                    self.input: [b_test_lr_imgs[idx]],
                    self.input_bicubic: [b_test_bicubic_imgs[idx]],
                    self.target: [b_test_hr_imgs[idx]]
                }
                t_summary, out, out_x, out_sk, PSNR = sess.run([merged, self.out, self.out_x, self.out_sk, self.PSNR], test_feed)

                # 记录到tensorboard
                test_writer.add_summary(t_summary, idx)

                # 保存sr图像
                ll_PSNR.append(test_hr_img_list[idx] + "-" + PSNR.astype('str'))
                ss += PSNR
                scipy.misc.imsave(self.test_output + test_hr_img_list[idx], out[0])
                scipy.misc.imsave(self.test_output + 'x_' + test_hr_img_list[idx], out_x[0])
                scipy.misc.imsave(self.test_output + 'sk_' + test_hr_img_list[idx], out_sk[0])

            ll_PSNR.append("AVG: " + (ss/len(b_test_lr_imgs)).astype('str'))
            print(ll_PSNR)


# test
network = MYSR()

network.evaluate()
