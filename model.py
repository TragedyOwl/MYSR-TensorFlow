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
        self.epoch = config.TRAIN.n_epoch
        self.batch_size = config.TRAIN.batch_size
        self.save_model_dir = config.TRAIN.save_model_dir

        # Placeholder for image inputs
        # self.input = x = tf.placeholder(tf.float32, [None, self.imgsize, self.imgsize, self.output_channels])
        self.input = x = tf.placeholder(tf.float32, [None, None, None, self.output_channels])
        # Placeholder for upscaled image ground-truth
        self.target = y = tf.placeholder(tf.float32, [None, self.imgsize*self.scale, self.imgsize*self.scale, self.output_channels])

        # 验证集Placeholder
        self.valid_input = []
        self.valid_target = []

        # 输入预处理
        # result = result / (255. / 2.)
        # TODO: 后边有relu层，注意将输入图像收缩至[-1, 1]区间是否合适
        # result = result - 1.
        # image_input = x - (255. / 2.)
        # image_input = image_input - 1
        # image_target = y - (255. / 2.)
        # image_target = image_target - 1

        image_input = x / (255. / 2.)
        image_input = image_input - 1
        image_target = y / (255. / 2.)
        image_target = image_target - 1


        # 貌似即使收缩至[-1, 1]区间，卷积层依旧可以有效适应，只是注意最后一层不能使用relu，b毕竟relu值域在[0, x]
        # ENCODER
        # 入口
        # One convolution before res blocks and to convert to required feature depth
        x = slim.conv2d(image_input, 256, [3, 3])

        conv_1 = x

        # scaling_factor = 0.1
        scaling_factor = 0.1

        # Add the residual blocks to the model
        for i in range(12):
            x = utils.resBlock(x, 256, scale=scaling_factor)

        # One more convolution, and then we add the output of our first conv layer
        x = slim.conv2d(x, 256, [3, 3])
        x += conv_1

        # Upsample output of the convolution
        # x = utils.upsample(x, self.scale, 256, None)

        # TODO:试试新的上采样
        x = tf.depth_to_space(x, self.scale)
        x = slim.conv2d(x, self.output_channels, [3, 3], activation_fn=tf.nn.tanh)


        # One final convolution on the upsampling output
        output = x  # slim.conv2d(x,output_channels,[3,3])

        # 结果 注意预处理的值
        # self.out = tf.clip_by_value(output+(255. / 2.), 0.0, 255.0)
        self.out = tf.clip_by_value((output+1)*(255. / 2.), 0.0, 255.0)
        self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(image_target, output))

        # Calculating Peak Signal-to-noise-ratio
        # Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        mse = tf.reduce_mean(tf.squared_difference((image_target+1)*(255. / 2.), tf.clip_by_value((output+1)*(255. / 2.), 0.0, 255.0)))
        PSNR = tf.constant(255**2, dtype=tf.float32) / mse
        PSNR = tf.constant(10, dtype=tf.float32) * utils.log10(PSNR)

        # Scalar to keep track for loss
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("PSNR", PSNR)
        # Image summaries for input, target, and output
        tf.summary.image("input_image", tf.cast(self.input, tf.uint8))
        tf.summary.image("target_image", tf.cast(self.target, tf.uint8))
        tf.summary.image("output_image", tf.cast(self.out, tf.uint8))

        # Tensorflow graph setup... session, saver, etc.
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_tf)
        self.saver = tf.train.Saver()
        print("Done building!")

    def train(self):
        ###====================== PRE-LOAD DATA ===========================###
        print("Begin loading data...")
        # 训练集
        train_hr_img_list = sorted(
            tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
        train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)

        # TODO: 添加验证集测试
        # 验证集，不裁剪，整体下采样
        # valid_hr_img_list = sorted(
        #     tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
        # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
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
        print("Begin training...")

        # GPU
        with self.sess as sess, tf.device('/gpu:0'):
            # Initialize all variables
            sess.run(init)
            # 训练集专用TB writer
            train_writer = tf.summary.FileWriter(config.TRAIN.save_tensorboard_train_dir, sess.graph)

            # 验证集专用TB writer
            valid_writer = tf.summary.FileWriter(config.TRAIN.save_tensorboard_valid_dir, sess.graph)

            # TODO: 验证集预处理，由于输入为整张图片，所以从处理到输入都不同
            # 验证集数据预处理
            # b_valid_hr_imgs = valid_hr_imgs
            # b_valid_lr_imgs = tl.prepro.threading_data(valid_hr_imgs, fn=utils.downsample_fn2)
            # 为验证集数据单独创建Place_holder
            # for i in range(len(b_valid_hr_imgs)):
            #     self.valid_input.append(tf.placeholder(tf.float32, [1, b_valid_lr_imgs[i].shape[0], b_valid_lr_imgs[i].shape[1], self.output_channels]))
            #     self.valid_target.append(tf.placeholder(tf.float32, [1, b_valid_hr_imgs[i].shape[0], b_valid_hr_imgs[i].shape[1], self.output_channels]))


            # TODO: 开始训练
            for epoch in range(0, self.epoch + 1):
                epoch_time = time.time()
                # TODO: 可以添加对学习率的处理

                for idx in tqdm(range(0, math.ceil(len(train_hr_imgs)/self.batch_size))):
                    step_tim = time.time()
                    # 加载随机处理后的HR, LR数据
                    b_train_hr_imgs_crop = tl.prepro.threading_data(train_hr_imgs[idx*self.batch_size:idx*self.batch_size + self.batch_size], fn=utils.crop_sub_imgs_fn,
                                                          is_random=True)
                    b_train_lr_imgs_crop = tl.prepro.threading_data(b_train_hr_imgs_crop, fn=utils.downsample_fn)

                    # run
                    feed = {
                        self.input: b_train_lr_imgs_crop,
                        self.target: b_train_hr_imgs_crop
                    }
                    summary, _ = sess.run([merged, train_op], feed)

                    # 记录到tensorboard
                    train_writer.add_summary(summary, epoch*self.batch_size + idx)

                step_time = time.time()
                print("Epoch [%2d/%2d] %4d time: %4.4fs" % (
                epoch, self.epoch, self.epoch,  step_time - epoch_time))

                # TODO: 每n个epoch保存一次模型
                if epoch % 1000 == 0 and epoch != 0:
                    self.saver.save(self.sess, self.save_model_dir, global_step=epoch)

                # TODO: 每n个epoch运行一下验证集
                if epoch % 10 == 0:
                    # 每张测试集图片都需要单独过模型
                    # 创建place_holder
                    pass

    def save(self):
        print("Saving...")
        self.saver.save(self.sess, self.save_model_dir)
        print("Saved!")

    def resume(self):
        print("Restoring...")
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_model_dir))
        print("Restored!")




























