import tensorflow as tf
import tensorlayer as tl
from config import config
from utils import *

# todo: 各类测试
# 基础参数加载
imgsize = config.TRAIN.imgsize
scale = config.TRAIN.scale
output_channels = config.TRAIN.output_channels
hr_img_path = config.TRAIN.hr_img_path
sr_img_path = config.TEST.sr_img_path

#Placeholder for image inputs
input = tf.placeholder(tf.float32, [None, imgsize, imgsize, output_channels])
output = tf.placeholder(tf.float32, [None, imgsize*scale, imgsize*scale, output_channels])

# BS数据预处理
# x = crop_sub_imgs_fn(input)

# 加载数据集
imgs = load_dataset(hr_img_path)
sample_imgs_target = tl.prepro.threading_data(imgs, fn=crop_sub_imgs_fn, is_random=True)
sample_imgs_input = tl.prepro.threading_data(sample_imgs_target, fn=downsample_fn)

# with tf.Session() as sess, tf.device('/gpu:0'):
#     test_feed = {input: x_}
#     x = sess.run(x, test_feed)
#     scipy.misc.imsave(sr_img_path + "result.jpg", x)

i = 0
for x in sample_imgs_target:
    scipy.misc.imsave(sr_img_path + "result" + str(i) + ".jpg", x)
    i += 1

