import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorlayer.prepro import *
from config import config
import os
import scipy.misc
import datetime
import numpy as np
import math


"""
用于每个batch_size中对原始图像进行随机裁剪
"""
def crop_sub_imgs_fn(img, is_random=True):
    imgsize = config.TRAIN.imgsize
    scale = config.TRAIN.scale
    x = scale * imgsize

    # TODO: 添加边界判断 倍数整除验证
    # 现由于固定输入图像wh大小，所以这些预处理均失效，仅用于报错提示
    if min(img.shape[0:2]) <= x:
        x = min(img.shape[0:2])
        x = (x//scale)*scale
        print("ERROR: image size %d must larger than scale*imgsize: %d*%d=%d" %(x, scale, imgsize, scale*imgsize))

    # TODO: 裁剪后大小必须小于源wh的大小
    # if min(img.shape[0:2]) == x:
    #     x -= scale

    # 囧。。。这里的crop要求裁剪后的wh必须小于源wh
    result = crop(img, wrg=x, hrg=x, is_random=is_random)

    # TODO: 图像输入预处理
    # result = result / (255. / 2.)
    # result = result - 1.

    return result

"""
对随机裁剪后的target图像进行下采样构造输入图像
"""
def downsample_fn(img):
    scale = config.TRAIN.scale
    x = img.shape[0] // scale
    result = imresize(img, size=[x, x], interp='bicubic', mode=None)

    # TODO: 图像输入预处理
    # result = result / (255. / 2.)
    # result = result - 1.

    return result

"""
对输入图像进行bicubic上采样
"""
def lr2bicubic_fn(img):
    scale = config.TRAIN.scale
    result = imresize(img, size=[img.shape[0] * scale, img.shape[1] * scale], interp='bicubic', mode=None)

    return result

"""
修整齐大小尺寸
"""
def return_fn(img):
    scale = config.TRAIN.scale
    x = img.shape[0] // scale * scale
    y = img.shape[1] // scale * scale
    result = imresize(img, size=[x, y], interp='bicubic', mode=None)

    return result

"""
对验证集数据进行下采样
"""
def downsample_fn2(img):
    scale = config.TRAIN.scale
    x = img.shape[0] // scale
    y = img.shape[1] // scale
    result = imresize(img, size=[x, y], interp='bicubic', mode=None)

    # TODO: 图像输入预处理
    # result = result / (255. / 2.)
    # result = result - 1.

    return result


"""
加载数据集
"""
def load_dataset(data_dir):
    imgs = []
    img_files = os.listdir(data_dir)
    for img in img_files:
        tmp = scipy.misc.imread(data_dir + "/" + img)
        imgs.append(tmp)
        # print(tmp.shape)

    return imgs


"""
EDSR特制残差块
"""
def resBlock(x, channels=64, kernel_size=[3, 3], scale=1):
    tmp = slim.conv2d(x, channels, kernel_size, activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, channels, kernel_size, activation_fn=None)
    tmp *= scale
    return x + tmp

"""
Dense块
"""
def denseBlock(x, channels=64, kernel_size=[3, 3], scale=1, n_block=1):
    # 入口
    tmp = x = slim.conv2d(x, channels, [1, 1], activation_fn=None)

    for i in range(n_block):
        tmp = resBlock(tmp, channels, kernel_size=kernel_size, scale=scale)

    tmp *= scale
    return x + tmp

"""
EDSR-Tensorflow作者写的上采样模块
"""
# TODO: 后期可以替换掉？
def upsample(x, scale=2, features=64, activation=tf.nn.relu):
    assert scale in [2, 3, 4]
    x = slim.conv2d(x, features, [3, 3], activation_fn=activation)
    if scale == 2:
        ps_features = 3 * (scale ** 2)
        x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
        # x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x, 2, color=True)
    elif scale == 3:
        ps_features = 3 * (scale ** 2)
        x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
        # x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x, 3, color=True)
    elif scale == 4:
        ps_features = 3 * (2 ** 2)
        for i in range(2):
            x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
            # x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
            x = PS(x, 2, color=True)
    return x

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X

"""
Tensorflow log base 10.
Found here: https://github.com/tensorflow/tensorflow/issues/1666
"""
def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

"""
特别版：以tf.depth_to_space(X, r)为核心对wh进行重组，实现上采样。。。
"""
def upsample2(x, scale=2, activation=tf.nn.relu):
    # x[?, w, h, c]中的c会减少scale的2次方倍用于将输入扩展为
    # x[?, w*scale, h*scale, c/(scale**2)]
    x = tf.depth_to_space(x, scale)

    # 是否要再接卷积层还原再说。。。


"""
写loss, PSNR日志
"""
def log_message(ff="./log/test.log", mm="a", message=""):
    with open(ff, mm) as f:
        message = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + message
        message += "\n"
        f.write(message)


"""
监测梯度爆炸现象
"""
def monitorGradientExplosion(monitor_dir, image_i, image_o, image_t):
    # 创建目录
    os.makedirs(monitor_dir + 'image_i/')
    os.makedirs(monitor_dir + 'image_o/')
    os.makedirs(monitor_dir + 'image_t/')

    # 存图片
    for idx in range(len(image_i)):
        scipy.misc.imsave(monitor_dir + 'image_i/' + str(idx) + '.png', image_i[idx])
        scipy.misc.imsave(monitor_dir + 'image_o/' + str(idx) + '.png', image_o[idx])
        scipy.misc.imsave(monitor_dir + 'image_t/' + str(idx) + '.png', image_t[idx])


"""
计算PSNR-单通道
"""
def PSNR_t(sr, hr, shave_border=0):
    height, width = sr.shape[:2]
    sr = sr[shave_border:height - shave_border, shave_border:width - shave_border]
    hr = hr[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = sr - hr
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def PSNR(sr, hr, shave_border=0):
    height, width = sr.shape[:2]
    sr = sr[shave_border:height - shave_border, shave_border:width - shave_border]
    hr = hr[shave_border:height - shave_border, shave_border:width - shave_border]
    mse = np.mean((sr * 1. - hr * 1.) ** 2)

    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX ** 2 / mse)

