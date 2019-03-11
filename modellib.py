import tensorflow.contrib.slim as slim
import tensorflow as tf
import utils
import Conv2DWN

def SR_CNN(self, image_input, k):

    x = slim.conv2d(image_input, self.output_channels * self.scale * self.scale, [k, k], activation_fn=None)
    x = tf.depth_to_space(x, self.scale)

    output = x

    return output


def VDSR_v1(self, image_input, image_input_bicubic, num_channels, num_block):
    # skip connect
    # sk = image_input_bicubic

    # conv1
    x = slim.conv2d(image_input_bicubic, num_channels, [3, 3])
    x = tf.nn.relu(x)

    for i in range(num_block):
        x = slim.conv2d(x, num_channels, [3, 3])
        x = tf.nn.relu(x)

    # conv-output
    x = slim.conv2d(x, self.output_channels, [3, 3])
    x = tf.nn.relu(x)

    # x += sk

    output = x

    return output


def VDSR_v1_b(self, image_input, image_input_bicubic, num_channels, num_block):
    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 7, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    # conv1
    x = slim.conv2d(image_input_bicubic, num_channels, [3, 3])
    x = tf.nn.relu(x)

    for i in range(num_block):
        x = slim.conv2d(x, num_channels, [3, 3])
        x = tf.nn.relu(x)

    # conv-output
    x = slim.conv2d(x, self.output_channels, [3, 3])
    x = tf.nn.relu(x)

    x += sk

    output = x

    return output


def EDSR_v1(self, image_input, num_channels, num_block):

    x = slim.conv2d(image_input, num_channels, [3, 3])

    conv_1 = x

    # scaling_factor = 0.1
    scaling_factor = 0.1

    # Add the residual blocks to the model
    for i in range(num_block):
        x = utils.resBlock(x, num_channels, scale=scaling_factor)

    # One more convolution, and then we add the output of our first conv layer
    x = slim.conv2d(x, num_channels, [3, 3])
    x += conv_1

    # Upsample output of the convolution
    # x = utils.upsample(x, self.scale, 256, None)

    # TODO:试试新的上采样
    x = slim.conv2d(x, self.output_channels * self.scale * self.scale, [3, 3], activation_fn=tf.nn.tanh)
    x = tf.depth_to_space(x, self.scale)

    # One final convolution on the upsampling output
    output = x # slim.conv2d(x,output_channels,[3,3])
    return output


# 添加了额外的底层跳过连接
def EDSR_v1_b(self, image_input, num_channels, num_block):
    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    x = slim.conv2d(image_input, num_channels, [3, 3])

    conv_1 = x

    # scaling_factor = 0.1
    scaling_factor = 0.1

    # Add the residual blocks to the model
    for i in range(num_block):
        x = utils.resBlock(x, num_channels, scale=scaling_factor)

    # One more convolution, and then we add the output of our first conv layer
    x = slim.conv2d(x, num_channels, [3, 3])
    x += conv_1

    # Upsample output of the convolution
    # x = utils.upsample(x, self.scale, 256, None)

    # TODO:试试新的上采样
    x = slim.conv2d(x, self.output_channels * self.scale * self.scale, [3, 3], activation_fn=tf.nn.tanh)
    x = tf.depth_to_space(x, self.scale)

    # One final convolution on the upsampling output
    output = x + sk  # slim.conv2d(x,output_channels,[3,3])
    return output

# 添加了额外的底层跳过连接
# 判断底层跳过连接是否可以吸收修复结构内的跳过连接
def EDSR_v1_b(self, image_input, num_channels, num_block):
    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    x = slim.conv2d(image_input, num_channels, [3, 3])

    # conv_1 = x

    # scaling_factor = 0.1
    scaling_factor = 0.1

    # Add the residual blocks to the model
    for i in range(num_block):
        x = utils.resBlock(x, num_channels, scale=scaling_factor)

    # One more convolution, and then we add the output of our first conv layer
    x = slim.conv2d(x, num_channels, [3, 3])
    # x += conv_1

    # Upsample output of the convolution
    # x = utils.upsample(x, self.scale, 256, None)

    # TODO:试试新的上采样
    x = slim.conv2d(x, self.output_channels * self.scale * self.scale, [3, 3], activation_fn=tf.nn.tanh)
    x = tf.depth_to_space(x, self.scale)

    # One final convolution on the upsampling output
    output = x + sk  # slim.conv2d(x,output_channels,[3,3])
    return output


def MYSR_v4(self, image_input, image_input_bicubic):
    x = image_input

    # TODO: model v4.3
    scaling_factor = 0.1
    x = utils.denseBlock(x, 32, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 64, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 64, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 128, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 128, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 128, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 256, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 256, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 256, scale=scaling_factor, n_block=2)
    x = utils.denseBlock(x, 256, scale=scaling_factor, n_block=2)

    # TODO: shuffle型上采样
    x = slim.conv2d(x, self.output_channels * self.scale * self.scale, [3, 3], activation_fn=tf.nn.tanh)
    x = tf.depth_to_space(x, self.scale)

    # bicubic图像连接
    x += image_input_bicubic

    # One final convolution on the upsampling output
    output = x
    return output


def MYSR_v5(self, image_input, num_channels, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels * 4,
          3,
          padding='same',
          name='conv0',
        )
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return x + skip

    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x += sk

    return x


# 去掉整体跳跃层的卷积结构，改为bicubic
def MYSR_v5_b(self, image_input, image_input_bicubic, num_channels, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels * 4,
          3,
          padding='same',
          name='conv0',
        )
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return x + skip

    # skip connect
    # x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    # x = tf.depth_to_space(x, self.scale)
    sk = image_input_bicubic

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x += sk

    return x


def MYSR_v5_1(self, image_input, image_input_bicubic, num_channels, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv0',
        )
        x = tf.concat([x, skip], 3)     # 扩大n_fea
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return x + skip

    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x += sk

    return x


# 去掉Relu前增加n_fea结构
def MYSR_v5_0(self, image_input, image_input_bicubic, num_channels, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv0',
        )
        # x = tf.concat([x, skip], 3)     # 扩大n_fea
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return x + skip

    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x += sk

    return x


# 去掉整体跳跃层的卷积结构，改为bicubic
def MYSR_v5_0_b(self, image_input, image_input_bicubic, num_channels, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv0',
        )
        # x = tf.concat([x, skip], 3)     # 扩大n_fea
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return x + skip

    # skip connect
    sk = image_input_bicubic

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x += sk

    return x


# Relu前增加n_fea为3倍，输入为x-2, x-1, x
def MYSR_v5_3(self, image_input, image_input_bicubic, num_channels, num_block):
    # TODO: 增加多层扩大结构
    # 定制residual_block
    # input x
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv0',
        )
        x = tf.concat([x, skip], 3)     # 扩大n_fea
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return x + skip

    # 定制residual_block
    # input x x-1
    def _residual_block1(x, x1, num_channels):
        skip = x
        skip1 = x1
        x = Conv2DWN.conv2d_weight_norm(
            x,
            num_channels,
            3,
            padding='same',
            name='conv0',
        )

        x = tf.concat([x, skip], 3)  # 扩大n_fea x2
        x = tf.concat([x, skip1], 3)  # 扩大n_fea x3

        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
            x,
            num_channels,
            3,
            padding='same',
            name='conv1',
        )
        return x + skip

    # 定制residual_block
    # input x x-1 x-2
    def _residual_block2(x, x1, x2, num_channels):
        skip = x
        skip1 = x1
        skip2 = x2
        x = Conv2DWN.conv2d_weight_norm(
            x,
            num_channels,
            3,
            padding='same',
            name='conv0',
        )

        x = tf.concat([x, skip], 3)  # 扩大n_fea x2
        x = tf.concat([x, skip1], 3)  # 扩大n_fea x3
        x = tf.concat([x, skip2], 3)  # 扩大n_fea x4

        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
            x,
            num_channels,
            3,
            padding='same',
            name='conv1',
        )
        return x + skip

    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # x2
    with tf.variable_scope('layer0'):
        x = x2 = _residual_block(x, num_channels)

    # x3
    with tf.variable_scope('layer1'):
        x = x1 = _residual_block1(x, x2, num_channels)

    # layer x4
    for i in range(2, num_block):
        with tf.variable_scope('layer{}'.format(i)):
            tmp = _residual_block2(x, x1, x2, num_channels)

        # x->x1 x1->x2
        x2 = x1
        x1 = x
        x = tmp

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x += sk

    return x


# 将模型划分为特征提取与特征整合两部分
def MYSR_v6(self, image_input, num_channels, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv0',
        )
        # x = tf.concat([x, skip], 3)     # 扩大n_fea
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return x + skip

    # 入口层
    x = sk = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')  # 入口

    # 特征提取
    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)
            sk = tf.concat([sk, x], 3)  # 收集各层特征提取结果-特征整合

    # SR
    x = Conv2DWN.conv2d_weight_norm(sk, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)

    return x


# 移除sk及sk中的NxN放大
def MYSR_v5_Dense1(self, image_input, num_channels, num_channels_scale, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels*num_channels_scale,
          1,
          padding='same',
          name='conv0',
        )
        # x = tf.concat([x, skip], 3)     # 扩大n_fea
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return tf.concat([skip, x], 3)

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x = x

    return x


# 保留sk及sk中的NxN放大
def MYSR_v5_Dense2(self, image_input, num_channels, num_channels_scale, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels*num_channels_scale,
          1,
          padding='same',
          name='conv0',
        )
        # x = tf.concat([x, skip], 3)     # 扩大n_fea
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return tf.concat([skip, x], 3)

    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x += sk

    return x


# 保留sk及sk中的5x5放大
# 额外输出(output, x, sk)
def MYSR_v5_Dense2_test(self, image_input, num_channels, num_channels_scale, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels*num_channels_scale,
          1,
          padding='same',
          name='conv0',
        )
        # x = tf.concat([x, skip], 3)     # 扩大n_fea
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return tf.concat([skip, x], 3)

    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    output = sk + x

    return output, x, sk


def WDSR_v1(self, image_input, num_channels, num_block):
    # 定制residual_block
    def _residual_block(x, num_channels):
        skip = x
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels * 4,
          3,
          padding='same',
          name='conv0',
        )
        x = tf.nn.relu(x)
        x = Conv2DWN.conv2d_weight_norm(
          x,
          num_channels,
          3,
          padding='same',
          name='conv1',
        )
        return x + skip

    # skip connect
    x = Conv2DWN.conv2d_weight_norm(image_input, self.output_channels * self.scale * self.scale, 5, padding='same')
    x = tf.depth_to_space(x, self.scale)
    sk = x

    # input
    x = Conv2DWN.conv2d_weight_norm(image_input, num_channels, 3, padding='same')   #入口

    # layer
    for i in range(num_block):
        with tf.variable_scope('layer{}'.format(i)):
            x = _residual_block(x, num_channels)

    # SR
    x = Conv2DWN.conv2d_weight_norm(x, self.output_channels * self.scale * self.scale, 3, padding='same')
    x = tf.depth_to_space(x, self.scale)

    # output
    x += sk

    return x


# SRDenseNet X2
def SRDenseNetX2(self, image_input, num_channels, num_block):

    def _dense_block(x, num_channels, n_conv):
        x = slim.conv2d(x, num_channels, [3, 3])
        skip = tf.nn.relu(x)

        for i in range(1, n_conv):
            x = slim.conv2d(skip, num_channels, [3, 3])
            x = tf.nn.relu(x)
            skip = tf.concat([skip, x], 3)

        return skip

    # 入口
    x = slim.conv2d(image_input, num_channels*8, [3, 3])

    skip = x
    for i in range(0, num_block):
        with tf.variable_scope('DenseBlock{}'.format(i)):
            x = _dense_block(skip, num_channels, 8)
            skip = tf.concat([skip, x], 3)

    x = skip

    # 瓶颈
    x = slim.conv2d(x, 256, [1, 1])

    # upsample
    x = slim.conv2d_transpose(x, 256, [2, 2], 2)

    # reconstruction
    x = slim.conv2d(x, self.output_channels, [3, 3])

    output = x

    return output


# SRDenseNet X4
def SRDenseNetX4(self, image_input, num_channels, num_block):

    def _dense_block(x, num_channels, n_conv):
        skip = slim.conv2d(x, num_channels, [3, 3])

        for i in range(1, n_conv):
            x = slim.conv2d(skip, num_channels, [3, 3])
            skip = tf.concat([skip, x], 3)

        return skip

    # 入口
    x = slim.conv2d(image_input, num_channels, [3, 3])

    # DenseBlock
    skip = _dense_block(x, num_channels, 8)

    for i in range(1, num_block):
        x = _dense_block(skip, num_channels, 8)
        skip = tf.concat([skip, x], 3)

    # 瓶颈
    x = slim.conv2d(skip, 256, [1, 1])

    # upsample
    x = slim.conv2d_transpose(x, 256, [2, 2], 2)
    x = slim.conv2d_transpose(x, 256, [2, 2], 2)

    # reconstruction
    x = slim.conv2d(x, self.output_channels, [3, 3])

    output = x

    return output

