import tensorflow.contrib.slim as slim
import utils
import tensorflow as tf


def EDSR_v1(self, image_input):
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
    return output


def MYSR_v4(self, image_input, image_input_bicubic):
    x = image_input

    # TODO: model v4.3
    scaling_factor = 0.3
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





