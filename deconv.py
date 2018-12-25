# deconv test

import tensorflow as tf

x = tf.constant(1.0, shape=[1, 1, 1, 4])

y = tf.depth_to_space(x, 2)

'''''
Wrong!!This is impossible
y5 = tf.nn.conv2d_transpose(x1,kernel,output_shape=[1,10,10,3],strides=[1,2,2,1],padding="SAME")
'''
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
x5_deconv = sess.run(y)
print(x5_deconv.shape)




