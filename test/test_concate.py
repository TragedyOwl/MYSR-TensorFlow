import tensorflow as tf
import numpy as np


t1 = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
t2 = [[[[7, 8, 9], [10, 11, 12], [4, 5, 6]]]]
print(np.shape(t1))
print(np.shape(t2))
# tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
print(tf.concat([t1, t2], 0).get_shape())  # [4, 3]
print(tf.concat([t1, t2], 3).get_shape())  # [2, 6]
