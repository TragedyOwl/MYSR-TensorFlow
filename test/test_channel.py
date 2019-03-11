import numpy as np


x = np.array([[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]])

print(x.shape)

y = np.expand_dims(x, axis=2)

print(y.shape)
print(y)

z = np.concatenate([y] * 3, 2)
print(z)






