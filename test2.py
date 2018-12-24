import tensorflow as tf
import tensorlayer as tl
from config import config
from utils import *
from tensorlayer.prepro import *


# 裁剪函数crop测试
# 结论：这里的crop要求裁剪后的wh必须小于源wh
# 加载图片
hr_img_path = config.TRAIN.hr_img_path
imgs = load_dataset(hr_img_path)

print(imgs[3].shape)
result = crop(imgs[3], wrg=imgs[3].shape[0], hrg=imgs[3].shape[0], is_random=False)
print(result.shape)

