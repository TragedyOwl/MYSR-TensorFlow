from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## data
config.TRAIN.imgsize = 16  # 训练时基础输入图像大小
config.TRAIN.scale = 2  # 放大倍数
config.TRAIN.output_channels = 3    # 通道数

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9
config.TRAIN.beta2 = 0.999
config.TRAIN.epsilon = 1e-08

## initialize G
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 20000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
# config.TRAIN.hr_img_path = 'data/DIV2K/DIV2K_train_HR/'
# config.TRAIN.lr_img_path = 'data/DIV2K/DIV2K_train_LR_bicubic/X4/'
# config.TRAIN.hr_img_path = 'data/benchmark/Urban100/Urban100_train_HR'
# config.TRAIN.hr_img_path = 'data/DIV2K/DIV2K_train_HR'
config.TRAIN.hr_img_path = 'data/benchmark/291'

config.VALID = edict()
## test set location
# config.VALID.hr_img_path = 'data/DIV2K/DIV2K_valid_HR/'
# config.VALID.lr_img_path = 'data/DIV2K/DIV2K_valid_LR_bicubic/X4/'
config.VALID.hr_img_path = 'data/test/test_valid/'

config.TEST = edict()
## test set location
config.TEST.hr_img_path = 'data/test/test_hr/'
config.TEST.sr_img_path = 'data/test/test_sr/'

## save model
config.TRAIN.save_epoch = 1000
config.TRAIN.save_model_dir = 'save/'
config.TRAIN.save_tensorboard_train_epoch = 1
config.TRAIN.save_tensorboard_train_dir = 'save/tensorboard/train/'
config.TRAIN.save_tensorboard_valid_epoch = 10
config.TRAIN.save_tensorboard_valid_dir = 'save/tensorboard/valid/'
config.TRAIN.save_tensorboard_test_dir = 'save/tensorboard/test/'

## log path
config.VALID.log_file = './log/valid.log'
config.TEST.log_file = './log/test.log'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
