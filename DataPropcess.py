import scipy.misc
import tensorlayer as tl
import cv2
import utils
import skimage


# Gray2RGB
def img_Gray2RGB(in_path, out_path):
    # read images
    hr_img_list = sorted(
        tl.files.load_file_list(path=in_path, regx='.*.png', printable=False))
    hr_imgs = tl.vis.read_images(hr_img_list, path=in_path, n_threads=32)

    for idx in range(len(hr_imgs)):
        if len(hr_imgs[idx].shape) == 2:
            # TODO: convert image format
            hr_imgs[idx] = cv2.cvtColor(hr_imgs[idx], cv2.COLOR_GRAY2RGB)

        # save images
        scipy.misc.imsave(out_path + hr_img_list[idx], hr_imgs[idx])


# RGB2YCbCr
def img_RGB2YCbCr(in_path, out_path):
    # read images
    hr_img_list = sorted(
        tl.files.load_file_list(path=in_path, regx='.*.png', printable=False))
    hr_imgs = tl.vis.read_images(hr_img_list, path=in_path, n_threads=32)

    # convert image format
    for idx in range(len(hr_imgs)):
        if len(hr_imgs[idx].shape) == 2:
            # TODO: convert image format
            hr_imgs[idx] = cv2.cvtColor(hr_imgs[idx], cv2.COLOR_GRAY2RGB)

        # save images
        scipy.misc.imsave(out_path + hr_img_list[idx], hr_imgs[idx])

    pass


def evaluate(hr_path, sr_path):
    # read images
    hr_img_list = sorted(
        tl.files.load_file_list(path=hr_path, regx='.*.*', printable=False))
    hr_imgs = tl.vis.read_images(hr_img_list, path=hr_path, n_threads=32)

    sr_img_list = sorted(
        tl.files.load_file_list(path=sr_path, regx='.*.*', printable=False))
    sr_imgs = tl.vis.read_images(sr_img_list, path=sr_path, n_threads=32)

    if len(hr_imgs) != len(sr_imgs):
        print('Error: len(hr_imgs) != len(sr_imgs)')
        return

    # RGB2YCbCr
    PSNR = 0.0
    for idx in range(len(hr_imgs)):
        # TODO: convert image format
        hr = hr_imgs[idx]
        sr = sr_imgs[idx]
        sr_shape = sr.shape
        hr = hr[:sr_shape[0], :sr_shape[1], 0]
        sr = sr[:, :, 0]
        PSNR += skimage.measure.compare_psnr(hr, sr)

    return PSNR / len(hr_imgs)


# Evaluate PSNR on the Y channel
# hr_path: RGB
# sr_path: RGB
def evaluate_Y(hr_path, sr_path):
    # read images
    hr_img_list = sorted(
        tl.files.load_file_list(path=hr_path, regx='.*.*', printable=False))
    hr_imgs = tl.vis.read_images(hr_img_list, path=hr_path, n_threads=32)

    sr_img_list = sorted(
        tl.files.load_file_list(path=sr_path, regx='.*.*', printable=False))
    sr_imgs = tl.vis.read_images(sr_img_list, path=sr_path, n_threads=32)

    if len(hr_imgs) != len(sr_imgs):
        print('Error: len(hr_imgs) != len(sr_imgs)')
        return

    # RGB2YCbCr
    PSNR = 0.0
    for idx in range(len(hr_imgs)):
        # TODO: convert image format
        hr = cv2.cvtColor(hr_imgs[idx], cv2.COLOR_RGB2YCrCb)
        sr = cv2.cvtColor(sr_imgs[idx], cv2.COLOR_RGB2YCrCb)
        sr_shape = sr.shape
        hr = hr[:sr_shape[0], :sr_shape[1], 0]
        sr = sr[:, :, 0]
        PSNR += skimage.measure.compare_psnr(hr, sr)

    return PSNR / len(hr_imgs)





# main test
# PSNR = evaluate_Y('./data/benchmark/Urban100/Urban100_train_HR', './data/test/test/sr')
# PSNR = evaluate('./data/benchmark/B100/B100_train_HR', './data/test/test/sr')
PSNR = evaluate('./data/DIV2K/DIV2K_valid_HR', './data/test/test/sr')

print(PSNR)
#
# img_Gray2RGB('./data/test/test/Set14/', './data/test/test/Set14_/')

