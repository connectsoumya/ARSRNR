from skimage.measure import compare_ssim
import numpy as np
import tensorflow as tf
import os

import sys
from os.path import dirname, abspath
import logging
import logging.config
from skimage.filters import prewitt
from imagegenerator.utils._haarpsi import haar_psi

if sys.version_info.major == 2:
    from ConfigParser import SafeConfigParser

    parser = SafeConfigParser()
else:
    from configparser import ConfigParser

    parser = ConfigParser()

my_path = os.path.abspath(os.path.dirname(__file__))
log_file_path = os.path.join(my_path, '../../logging.ini')
config_file_path = os.path.join(my_path, '../../config.ini')
parser.read(config_file_path)
logging.config.fileConfig(log_file_path)



def psnr(img_1, img_2):
    """
    Measure the Peak Signal to Noise Ratio between two images.
    :param img_1: Image 1
    :param img_2: Image 2
    # :type img_1: np.ndarray
    # :type img_2: np.ndarray
    :return: PSNR
    """
    mse = np.mean((img_1 - img_2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20.0 * np.log10(pixel_max / np.sqrt(mse))


def ssim(img_1, img_2, multichannel=True, **kwargs):
    """
    Measure the structured similarity index between two images. Refer to
    http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim
    :param img_1: The original image
    :param img_2: The image to be compared to the original image
    # :type img_1: np.ndarray
    # :type img_2: np.ndarray
    :param kwargs:
    :return: SSIM
    """
    mssim = 0
    if len(img_1.shape) == 3:
        return compare_ssim(img_1, img_2, data_range=img_2.max() - img_2.min(), multichannel=multichannel, **kwargs)
    else:
        for i in range(img_1.shape[0]):
            im1 = img_1[i, :, :, :]
            im2 = img_2[i, :, :, :]
            mssim += compare_ssim(im1, im2, data_range=im2.max() - im2.min(), multichannel=multichannel, **kwargs)
        return mssim / img_1.shape[0]


def exp_loss(x_pred, x_real):
    """

    :param x_pred:
    :param x_real:
    :return:
    """
    l2 = tf.reduce_mean(tf.exp(tf.abs(x_pred - x_real)))
    return l2


def l1_loss(x_pred, x_real):
    l1 = tf.reduce_mean(tf.abs(x_pred - x_real))
    return l1


def gmsd(x_pred, x_real, c=0.0026):
    """
    https://arxiv.org/pdf/1308.3052.pdf
    :param x_pred: 
    :param x_real: 
    :param c: 
    :return: 
    """
    gms = _gms(x_pred, x_real, c)
    gmsm = _gmsm(gms)
    return np.sqrt(np.mean((gms - gmsm) ** 2))


def _gms(x_pred, x_real, c):
    """

    :rtype: np.ndarray
    """

    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    grad_pred = np.zeros(shape=x_pred.shape[:3])
    grad_real = np.zeros(shape=x_real.shape[:3])
    if len(x_pred.shape) == 4 and x_pred.shape[-1] == 3:
        for i in range(x_pred.shape[0]):
            grad_pred[i, :, :] = prewitt(rgb2gray(x_pred[i, :, :, :]))
            grad_real[i, :, :] = prewitt(rgb2gray(x_real[i, :, :, :]))
    elif len(x_pred.shape) == 3 and x_pred.shape[-1] == 3:
        grad_pred = prewitt(rgb2gray(x_pred))
        grad_real = prewitt(rgb2gray(x_real))
    elif len(x_pred.shape) == 4 and x_pred.shape[-1] == 1:
        for i in range(x_pred.shape[0]):
            grad_pred[i, :, :] = prewitt(x_pred[i, :, :, 0])
            grad_real[i, :, :] = prewitt(x_real[i, :, :, 0])
    elif len(x_pred.shape) == 3 and x_pred.shape[-1] == 1:
        grad_pred = prewitt(x_pred)
        grad_real = prewitt(x_real)
    else:
        raise ValueError('Check the function in the source code. You might need to develop this condition.')

    gms = (2 * grad_real * grad_pred + c) / (grad_real ** 2 + grad_pred ** 2 + c)
    return gms


def haarpsi(x_pred, x_real, preprocess_with_subsampling=True):
    haar = []
    if len(x_pred.shape) == 4 and x_pred.shape[-1] == 3:
        for i in range(x_pred.shape[0]):
            haar.append(haar_psi(x_pred[i, :, :, :], x_real[i, :, :, :], preprocess_with_subsampling)[0])
    elif len(x_pred.shape) == 3 and x_pred.shape[-1] == 3:
        haar = haar_psi(x_pred, x_real, preprocess_with_subsampling)[0]
    elif len(x_pred.shape) == 4 and x_pred.shape[-1] == 1:
        for i in range(x_pred.shape[0]):
            haar.append(haar_psi(x_pred[i, :, :, 0], x_real[i, :, :, 0], preprocess_with_subsampling)[0])
    elif len(x_pred.shape) == 3 and x_pred.shape[-1] == 1:
        haar = haar_psi(x_pred[:, :, 0], x_real[:, :, 0], preprocess_with_subsampling[0])
    else:
        raise ValueError('Check the function in the source code. You might need to develop this condition.')
    return np.mean(haar)


def _gmsm(gms):
    return np.mean(gms)
