import warnings
import os
import matplotlib.pyplot as plt
import logging
import logging.config
import sys
# try:
#     import gmpy as gmp
# except ImportError:
#     import gmpy2 as gmp
import numpy as np
from PIL import Image
from os.path import dirname, abspath
from imageio import imsave

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


def save_image(image, filename, ext='.jpg'):
    """
    Save a numpy array to image.
    :param image: The numpy array
    :param filename: 
    :param ext: 
    :return: 
    """
    try:
        im = Image.fromarray(image)
    except TypeError:
        im = Image.fromarray(image, 'RGB')
    im = np.array(im)
    imsave(filename + ext, im)


def plot_generated(img, save=False, img_path=None, ext='.jpg', keep_previous=False, show_img=False):
    """
    Plot the generated images for the GAN.
    :param img: 
    :param save: 
    :param img_path: 
    :param ext: 
    :param keep_previous: 
    :param show_img: 
    :return: 
    """
    if save and img_path is None:
        warnings.warn('Cannot save image as image path not provided')
        logging.warning('Output images will not be saved. Path not provided by user')
    else:
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
    channels = img.shape[-1]
    numfiles = len([name for name in os.listdir(img_path) if os.path.isfile(img_path + name)])
    if len(img.shape) == 3:
        plt.figure()
        plt.imshow(img)
        plt.tight_layout()
        if save and img_path:
            if not keep_previous:
                img_name = img_path
                plt.savefig(img_name + ext, bbox_inches='tight')
            else:
                img_name = img_path + 'gen_plt_' + str(numfiles)
                plt.savefig(img_name + ext, bbox_inches='tight')
        else:
            if show_img:
                plt.show()
        plt.close('all')
    else:
        for i in range(img.shape[0]):
            plt.figure()
            if channels == 1:
                plt.imshow(img[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(img[i, :, :, :])
            plt.tight_layout()
            if save and img_path:
                if not keep_previous:
                    img_name = img_path + 'gen_plt_' + str(i)
                    plt.savefig(img_name + ext, bbox_inches='tight')
                else:
                    img_name = img_path + 'gen_plt_' + str(numfiles + i)
                    plt.savefig(img_name + ext, bbox_inches='tight')
            else:
                if show_img:
                    plt.show()
            plt.close('all')


def plot_data(data, figsize=(16, 8), img_path=None):
    if type(data) is dict:
        plt.figure(figsize=figsize)
        for k, v in data.iteritems():  # for python2
            plt.figure(figsize=figsize)
            plt.grid(b=True, which='both', axis='both')
            plt.plot(v, label=k)
            plt.legend()
            plt.savefig(img_path + k + '.png', bbox_inches='tight')
    if type(data) is list or type(data) is tuple:
        plt.plot(data, label='data')
        plt.legend()


# def visualize_filters(self, ip, img=0):
#     filters = int(ip.shape[-1])
#     row, col = self.factors(filters)
#     plt.subplots(nrows=row, ncols=col)
#     for i in range(filters):
#         plt.subplot(row, col, i + 1)
#         plt.imshow(ip[img, :, :, i], cmap='hot', interpolation='nearest')
#     plt.show()
#
#
# def factors(n):
#     if gmp.is_square(n):
#         return (int(np.sqrt(n)), int(np.sqrt(n)))
#     fac = set(reduce(list.__add__, ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))
#     lim = np.sqrt(n)
#     fac_1 = []
#     fac_2 = []
#     for f in fac:
#         if f < lim:
#             fac_1.append(f)
#         else:
#             fac_2.append(f)
#     return (max(fac_1), min(fac_2))
