import sys
import os
from PIL import Image
import numpy as np
from random import randint
from os.path import dirname, abspath
import logging
import logging.config
import glob
import random

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
IMG_FILE = ['jpg', 'png']


class DataGenerator(object):
    def __init__(self, data_path, gt_path=None, ignore_list_path=None, **kwargs):
        """
        Generator to yield data in proper format. The tree structure of the image organization is in the documentation. 
        :param gt_path:
        :param data_path:
        :param ignore_list_path: 
        :param data_resolution: The resolution of the training/testing data to be yielded.
        :param gt_resolution: The resolution of the ground truth data to be yielded.
        :param front_pose_data_resolution: Training data and ground truth resolution for frontal pose generation data
                                            are the same, so this parameter holds that resolution.
        """
        allowed_kwargs = {'data_resolution',
                          'gt_resolution',
                          'front_pose_data_resolution'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed: ' + str(k))
        self.gt_path = gt_path
        self.data_path = data_path
        self.ignore_list_path = ignore_list_path
        self.data_resolution = (64, 64)
        self.gt_resolution = (256, 256)
        self.front_pose_data_resolution = (128, 128)
        self.__dict__.update(kwargs)

    def no_gt_folders(self):
        """
        Returns the list of folders where no suitable ground truth was found.
        :return: 
        """
        try:
            with open(self.ignore_list_path) as f:
                content = f.readlines()
            logging.debug('Read ignore list file successfully')
        except IOError:
            logging.error('%s does not exist' % self.ignore_list_path)
            raise
        except TypeError:
            logging.error('Ignore list file not supplied')
            raise
        content = [x.strip() for x in content]
        logging.debug('Whitespace characters removed from ignore list contents')
        f.close()
        return content

    def yield_data_batch(self, batch_size):
        """
        This method yields the training data and the labels for the frontal pose face generation dataset.
        :param batch_size: 
        :return: 
        """
        batch_counter = 0
        if self.ignore_list_path is not None:
            ignore_folders = self.no_gt_folders()
        else:
            ignore_folders = []
        res = self.front_pose_data_resolution
        data_batch = np.zeros(shape=(batch_size,) + res + (3,))
        ground_truth_batch = np.zeros(shape=(batch_size,) + res + (3,))
        label_batch = np.zeros(shape=(batch_size,), dtype=int)
        dirs = ['/%s' % i for i in os.listdir(self.data_path) if
                os.path.isdir('%s/%s' % (self.data_path, i)) and i not in ignore_folders]
        for i in range(200):
            for img_dir in dirs:
                img_path = self.data_path + img_dir + '/' + str(i) + '.jpg'
                label_path = self.gt_path + img_dir + '/gt.jpg'
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    label = int(img_dir[1:])
                    try:
                        ground_truth = Image.open(label_path)
                    except IOError:
                        logging.error('Ground truth not found in %s' % label_path)
                        raise
                    if batch_counter < batch_size:
                        data_batch[batch_counter, :, :, :] = image
                        ground_truth_batch[batch_counter, :, :, :] = ground_truth
                        label_batch[batch_counter,] = label
                        batch_counter += 1
                    else:
                        batch_counter = 0
                        yield data_batch / 255., ground_truth_batch / 255., label_batch

    def yield_predict_batch(self, batch_size):
        """
        
        :param batch_size: 
        :return: 
        """
        batch_counter = 0
        data_batch = np.zeros(shape=(batch_size,) + self.data_resolution + (3,))
        dirs = ['/%s' % i for i in os.listdir(self.data_path) if
                os.path.isdir('%s/%s' % (self.data_path, i))]
        for i in range(200):
            for img_dir in dirs:
                img_path = self.data_path + img_dir + '/' + str(i) + '.jpg'
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    image = image.resize(self.data_resolution, Image.ANTIALIAS)
                    if batch_counter < batch_size:
                        data_batch[batch_counter, :, :, :] = image
                        batch_counter += 1
                    else:
                        batch_counter = 0
                        yield data_batch / 255.

    def yield_predict_batch_ar(self, batch_size):
        """

        :param batch_size: 
        :return: 
        """
        for img_path_dir in glob.glob(self.data_path + '/*'):
            for img_path in glob.iglob(img_path_dir + '/*.jpg'):
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    data_batch = np.expand_dims(image, axis=0)
                    yield data_batch / 255., img_path.replace(self.data_path + '/', '')

    def yield_predict_batch_sr(self, batch_size):
        """

        :param batch_size: 
        :return: 
        """
        # todo : work on this part
        for img_path in glob.iglob(self.data_path + '/*.png'):
            if os.path.exists(img_path):
                image = Image.open(img_path)
                w, h = image.size
                image = image.resize((w / 4, h / 4), Image.BICUBIC)
                # # ------resize to bigger one and feed into network----------
                # image = image.resize((w, h), Image.BICUBIC)
                # # ----------------------------------------------------------
                if w > 24 and h > 24:
                    data_batch = np.zeros(shape=(batch_size,) + (h / 4, w / 4, 3))
                    if len(np.array(image).shape) == 2:
                        logging.info('BW Image ' + img_path)
                        data_batch[:, :, :, 0] = np.array(image)
                        data_batch[:, :, :, 1] = np.array(image)
                        data_batch[:, :, :, 2] = np.array(image)
                    else:
                        data_batch[:, :, :, :] = np.array(image)
                    yield data_batch / 255., img_path.replace(self.data_path + '/', '')

    def yield_predict_batch_sr_vgg(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        # todo : work on this part
        dx = dy = 224
        for img_path in glob.iglob(self.data_path + '/*.png'):
            if os.path.exists(img_path):
                image = Image.open(img_path)
                w, h = image.size
                if w > 256 and h > 256:
                    x = random.randint(0, w - dx - 1)
                    y = random.randint(0, h - dy - 1)
                    image = image.crop((x, y, x + dx, y + dy))
                    image = image.resize((dx / 4, dy / 4), Image.BICUBIC)
                    # ------resize to bigger one and feed into network----------
                    image = image.resize((dx, dy), Image.BICUBIC)
                    # ----------------------------------------------------------
                    if w > 224 and h > 224:
                        data_batch = np.zeros(shape=(batch_size,) + (dx, dy, 3))
                        if len(np.array(image).shape) == 2:
                            logging.info('BW Image ' + img_path)
                            data_batch[:, :, :, 0] = np.array(image)
                            data_batch[:, :, :, 1] = np.array(image)
                            data_batch[:, :, :, 2] = np.array(image)
                        else:
                            data_batch[:, :, :, :] = np.array(image)
                        yield data_batch / 255., img_path.replace(self.data_path + '/', '')

    def yield_sr_batch(self, batch_size):
        """
        A method specially built for yielding super-resolution data batches.
        :param batch_size: 
        :return: 
        """
        batch_counter = 0
        data_batch = np.zeros(shape=(batch_size,) + self.data_resolution + (3,))
        gt_batch = np.zeros(shape=(batch_size,) + self.gt_resolution + (3,))
        dirs = ['/%s' % i for i in os.listdir(self.data_path) if
                os.path.isdir('%s/%s' % (self.data_path, i))]
        for i in range(200):
            for img_dir in dirs:
                img_path = self.data_path + img_dir + '/' + str(i) + '.jpg'
                if os.path.exists(img_path):
                    image_hr = Image.open(img_path).resize(self.gt_resolution, Image.ANTIALIAS)
                    image_lr = Image.open(img_path).resize(self.data_resolution, Image.ANTIALIAS)
                    if batch_counter < batch_size:
                        data_batch[batch_counter, :, :, :] = image_lr
                        gt_batch[batch_counter, :, :, :] = image_hr
                        batch_counter += 1
                    else:
                        batch_counter = 0
                        yield data_batch / 255., gt_batch / 255.

    def yield_ar_batch(self, batch_size):
        """
        This method yields the training data and the labels for artifact removal.
        :param batch_size: 
        :return: 
        """
        batch_counter = 0
        if self.ignore_list_path is not None:
            ignore_folders = self.no_gt_folders()
        else:
            ignore_folders = []
        res = self.gt_resolution
        data_batch = np.zeros(shape=(batch_size,) + res + (3,))
        ground_truth_batch = np.zeros(shape=(batch_size,) + res + (3,))
        dirs = ['/%s' % i for i in os.listdir(self.data_path) if
                os.path.isdir('%s/%s' % (self.data_path, i)) and i not in ignore_folders]
        for i in range(200):
            for img_dir in dirs:
                img_path = self.data_path + img_dir + '/' + str(i) + '.jpg'
                label_path = self.gt_path + img_dir + '/' + str(i) + '.jpg'
                if os.path.exists(img_path):
                    # image = Image.open(img_path)
                    image = Image.open(img_path).resize(self.gt_resolution, Image.ANTIALIAS)
                    try:
                        ground_truth = Image.open(label_path).resize(self.gt_resolution, Image.ANTIALIAS)
                    except IOError:
                        logging.error('Ground truth not found in %s' % label_path)
                        raise
                    if batch_counter < batch_size:
                        data_batch[batch_counter, :, :, :] = image
                        ground_truth_batch[batch_counter, :, :, :] = ground_truth
                        batch_counter += 1
                    else:
                        batch_counter = 0
                        yield data_batch / 255., ground_truth_batch / 255.

    def yield_ar_batch_v1(self, batch_size):
        """
        A method specially built for yielding AR data batches for the Luma (Y of YCbCr) channel only.
        :param batch_size: 
        :return: 
        """
        batch_counter = 0
        data_batch = np.zeros(shape=(batch_size,) + self.data_resolution + (1,))
        gt_batch = np.zeros(shape=(batch_size,) + self.gt_resolution + (1,))
        dx = dy = 256  # resolution of each image
        for img_path in glob.iglob(self.data_path + '/*.JPEG'):
            gt_path = img_path.replace('_LR_10', '')  # hardcoded for our imagenet subset
            gt_path = gt_path.replace('_LR_20', '')  # hardcoded for our imagenet subset
            if os.path.exists(gt_path):
                img = Image.open(gt_path)
                # data augmentation by random cropping
                w, h = img.size
                if w > 256 and h > 256:
                    x = random.randint(0, w - dx - 1)
                    y = random.randint(0, h - dy - 1)
                    try:
                        img = img.crop((x, y, x + dx, y + dy))
                        # image_hr = np.array(img.convert('YCbCr'))[:, :, 0]  # Take only the Y channel
                        image_hr = img.convert('YCbCr').split()[0]  # Take only the Y channel
                        image_lr = Image.open(img_path).crop((x, y, x + dx, y + dy))
                        if batch_counter < batch_size:
                            data_batch[batch_counter, :, :, :] = np.reshape(image_lr, (256, 256, 1))
                            gt_batch[batch_counter, :, :, :] = np.reshape(image_hr, (256, 256, 1))
                            batch_counter += 1
                        else:
                            batch_counter = 0
                            yield data_batch / 255., gt_batch / 255.
                    except Exception as e:
                        print (str(e))

    def yield_sr_batch_v2(self, batch_size):
        """
        SR
        :param batch_size: 
        :return: 
        """
        res = self.gt_resolution[0]
        batch_counter = 0
        data_batch = np.zeros(shape=(batch_size,) + self.data_resolution + (3,))
        gt_batch = np.zeros(shape=(batch_size,) + self.gt_resolution + (3,))
        dx = dy = res  # resolution of each image
        for gt_path in glob.iglob(self.data_path + '/*.JPEG'):
            if os.path.exists(gt_path):
                img = Image.open(gt_path)
                # data augmentation by random cropping
                w, h = img.size
                if w > res and h > res:
                    x = random.randint(0, w - dx - 1)
                    y = random.randint(0, h - dy - 1)
                    try:
                        image_hr = img.crop((x, y, x + dx, y + dy))
                        image_lr = image_hr.resize(self.data_resolution, Image.BICUBIC)
                        # image_lr = image_lr.resize(self.gt_resolution, Image.BICUBIC)
                        if batch_counter < batch_size:
                            try:
                                data_batch[batch_counter, :, :, :] = np.array(image_lr)
                                gt_batch[batch_counter, :, :, :] = np.array(image_hr)
                                batch_counter += 1
                            except ValueError:
                                logging.info('BW Image ' + gt_path)
                                data_batch[batch_counter, :, :, 0] = np.array(image_lr)
                                data_batch[batch_counter, :, :, 1] = np.array(image_lr)
                                data_batch[batch_counter, :, :, 2] = np.array(image_lr)
                                gt_batch[batch_counter, :, :, 0] = np.array(image_hr)
                                gt_batch[batch_counter, :, :, 1] = np.array(image_hr)
                                gt_batch[batch_counter, :, :, 2] = np.array(image_hr)
                                batch_counter += 1
                        else:
                            batch_counter = 0
                            yield data_batch / 255., gt_batch / 255.
                    except IOError as err:
                        logging.error(str(err))
                        continue

    def yield_sr_batch_v2_vgg(self, batch_size):
        """
        SR
        :param batch_size:
        :return:
        """
        batch_counter = 0
        data_batch = np.zeros(shape=(batch_size,) + self.gt_resolution + (3,))
        gt_batch = np.zeros(shape=(batch_size,) + self.gt_resolution + (3,))
        dx = dy = 224  # resolution of each image
        for gt_path in glob.iglob(self.data_path + '/*.JPEG'):
            if os.path.exists(gt_path):
                img = Image.open(gt_path)
                # data augmentation by random cropping
                w, h = img.size
                if w > 224 and h > 224:
                    x = random.randint(0, w - dx - 1)
                    y = random.randint(0, h - dy - 1)
                    try:
                        image_hr = img.crop((x, y, x + dx, y + dy))
                        image_lr = image_hr.resize(self.data_resolution, Image.BICUBIC)
                        image_lr = image_lr.resize(self.gt_resolution, Image.BICUBIC)
                        if batch_counter < batch_size:
                            try:
                                data_batch[batch_counter, :, :, :] = np.array(image_lr)
                                gt_batch[batch_counter, :, :, :] = np.array(image_hr)
                                batch_counter += 1
                            except ValueError:
                                logging.info('BW Image ' + gt_path)
                                data_batch[batch_counter, :, :, 0] = np.array(image_lr)
                                data_batch[batch_counter, :, :, 1] = np.array(image_lr)
                                data_batch[batch_counter, :, :, 2] = np.array(image_lr)
                                gt_batch[batch_counter, :, :, 0] = np.array(image_hr)
                                gt_batch[batch_counter, :, :, 1] = np.array(image_hr)
                                gt_batch[batch_counter, :, :, 2] = np.array(image_hr)
                                batch_counter += 1
                        else:
                            batch_counter = 0
                            yield data_batch / 255., gt_batch / 255.
                    except IOError as err:
                        logging.error(str(err))
                        continue

    def yield_arsr_batch(self, batch_size):
        """
        This method yields the training data and the labels for artifact removal and super resolution combined.
        :param batch_size: 
        :return: 
        """
        batch_counter = 0
        data_batch = np.zeros(shape=(batch_size,) + self.data_resolution + (3,))
        gt_batch = np.zeros(shape=(batch_size,) + self.gt_resolution + (3,))
        dirs = ['/%s' % i for i in os.listdir(self.data_path) if
                os.path.isdir('%s/%s' % (self.data_path, i))]
        for i in range(200):
            for img_dir in dirs:
                img_path = self.data_path + img_dir + '/' + str(i) + '.jpg'
                gt_path = self.gt_path + img_dir + '/' + str(i) + '.jpg'
                if os.path.exists(img_path):
                    # image_hr = Image.open(gt_path)
                    if True:  # image_hr.size[0] == 256:
                        image_hr = Image.open(gt_path).resize(self.gt_resolution, Image.ANTIALIAS)
                        image_lr = Image.open(img_path).resize(self.data_resolution, Image.ANTIALIAS)
                        if batch_counter < batch_size:
                            data_batch[batch_counter, :, :, :] = image_lr
                            gt_batch[batch_counter, :, :, :] = image_hr
                            batch_counter += 1
                        else:
                            batch_counter = 0
                            yield data_batch / 255., gt_batch / 255.

    def yield_arsr_with_occlusion(self, batch_size):
        """
        This method yields the training data and the labels for artifact removal and super resolution combined with
        random rectangular occlusions in place.
        :param batch_size: 
        :return: 
        """
        batch_counter = 0
        data_batch = np.zeros(shape=(batch_size,) + self.data_resolution + (3,))
        gt_batch = np.zeros(shape=(batch_size,) + self.gt_resolution + (3,))
        dirs = ['/%s' % i for i in os.listdir(self.data_path) if
                os.path.isdir('%s/%s' % (self.data_path, i))]
        occlusion_patch = np.zeros(shape=(10, 30, 3))
        for i in range(200):
            for img_dir in dirs:
                img_path = self.data_path + img_dir + '/' + str(i) + '.jpg'
                gt_path = self.gt_path + img_dir + '/' + str(i) + '.jpg'
                if os.path.exists(img_path):
                    image_hr = Image.open(gt_path)
                    x, y = randint(0, 54), randint(0, 34)
                    if image_hr.size[0] == 256:
                        image_lr = np.array(Image.open(img_path).resize(self.data_resolution, Image.ANTIALIAS))
                        image_lr[x:x + 10, y:y + 30, :] = occlusion_patch
                        if batch_counter < batch_size:
                            data_batch[batch_counter, :, :, :] = image_lr
                            gt_batch[batch_counter, :, :, :] = image_hr
                            batch_counter += 1
                        else:
                            batch_counter = 0
                            yield data_batch / 255., gt_batch / 255.


if __name__ == '__main__':
    gt_path_ = parser.get('DataGenerator', 'face_gt')
    tr_path_ = parser.get('DataGenerator', 'face_tr')
    ignore_list_path_ = parser.get('DataGenerator', 'ignore')
    batch_size_ = int(parser.get('main', 'batch_size'))
    gen = DataGenerator(gt_path_, tr_path_, ignore_list_path_)
    im_batch, lbl_batch = gen.yield_data_batch(batch_size_)
