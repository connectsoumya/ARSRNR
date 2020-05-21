import os
from tqdm import tqdm
from PIL import Image


def enumerate_dirname(path):
    """
    Enumerates the directory name inside a folder. 
    :param path: This is the root folder.
    :return: 
    """
    for (dirpath, dirname_list, _) in os.walk(path):
        for dirname in tqdm(dirname_list):
            if dirname[0] != '.':
                old_dir = dirpath + dirname
                new_dir = dirpath + '%.5d' % dirname_list.index(dirname)
                if os.path.exists(new_dir):
                    print ('Directory %s already exist' % new_dir)
                    continue
                else:
                    try:
                        os.rename(old_dir, new_dir)
                    except:
                        print('Couldn\'t rename %s' % old_dir)
                        break


def enumerate_filename(path):
    """
    Enumerates the file name inside a folder. 
    :param path: It is the root folder
    :return: 
    """
    for (dirpath_0, dirname_list, _) in os.walk(path):
        for dirname in tqdm(dirname_list):
            if dirname[0] != '.':
                dirpath_1 = dirpath_0 + '/' + dirname + '/'
                for (dirpath_2, __, filename_list) in os.walk(dirpath_1):
                    for filename in filename_list:
                        if filename[0] != '.':
                            old_file = dirpath_2 + filename
                            new_file = dirpath_2 + str(filename_list.index(filename)) + '.jpg'
                            if os.path.exists(new_file):
                                print ('File %s already exist' % new_file)
                                continue
                            else:
                                try:
                                    os.rename(old_file, new_file)
                                except:
                                    print('Couldn\'t rename %s' % old_file)
                                    continue


def compress_img(path, quality):
    """

    :param path: 
    :param quality: Quality of the compressed image in percentage. eg. 10 means the new image will have a quality of 
    10% compared to the original one. 
    :return: 
    """
    for (dirpath, dirname_list, _) in tqdm(os.walk(path)):
        for dirname in tqdm(dirname_list):
            if dirname[0] != '.':
                newpath = dirpath + dirname + '/'
                for (_dirpath, __, filename_list) in os.walk(newpath):
                    for filename in filename_list:
                        if filename[0] != '.':
                            filepath = _dirpath + filename
                            im1 = Image.open(filepath)
                            im1.save(filepath[:-3] + 'jpg', 'JPEG', quality=quality)


def compress_img_luma(path, quality):
    """

    :param path: 
    :param quality: Quality of the compressed image in percentage. eg. 10 means the new image will have a quality of 
    10% compared to the original one. 
    :return: 
    """
    for (dirpath, dirname_list, _) in os.walk(path):
        for dirname in tqdm(dirname_list):
            if dirname[0] != '.':
                newpath = dirpath + '/' + dirname + '/'
                for (_dirpath, __, filename_list) in os.walk(newpath):
                    for filename in tqdm(filename_list):
                        if filename[0] != '.':
                            filepath = _dirpath + filename
                            im1 = Image.open(filepath)
                            try:
                                im1 = im1.convert('YCbCr').split()[0]
                                im1.save(filepath, "JPEG", quality=quality)
                            except Exception as e:
                                print (str(e))
                                continue



if __name__ == '__main__':
    datapath = '/work/ILSVRC2013_DET_test_LR_10'
    compress_img_luma(datapath, 10)

