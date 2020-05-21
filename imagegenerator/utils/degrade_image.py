import glob
from PIL import Image


for imgs_path in glob.glob('/media/soumya/HDD_2/TIP2020/Original_4x/*/*_speckle_jpeg/*.jpg'):
    im = Image.open(imgs_path)
    size = [int(i/4) for i in im.size]
    im = im.resize(size, resample=Image.BICUBIC)
    im.save(imgs_path)
