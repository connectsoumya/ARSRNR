from facegenerator.utils.metrics import psnr
from facegenerator.utils.metrics import gmsd
from facegenerator.utils.metrics import haarpsi
from facegenerator.utils.metrics import ssim
from glob import glob
import numpy as np
import cv2
from PIL import Image


def _read_image(image, use='pil', shape=None):
    if use == 'pil':
        if shape is not None:
            return np.array(Image.open(image).resize((shape[1], shape[0]))) / 255.0
        else:
            return np.array(Image.open(image)) / 255.0
    elif use == 'cv2':
        if shape is not None:
            return cv2.resize(cv2.imread(image), shape[:2]) / 255.0
        else:
            return cv2.imread(image), shape[:2] / 255.0


def get_files_list(path_original, path_degraded, path_enhanced):
    enhanced_files = glob(path_enhanced + '/*')

    degraded_files = glob(path_degraded + '/*')

    original_files = glob(path_original + '/*')

    return sorted(original_files), sorted(degraded_files), sorted(enhanced_files)


noise = 'speckle_jpeg'
dataset = 'BSD100'
sr_algo = 'bicubic'
ar_algo = 'bilateral'


def print_results(path_original='/media/soumya/HDD_1/ARSR_Datasets/test_sets/%s' % dataset,
                  path_degraded='/media/soumya/HDD_1/ARSR_Datasets/test_sets/%s_degraded/%s_' % (
                          dataset, dataset) + noise,
                  path_enhanced='/media/soumya/HDD_2/TIP2020/results/%s+%s/%s/%s_' % (
                          sr_algo, ar_algo, dataset, dataset) + noise):
    # path_enhanced='/media/soumya/HDD_1/testing_subset/to_test/enh/LIVE1_gauss'):
    ori, de, en = get_files_list(path_original, path_degraded, path_enhanced)
    file_triplets = list(zip(ori, ori, en))
    psnr_scores_deg = []
    psnr_scores_enh = []
    ssim_scores_deg = []
    ssim_scores_enh = []
    gmsd_scores_deg = []
    gmsd_scores_enh = []
    haar_scores_deg = []
    haar_scores_enh = []
    for f in file_triplets:
        k = {}
        k[0] = _read_image(f[0])
        # k[1] = _read_image(f[1])
        k[2] = _read_image(f[2], shape=k[0].shape)
        k[0] = k[0][10:-10, 10:-10, :]
        # k[1] = k[1][10:-10, 10:-10, :]
        k[2] = k[2][10:-10, 10:-10, :]
        # psnr_scores_deg.append(psnr(k[0], k[1]))
        psnr_scores_enh.append(psnr(k[0], k[2]))
        # ssim_scores_deg.append(ssim(k[0], k[1]))
        ssim_scores_enh.append(ssim(k[0], k[2]))
        # gmsd_scores_deg.append(gmsd(k[0], k[1]))
        gmsd_scores_enh.append(gmsd(k[0], k[2]))
        # haar_scores_deg.append(haarpsi(k[0], k[1]))
        haar_scores_enh.append(haarpsi(k[0], k[2]))
    # print('PSNR: enhanced=%2.3f' % (np.mean(psnr_scores_enh)))
    # print('SSIM: enhanced=%1.3f' % (np.mean(ssim_scores_enh)))
    # print('GMSD: enhanced=%1.3f' % (np.mean(gmsd_scores_enh)))
    # print('HPSI: enhanced=%1.3f' % (np.mean(haar_scores_enh)))
    print('%2.3f & %1.3f & %1.3f & %1.3f' % (
    np.mean(psnr_scores_enh), np.mean(ssim_scores_enh), np.mean(gmsd_scores_enh), np.mean(haar_scores_enh)))


if __name__ == '__main__':
    noise = 'speckle_jpeg'
    dataset_ = ['LIVE1', 'Set14', 'BSD100']
    sr_algo_ = ['bicubic', 'SRGAN', 'SRCNN', 'Enet']
    ar_algo_ = ['bilateral', 'ARCNN', 'IRCNN', 'DIP']

    for ar_algo in ar_algo_:
        for sr_algo in sr_algo_:
            for dataset in dataset_:
                print(dataset, sr_algo, ar_algo)
                print_results(path_original='/media/soumya/HDD_1/ARSR_Datasets/test_sets/%s' % dataset,
                              path_degraded='/media/soumya/HDD_1/ARSR_Datasets/test_sets/%s_degraded/%s_' % (
                                  dataset, dataset) + noise,
                              path_enhanced='/media/soumya/HDD_2/TIP2020/results/Ours/%s/%s_' % (
                                  dataset, dataset) + noise)
                              # path_enhanced='/media/soumya/HDD_2/TIP2020/results/%s+%s/%s/%s_' % (
                              #     sr_algo, ar_algo, dataset, dataset) + noise)
