import tensorflow as tf
from imagegenerator.utils._vgg19 import vgg19_simple_api
from imagegenerator.utils.canny import TF_Canny


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


def mse_loss(x_pred, x_real):
    mse = tf.reduce_mean(tf.abs(x_pred - x_real) ** 2)
    return mse


def rgb2gray(rgb):
    r = tf.slice(rgb, [0, 0, 0, 0], [tf.shape(rgb)[0], 128, 128, 1])
    g = tf.slice(rgb, [0, 0, 0, 1], [tf.shape(rgb)[0], 128, 128, 1])
    b = tf.slice(rgb, [0, 0, 0, 2], [tf.shape(rgb)[0], 128, 128, 1])
    gray = tf.multiply(0.2989, r) + tf.multiply(0.5870, g) + tf.multiply(0.1140, b)
    return gray


"""
tf.slice(t, [1, 0, 0], [1, 2, 3]) means:
tf.slice(t, [dim_0 start idx, dim_1 start idx, dim_2 start idx], 
[dim_0 num of dims to fetch, dim_1 num of dims to fetch, dim_2 num of dims to fetch])
"""


def canny_loss(x_pred, x_real, raw_edge=False):
    x_pred_tensor = rgb2gray(x_pred)
    x_real_tensor = rgb2gray(x_real)
    edges_tensor_pred = TF_Canny(x_pred_tensor, return_raw_edges=raw_edge)
    edges_tensor_real = TF_Canny(x_real_tensor, return_raw_edges=raw_edge)
    return l1_loss(edges_tensor_pred, edges_tensor_real)


def canny_l1_loss(x_pred, x_real, raw_edge=False, canny_ratio=0.4):
    canny = canny_loss(x_pred, x_real, raw_edge)
    l1 = l1_loss(x_pred, x_real)
    return canny_ratio * canny + (1 - canny_ratio) * l1


def vgg_loss(x_pred, x_real):
    net_vgg, vgg_target_emb = vgg19_simple_api((x_real + 1) / 2, reuse=tf.AUTO_REUSE)
    _, vgg_predict_emb = vgg19_simple_api((x_pred + 1) / 2, reuse=True)

    def vggloss_func(list1, list2):
        return tf.norm(list1[1].outputs - list2[1].outputs)

    vgg_loss = vggloss_func(vgg_predict_emb, vgg_target_emb)
    return net_vgg, vgg_loss


def canny_vgg_loss(x_pred, x_real, raw_edge=False, canny_ratio=0.4):
    canny = canny_loss(x_pred, x_real, raw_edge)
    net_vgg, vgg = vgg_loss(x_pred, x_real)
    return net_vgg, canny_ratio * canny + (1 - canny_ratio) * vgg
