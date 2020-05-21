import tensorflow as tf


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def is_square(x):
    if x ** 0.5 - int(x ** 0.5) == 0.0:
        return True
    else:
        return False

#
# def best_gmsd(self, gt, img, ip):
#     best = []
#     for i in range(gt.shape[0]):
#         _gmsd = gmsd(gt[i], img[i])
#         if _gmsd < self.best_gmsd_:
#             self.best_gmsd_ = _gmsd
#             plot_generated(img[i], save=True, img_path=gen_img_save_path + '/gmsd_op', show_img=False)
#             plot_generated(gt[i], save=True, img_path=gen_img_save_path + '/gmsd_gt', show_img=False)
#             plot_generated(ip[i], save=True, img_path=gen_img_save_path + '/gmsd_ip', show_img=False)
#             best.append(_gmsd)
#     try:
#         print('best gmsd ' + str(max(best)))
#     except ValueError as err:
#         pass
#
#
# def best_ssim(self, gt, img, ip):
#     best = []
#     for i in range(gt.shape[0]):
#         _ssim = ssim(gt[i], img[i], multichannel=True)
#         if _ssim > self.best_ssim_:
#             self.best_ssim_ = _ssim
#             plot_generated(img[i], save=True, img_path=gen_img_save_path + '/ssim_op', show_img=False)
#             plot_generated(gt[i], save=True, img_path=gen_img_save_path + '/ssim_gt', show_img=False)
#             plot_generated(ip[i], save=True, img_path=gen_img_save_path + '/ssim_ip', show_img=False)
#             best.append(_ssim)
#     try:
#         print(max(best))
#     except ValueError as err:
#         pass
#
#
# def worst_psnr(self, gt, img, ip):
#     worst = []
#     for i in range(gt.shape[0]):
#         _psnr = psnr(gt[i], img[i])
#         if _psnr < self.worst_psnr_:
#             self.worst_psnr_ = _psnr
#             plot_generated(img[i], save=True, img_path=gen_img_save_path + '/worstpsnr_op', show_img=False)
#             plot_generated(gt[i], save=True, img_path=gen_img_save_path + '/worstpsnr_gt', show_img=False)
#             plot_generated(ip[i], save=True, img_path=gen_img_save_path + '/worstpsnr_ip', show_img=False)
#             worst.append(_psnr)
#     try:
#         print(min(worst))
#     except ValueError as err:
#         pass
#
#
# def worst_gmsd(self, gt, img, ip):
#     worst = []
#     for i in range(gt.shape[0]):
#         _gmsd = gmsd(gt[i], img[i])
#         if _gmsd > self.worst_gmsd_:
#             self.worst_gmsd_ = _gmsd
#             plot_generated(img[i], save=True, img_path=gen_img_save_path + '/worstgmsd_op', show_img=False)
#             plot_generated(gt[i], save=True, img_path=gen_img_save_path + '/worstgmsd_gt', show_img=False)
#             plot_generated(ip[i], save=True, img_path=gen_img_save_path + '/worstgmsd_ip', show_img=False)
#             worst.append(_gmsd)
#     try:
#         print('worst gmsd ' + str(min(worst)))
#     except ValueError as err:
#         pass
#
#
# def worst_ssim(self, gt, img, ip):
#     worst = []
#     for i in range(gt.shape[0]):
#         _ssim = ssim(gt[i], img[i], multichannel=True)
#         if _ssim < self.worst_ssim_:
#             self.worst_ssim_ = _ssim
#             plot_generated(img[i], save=True, img_path=gen_img_save_path + '/worstssim_op', show_img=False)
#             plot_generated(gt[i], save=True, img_path=gen_img_save_path + '/worstssim_gt', show_img=False)
#             plot_generated(ip[i], save=True, img_path=gen_img_save_path + '/worstssim_ip', show_img=False)
#             worst.append(_ssim)
#     try:
#         print(min(worst))
#     except ValueError as err:
#         pass
#
#
# def best_haarpsi(self, gt, img, ip):
#     best = []
#     for i in range(gt.shape[0]):
#         _haarpsi = haarpsi(rgb2gray(gt[i]), rgb2gray(img[i]))[0]
#         if _haarpsi > self.best_haarpsi_:
#             self.best_haarpsi_ = _haarpsi
#             plot_generated(img[i], save=True, img_path=gen_img_save_path + '/besthaarpsi_op', show_img=False)
#             plot_generated(gt[i], save=True, img_path=gen_img_save_path + '/besthaarpsi_gt', show_img=False)
#             plot_generated(ip[i], save=True, img_path=gen_img_save_path + '/besthaarpsi_ip', show_img=False)
#             best.append(_haarpsi)
#     try:
#         print('best haarpsi ' + str(max(best)))
#     except ValueError as err:
#         pass
#
#
# def worst_haarpsi(self, gt, img, ip):
#     worst = []
#     for i in range(gt.shape[0]):
#         _haarpsi = haarpsi(rgb2gray(gt[i]), rgb2gray(img[i]))[0]
#         if _haarpsi < self.worst_haarpsi_:
#             self.worst_haarpsi_ = _haarpsi
#             plot_generated(img[i], save=True, img_path=gen_img_save_path + '/worsthaarpsi_op', show_img=False)
#             plot_generated(gt[i], save=True, img_path=gen_img_save_path + '/worsthaarpsi_gt', show_img=False)
#             plot_generated(ip[i], save=True, img_path=gen_img_save_path + '/worsthaarpsi_ip', show_img=False)
#             worst.append(_haarpsi)
#     try:
#         print('worst haarpsi ' + str(min(worst)))
#     except ValueError as err:
#         pass
