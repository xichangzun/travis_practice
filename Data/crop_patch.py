import os

import cv2
import numpy as np
from openslide import OpenSlide
from skimage.transform.integral import integral_image, integrate


# 获取patch的标注区域和概率
def crop_patch(
        file_path_tif,
        file_path_tis_mask,
        file_path_lab_mask,
        mask_level,
        patch_level,
        size_patch
):
    """

    :param file_path_tif: 一个tiff文件的路径名
    :param file_path_tis_mask: 组织掩码的目录，用于提取有效区域
    :param file_path_lab_mask:标注区域掩码的目录，用于提取标注区域占整个patch的面积的比例
    :param size_patch: 需要的patch的面积的大小
    :param mask_level: 用于制作掩码的tiff的层，除决赛测试集在第0层之外，其余所有都在第1层
    :param patch_level: 在tiff的哪一层提取的patch，目前在第0层
    :return: patch的numpy数组，标注区域占到patch的面积
    """
    cur_slide_name = file_path_tif.split('.tiff')[0]
    branch_path_name = cur_slide_name.split('/')[-3:-1]
    # print(branch_path_name)
    cur_slide_name = cur_slide_name.split('/')[-1]
    filename_tis_msk = cur_slide_name + '_tissue_mask_lv_' + str(mask_level) + '.jpg'
    cur_path_tis_msk = os.path.join(file_path_tis_mask, branch_path_name[0], branch_path_name[1], filename_tis_msk)
    file_name_lab_mask = cur_slide_name + '_mask_lv_' + str(mask_level) + '.jpg'
    cur_path_lab_mask = os.path.join(file_path_lab_mask, branch_path_name[0], branch_path_name[1], file_name_lab_mask)
    if not os.path.exists(cur_path_lab_mask):
        print(cur_path_lab_mask)
    lab_mask = cv2.imread(cur_path_lab_mask, 0)
    integral_image_lab = integral_image(lab_mask.T / 255)
    slide = OpenSlide(file_path_tif)
    slide_w_lv_1, slide_h_lv_1 = slide.level_dimensions[mask_level]
    downsample = slide.level_downsamples[mask_level]
    size_patch_lv_1 = int(size_patch / downsample)  # k层上裁剪映射patchsize大小
    tissue_mask = cv2.imread(cur_path_tis_msk, 0)
    integral_image_tissue = integral_image(tissue_mask.T / 255)
    _, contours_tissue, _ = cv2.findContours(tissue_mask,
                                             cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)
    # Bounding box vertex
    p_x_left = 0
    p_x_right = slide_w_lv_1
    p_y_top = 0
    p_y_bottom = slide_h_lv_1
    # Make candidates of patch coordinate (level : 1)
    candidate_x = \
        np.arange(round(p_x_left), round(p_x_right)).astype(int)
    candidate_y = \
        np.arange(round(p_y_top), round(p_y_bottom)).astype(int)
    # Pick coordinates randomly
    len_x = candidate_x.shape[0]
    len_y = candidate_y.shape[0]
    number_patches = 1
    random_index_x = np.random.choice(len_x, number_patches, replace=False)
    random_index_y = np.random.choice(len_y, number_patches, replace=True)
    patch_x = candidate_x[random_index_x[0]]
    patch_y = candidate_y[random_index_y[0]]

    # Check if out of range
    while (patch_x + size_patch_lv_1 > slide_w_lv_1) or \
            (patch_y + size_patch_lv_1 > slide_h_lv_1):
        random_index_x = np.random.choice(len_x, number_patches, replace=False)
        random_index_y = np.random.choice(len_y, number_patches, replace=True)
        patch_x = candidate_x[random_index_x[0]]
        patch_y = candidate_y[random_index_y[0]]
        if (patch_y + size_patch_lv_1 > slide_h_lv_1) or \
                (patch_x + size_patch_lv_1 > slide_w_lv_1):
            continue
            # Check ratio of tumor region
        tissue_integral = integrate(integral_image_tissue,
                                    (patch_x, patch_y),
                                    (patch_x + size_patch_lv_1 - 1,
                                     patch_y + size_patch_lv_1 - 1))
        tissue_ratio = tissue_integral / (size_patch_lv_1 ** 2)
        if tissue_ratio < 0.3:
            continue
    patch_x_lv_0 = int(round(patch_x * downsample))
    patch_y_lv_0 = int(round(patch_y * downsample))
    lab_integral = integrate(integral_image_lab, (patch_x, patch_y),
                             (patch_x + size_patch_lv_1 - 1,
                              patch_y + size_patch_lv_1 - 1)
                             )
    lab_ratio = lab_integral / (size_patch_lv_1 ** 2)
    lab_ratio = float('%.1f' % lab_ratio)
    patch = slide.read_region((patch_x_lv_0, patch_y_lv_0),
                              patch_level,
                              (size_patch, size_patch))
    patch = np.array(patch)
    patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)

    return patch, lab_ratio
