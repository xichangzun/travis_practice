import os
import time

import torch.utils.data as data
from PIL import Image

from Data.crop_patch import crop_patch


# import inception_file_pth as pth

# file_path_slide   tiff文件的地址
# #决赛测试集在第0层生成掩码，其他数据集在第1层生成掩码
# file_path_origin_jpg    tiff中切出的一层图像的文件夹路径（一般是对应mask的相同层级的图像）
# file_path_tis_msk   完整图片生成的mask的文件夹路径
# file_path_lab_msk   标注文件生成的mask的文件夹路径


class Camel(data.Dataset):

    def __init__(self, image_folder,
                 file_path_tis_msk,
                 file_path_lab_msk,
                 mask_level,
                 patch_level,
                 size_patch,
                 transform=None, ):

        super(Camel, self).__init__()
        self.image_folder = image_folder
        self.file_path_tis_msk = file_path_tis_msk
        self.file_path_lab_msk = file_path_lab_msk
        self.mask_level = mask_level
        self.patch_level = patch_level
        self.size_patch = size_patch
        self.transform = transform
        self.file_name = []

        for folder in self.image_folder:
            for file in os.listdir(folder):
                self.file_name.append(folder + file)

    def __getitem__(self, index):
        s = time.time()
        file = self.file_name[index]
        patch, ratio = self.get_patch_ratio(file)
        target = Image.fromarray(patch)
        if ratio == 0:
            label = 0
        else:
            label = 1

        if self.transform is not None:
            target = self.transform(target)
        # print(time.time() - s)
        return target, label

    def __len__(self):
        return len(self.file_name)

    def get_patch_ratio(self, file_name):
        target, ratio = crop_patch(file_name,
                                   self.file_path_tis_msk,
                                   self.file_path_lab_msk,
                                   self.mask_level,
                                   self.patch_level,
                                   self.size_patch)
        return target, ratio


def get_dataset(train_transform=None, val_transform=None):
    val_image_folder = [#'/data/other/stomach_cancer/tiff/初赛测试集/初赛测试二/',
                        '/data/other/stomach_cancer/tiff/决赛测试集/决赛测试/',
                        ]

    val_file_path_tis_msk = '/data/other/stomach_cancer/mask/tissue_mask/'
    val_file_path_lab_msk = '/data/other/stomach_cancer/mask/lab_mask/'

    train_image_folder = ['/data/other/stomach_cancer/tiff/初赛训练集/初赛训练/',
                          '/data/other/stomach_cancer/tiff/决赛训练集/决赛训练/',
                          ]

    train_file_path_tis_msk = '/data/other/stomach_cancer/mask/tissue_mask/'
    train_file_path_lab_msk = '/data/other/stomach_cancer/mask/lab_mask/'

    mask_level = 0
    patch_level = 0
    size_patch = 299

    train_dataset = Camel(train_image_folder,
                          train_file_path_tis_msk, train_file_path_lab_msk,
                          mask_level, patch_level, size_patch, train_transform)

    val_dataset = Camel(val_image_folder,
                        val_file_path_tis_msk, val_file_path_lab_msk,
                        mask_level, patch_level, size_patch, val_transform)
    return train_dataset, val_dataset
