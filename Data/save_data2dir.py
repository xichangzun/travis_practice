import os
import random
import shutil

import deepdish as dd


"""
mkdir data
mkdir data/test
mkdir data/train
mkdir data/validate
mkdir data/train/0
mkdir data/train/1
mkdir data/test/0
mkdir data/test/1
mkdir data/validate/0
mkdir data/validate/1

rm -rf data/validate
rm -rf data/train
rm -rf data/test
mkdir data/test
mkdir data/train
mkdir data/validate
mkdir data/train/0
mkdir data/train/1
mkdir data/test/0
mkdir data/test/1
mkdir data/validate/0
mkdir data/validate/1

cp patch/初赛训练集/negative/* data/train/0
cp patch/决赛训练集/negative/* data/train/0
cp patch/初赛训练集/positve/* data/train/1
cp patch/决赛训练集/positve/* data/train/1

cp patch/决赛测试集/negative/* data/validate/0
cp patch/决赛测试集/positve/* data/validate/1

cp patch/初赛测试集/初赛测试二/negative/* data/test/0/
cp patch/初赛测试集/初赛测试二/positive/* data/test/1
"""



#    : 'path'
# keys is ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
# train_dic = {}
# test_dic = {}
#
# # path for each image folder
# path = "/home/hdc/wwj_test/xiaoli/Stomachcancer/patch"
# path1 = "/home/hdc/wwj_test/xiaoli/Stomachcancer/patch1"
# ls = os.listdir(path)
# random.shuffle(ls)
#
# # i: take the 30% as test set
# test_loop = int(len(ls) * 0.3)
# for i in ls:
#     pic_path = path + "/" + i
#     pic_lst = os.listdir(pic_path)
#     if test_loop > 0:
#         for j in pic_lst:
#             # x.x.jpg
#             temp = j.split('_')[-1]
#             cancer_score = temp[:3]
#             if cancer_score == '-0.':
#                 cancer_score = float(0.0)
#             if float(cancer_score) in test_dic:
#                 test_dic[float(cancer_score)].append(pic_path + "/" + j)
#             else:
#                 test_dic[float(cancer_score)] = [pic_path + "/" + j]
#     else:
#         for j in pic_lst:
#             # x.x.jpg
#             temp = j.split('_')[-1]
#             cancer_score = temp[:3]
#             if cancer_score == '-0.':
#                 cancer_score = float('0.0')
#             if float(cancer_score) in train_dic:
#                 train_dic[float(cancer_score)].append(pic_path + "/" + j)
#             else:
#                 train_dic[float(cancer_score)] = [pic_path + "/" + j]
#     test_loop -= 1
# len(train_dic[0])
# len(train_dic[1])
#
# dd.io.save('test.h5', test_dic)
# dd.io.save('train.h5', train_dic)
#
# os.mkdir('train_data')
# os.mkdir('test_data')
# os.mkdir('train_data/1')
# os.mkdir('train_data/0')
# os.mkdir('test_data/1')
# os.mkdir('test_data/0')
# # move file to match the ImageData
# for i in test_dic:
#     for j in test_dic[i]:
#         if i < 0.1:
#             shutil.copy2(j, 'test_data/0')
#         else:
#             shutil.copy2(j, 'test_data/1')
#     for j in train_dic[i]:
#         if i < 0.1:
#             shutil.copy2(j, 'train_data/0')
#         else:
#             shutil.copy2(j, 'train_data/1')
#
#
# ls = os.listdir('train_data/1')
# len(ls)
# ls = os.listdir('train_data/0')
# len(ls)
# ls = os.listdir('test_data/0')
# len(ls)
# ls = os.listdir('test_data/1')
# len(ls)
