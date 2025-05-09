# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

#IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)  # RGB 三个通道的均值
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)  # RGB 三个通道的标准差
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from PIL import Image     #用于图像处理，支持打开，修改，保存图像文件
import torch              #pytorch 核心库，用于张量操作、自动微分等
from torch.utils.data import Dataset   #PyTorch数据加载模块，用于定义自定义数据集类
import os
import sys                #提供与Python解释器交互的功能，如获取运行环境参数
import json               #用于处理JSON数据，支持解析和生成JSON格式文件
import pickle             #用于序列化和反序列化Python对象，方便保存和加载数据
import random

from tqdm import tqdm     #进度条库

import matplotlib.pyplot as plt

from functools import partial         #用于创建部分函数，将部分参数固定，生成新的函数
from collections import OrderedDict   #一个数据结构，用于存储有序的字典（保持插入顺序）

import torch.nn as nn                 #pytorch 的神经网络模块，包含各种神经网络层和损失函数
# from fvcore.nn import FlopCountAnalysis     #计算神经网络的FLOPs(浮点运算次数)和参数数量，用于分析模型的复杂度

import math
import torch.optim as optim            #PyTorch的优化器模块
import torch.optim.lr_scheduler as lr_scheduler   #用于调整学习率的调度器模块
from torch.utils.tensorboard import SummaryWriter     #用于将训练日志输出到Tensorboard以便可视化
from torchvision import transforms       #PyTorch的计算机视觉工具集，提供图像变换、预处理等功能。

# 根据传入的参数选择并构建不同的数据集
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'LIMUC':
        #数据集的划分
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data_no_CV(args.data_path)
        # 创建类别名称到索引的映射
        class_to_index = create_class_to_index_mapping(train_images_label + val_images_label)
        if is_train:
            # 实例化训练数据集
            dataset = MyDataSet(images_path=train_images_path,
                                   images_class=train_images_label,
                                   transform=transform,
                                   class_to_index=class_to_index)
        else:
            dataset = MyDataSet(images_path=val_images_path,
                                    images_class=val_images_label,
                                    transform=transform,
                                    class_to_index=class_to_index)
        nb_classes = 4

    return dataset, nb_classes


# 根据训练或验证模式以及其他配置构建图像预处理和增强的变换
def build_transform(is_train, args):
    resize_im = args.input_size > 32  # 判断是否需要缩放
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,   #输入图像大小
            is_training=True,
            color_jitter=args.color_jitter,  # 图像颜色抖动强度 0.3
            auto_augment=args.aa,            # 自动数据增强策略 rand-m9-mstd0.5-inc1
            interpolation=args.train_interpolation,    # 图像插值方法 bicubic
             
            # 随机擦除的相关参数
            re_prob=args.reprob,       # 0.25     
            re_mode=args.remode,       # pixel
            re_count=args.recount,     # 1
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)  # eval_crop_ratio——评估时的裁剪比例 0.875
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images，插值方法3——双三次插值
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


#======================================================================================================
def create_class_to_index_mapping(classes):
    """创建类别名称到索引的映射"""
    class_to_index = {cls: idx for idx, cls in enumerate(sorted(set(classes)))}
    return class_to_index

from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None, class_to_index=None):
        self.images_path = images_path      #图像文件路径的列表
        self.images_class = [class_to_index[cls] for cls in images_class]  # 将类别名称转换为索引，图像对应类别列表
        self.transform = transform          #图像的预处理或图像增强方法（eg. torchvision.transforms）
        #class_to_index: 类别名称到索引的映射字典

    def __len__(self):       #魔术方法：返回数据集大小（图像路径的数量）
        return len(self.images_path)

    def __getitem__(self, item):  #魔术方法：以item 为索引，获取数据集中的单个样本
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':     #图像需为RGB模式
            raise ValueError(f"图片 {self.images_path[item]} 不是 RGB 模式。")
        label = self.images_class[item]

        if self.transform:         #对图像进行预处理
            img = self.transform(img)

        return img, label

    @staticmethod   #静态方法，用于将多个样本合并成一个批次
    def collate_fn(batch):     #batch为一个列表，每个元素是__getitem__方法返回的（img,label）元组
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)     #将图像张量在第0维堆叠成一个批次
        labels = torch.tensor(labels, dtype=torch.long)  # 将标签列表转换为张量，并设置为长整型
        return images, labels
    
def read_split_data_no_CV(root: str, val_rate: float = 0.1):
    '''
    root: 数据集的根目录，每个类别对应一个子文件夹
    val_rate: 验证集的比例，默认为0.2
    '''
    random.seed(0)  # 保证随机结果可复现，固定随机种子，确保每次运行时的随机结果一致
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 列出文件夹下的所有文件，并获取拼接完整路径
    data_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    data_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(data_class))
    print(class_indices)

    #json.dumps():将Python对象（字典、列表）编码为JSON格式字符串）
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('../saves/save_json/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    supported = [".jpg", ".JPG", ".png", ".PNG",".bmp"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in data_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        # 排序，保证各平台顺序一致
        images.sort()

        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))

    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    #校验训练集和数据集是否有数据
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."


    #绘制每种类别样本数量的柱状图
    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(data_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(data_class)), data_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('data class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label
    

def read_split_data_CV(root: str, val_rate: float = 0.2):
    '''
    root: 数据集的根目录，每个类别对应一个子文件夹
    val_rate: 验证集的比例，默认为0.2
    '''
    random.seed(0)  # 保证随机结果可复现，固定随机种子，确保每次运行时的随机结果一致
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 列出文件夹下的所有文件，并获取拼接完整路径
    data_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    data_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(data_class))
    print(class_indices)

    #json.dumps():将Python对象（字典、列表）编码为JSON格式字符串）
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('../saves/save_json/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    #-------------------------------------------------
    every_class_num = []  # 存储每个类别的样本总数

    
    all_images_path = []   # 存储所有图片路径
    all_images_label = []  # 存储所有图片对应标签
    #-------------------------------------------------

    supported = [".jpg", ".JPG", ".png", ".PNG",".bmp"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in data_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        # 排序，保证各平台顺序一致
        images.sort()

        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        
        all_images_path.extend(images)
        all_images_label.extend([image_class] * len(images))

    print("{} images were found in the dataset.".format(sum(every_class_num)))

    #绘制每种类别样本数量的柱状图
    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(data_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(data_class)), data_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('data class distribution')
        plt.show()

#--------------------------------------------------------------------

#     return train_images_path, train_images_label, val_images_path, val_images_label
    return all_images_path, all_images_label,len(data_class)

