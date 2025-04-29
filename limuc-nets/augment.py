# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
import torch
from torchvision import transforms

from timm.data.transforms import str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor

import numpy as np
from torchvision import datasets, transforms
import random



from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
 
    
    
class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
    
    
def new_data_aug_generator(args = None):      # 生成图像数据增强操作的函数
    img_size = args.input_size                # 输入图像大小
    remove_random_resized_crop = args.src     # 决定是否移出RandomResizedCrop（随机裁剪）操作
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]    # ImageNet 数据集的均值和标准差，用于图像的归一化处理
    primary_tfl = []               # 用于存储主要的变换操作 Transform List
    scale=(0.8, 1.0)              # RandomResizedCrop 操作的缩放范围，指定了裁剪区域的相对大小范围
    interpolation='bicubic'        # 插值方法，指定图像重采样时使用的插值方法
    if remove_random_resized_crop:
        primary_tfl = [            # 主要变换操作
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [  
#             RandomResizedCropAndInterpolation(
#                 img_size, scale=scale, interpolation=interpolation),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip()
        ]

    # 辅助变换操作    
    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),      
                                              Solarization(p=1.0),
                                              GaussianBlur(p=1.0)])]
   
    if args.color_jitter is not None and not args.color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))
    
    # 最终变换操作
    final_tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)


#--------------------------------可视化数据增强结果--------------------------------------------------------
import os
import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt
from PIL import Image

# 数据增强管道
seq = iaa.Sequential([
#     iaa.Crop(px=(0, 25), keep_size=True),  # 随机裁剪，范围为0到25像素
    iaa.Fliplr(0.5),                       # 水平翻转，概率为50%
    iaa.Flipud(0.5), 
#     iaa.GaussianBlur(sigma=(0, 3.0))      # 高斯模糊，sigma值在0到3之间随机选择
])

def augment_and_save_images(image_paths, save_dir, seq=None):
    """
    对输入的图片路径列表进行数据增强，并保存增强后的图片到指定目录。
    
    参数：
        image_paths:原始图片路径列表
        save_dir:保存增强后图片的目录路径
        seq: 数据增强序列（imgaug增强管道)
    返回：
        augmented_paths: 保存增强后的图片的路径列表
    """
    os.makedirs(save_dir, exist_ok=True)        # 创建保存目录
    augmented_paths = []     # 用于保存增强后图片的路径列表
    
    for i, image_path in enumerate(image_paths):
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)    # 转换为NumPy数组
        
        # 应用数据增强
        augmented_image_np = seq(image=image_np)
        
        # 获取原始图片的文件名
        original_filename = os.path.basename(image_path)
        
        #构造增强后的图片的保存路径
        save_path = os.path.join(save_dir, original_filename)
        
        # 将增强后的图片保存到指定目录
        augmented_image = Image.fromarray(augmented_image_np.astype(np.uint8))
        augmented_image.save(save_path)
        
        # 记录保存路径
        augmented_paths.append(save_path)
        
    return augmented_paths

# 可视化增强结果
def visualize_augmented_images(image_paths, augmented_paths, cols=2, rows=0):
    """
    将原图与增强后的图片并列显示并保存。
    
    参数:
        image_paths: 原始图片路径列表
        augmented_paths: 增强后的图片路径列表
        cols: 每行显示的列数，默认为 2 (原图 + 增强图)。
        rows: 行数，默认根据图片数量自动计算。
    """
    
    if len(image_paths) != len(augmented_paths):
        raise ValueError("原图和增强后的图片数量不一致，请检查输入数据。")
        
    # 计算需要的行数
    num_images = 2 * len(image_paths)
    rows = rows or (num_images // cols + (num_images % cols > 0))
    
    # 计算子图
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2)) #每张图片4×3英寸
    axes = axes.flatten()   # 展平子图以便于迭代
    
    # 显示原图和增强后的图片
    for i,(original_path,augmented_path) in enumerate(zip(image_paths, augmented_paths)):
        
        original = np.array(Image.open(original_path)).astype(np.float32)
        augmented = np.array(Image.open(augmented_path)).astype(np.float32)

        
        # 归一化到0-1范围
        original = (original - original.min())/(original.max() - original.min())
        augmented = (augmented - augmented.min())/(augmented.max() - augmented.min())
        
        # 显示原图
        axes[2*i].imshow(original)
        axes[2*i].set_title("original")
        axes[2*i].axis('off')
        
        # 显示增强后的图片
        axes[2*i + 1].imshow(augmented)
        axes[2*i + 1].set_title("augmented")
        axes[2*i + 1].axis('off')
        
    for j in range(num_images,len(axes)):
        axes[j].axis('off')
    
    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)  # 去掉图片之间的缝隙，wspace 和 hspace 分别控制水平和垂直间距

    save_dir="../LIMUC/saves/"
    os.makedirs(save_dir,exist_ok=True)

    save_path = os.path.join(save_dir,f"Images_comparison.png")
    plt.savefig(save_path,bbox_inches='tight',pad_inches=0,dpi=300)
        
    plt.close(fig)


def read_and_augment(image_paths, save_dir, seq=seq):
    """
    从指定文件夹加载图片，进行数据增强，并保存增强后的图片
    
    参数：
        image_folder: 原始图片文件夹路径
        save_dir: 保存增强后图片的目录路径
        seq：数据增强序列（imgaug）
    
    返回：
        augmented_paths: 保存增强后图片的路径列表
    """
    
#     # 加载图片路径
#     image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
#     # 如果没有图片，则退出
#     if not images:
#         print("No images found in the specified folder.")
#         exit()
    
    # 对图片进行增强并保存
    os.makedirs(save_dir,exist_ok=True)
    augmented_paths = augment_and_save_images(image_paths, save_dir, seq)
    
    # 可视化对比图
    visualize_augmented_images(image_paths[:12], augmented_paths[:12], cols=2, rows=0)
    
    return augmented_paths