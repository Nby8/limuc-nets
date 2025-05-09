import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
import itertools
import torch
import time
from scipy.stats import ranksums
from torch import save

from math import sqrt
from statistics import mean, pstdev
import numpy as np


def get_batch_size_for_model(model_name=""):
    if model_name == "ResNet18":
        batch_size = 64
    elif model_name == "ResNet50":
        batch_size = 32
    elif model_name == "VGG16_bn":
        batch_size = 12
    elif model_name == "DenseNet121":
        batch_size = 16
    elif model_name == "Inception_v3":
        batch_size = 32
    elif model_name == "mobilenet_v3_large":
        batch_size = 32
    else:
        batch_size = 16

    return batch_size


def plot_confusion_matrix_and_save(cm,
                                   target_names,
                                   path,
                                   title='Confusion matrix',
                                   cmap=None,
                                   normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.1 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(path)
    plt.show()


def save_confusion_matrix(args, cm,
                          target_names,
                          path,
                          fold,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          is_remission = False
                         ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    
#     plt.rcParams["font.weight"] = "bold"
#     plt.rcParams["axes.labelweight"] = "bold"
#     plt.rcParams["axes.titleweight"] = "bold"
#     plt.rcParams['font.family'] = 'Times New Roman'

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
#     plt.title(title)
#     plt.colorbar()

    if is_remission:
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        
        dig_fontsize = 40
        other_fontsize = 40
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        
        dig_fontsize = 33
        other_fontsize = 38
        
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=other_fontsize-5)
        plt.yticks(tick_marks, target_names,fontsize=other_fontsize-5)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.1 if normalize else cm.max() / 1.3
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=dig_fontsize)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=dig_fontsize)  
    
    plt.ylabel('True label', fontsize=other_fontsize, labelpad=15)
    plt.xlabel('Predicted label', fontsize=other_fontsize, labelpad=15)
    
    plt.tick_params(axis='x', which='both', pad=15)  # x 轴方向增加 padding，使 xlabel 离刻度更远
    plt.tick_params(axis='y', which='both', pad=15)  # y 轴方向增加 padding
    
#     if fold < (2-1):
#         plt.title(f"Confusion Matrix for {args.model_name} {args.loss} fold {fold}", fontsize=40, pad=25)
#     else:
#         plt.title(f"Confusion Matrix for {args.model_name}", fontsize=40, pad=25)
    
    plt.tight_layout()
    if is_remission:
        plt.subplots_adjust(left=0.5, bottom=0.5)  # 增加底部边距
    else:
        plt.subplots_adjust(left=0.3, bottom=0.3)
    
    plt.savefig(path)
    # plt.show()
    plt.close()
    
# Function to generate a Grad-CAM heatmap
def gen_cam(img, mask,i, fold):
    
    import matplotlib.pyplot as plt
    
    '''
        image: 原始输入图像，归一化到[0.1] 范围的NumPy数组，形状为（H，W，3）
        mask:  Grad-CAM 生成的热力图掩码，二维数组（H,W), 值在[0,1], 表示每个像素的重要性
    '''
    img = img.cpu()
    img = unnormalize(img)
    img = img.squeeze(0)
    
    # Step 4: 手动提取通道并组合成 RGB 图像
    r_channel = img[0, :, :]  # 提取红色通道
    g_channel = img[1, :, :]  # 提取绿色通道
    b_channel = img[2, :, :]  # 提取蓝色通道

    # 合并为 RGB 图像
    rgb_image = np.stack([r_channel, g_channel, b_channel], axis=-1)  # 形状变为 (H, W, C) 
    
    rgb_image = np.array(rgb_image)           # 转换为 NumPy 数组
    img = rgb_image.astype(np.float32)  # 确保数据类型为 float32
    
    root_path_orin = f'../saves/save_image/orin/fold_{fold}/'
    path_orin = os.path.join(root_path_orin,f'orin_{i}.png')
    
    os.makedirs(root_path_orin, exist_ok=True)
    
    plt.imshow(img)  # 显示图像
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(path_orin)

#     print(type(img))  # 应该输出 <class 'numpy.ndarray'>
#     print(img.shape)  # 应该输出类似 (height, width, channels)
#     print(img.dtype)  # 应该输出 uint8 或 float32

    # 检查并归一化图像
    if np.max(img) > 1:
        img = img / 255  # 如果像素值超过 [0, 1]，进行归一化
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create a heatmap from the Grad-CAM mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)   
    
    heatmap = np.float32(heatmap) / 255
    cam = (1 - 0.7) * heatmap + 0.7 * img             # 原始图像与热力图叠加
    cam = cam / np.max(cam)  # Normalize the result           # 归一化叠加结果，去报像素值不超出范围
    return np.uint8(255 * cam)  # Convert to 8-bit image       #将图像值缩放到 [0,255]后转化为8位无符号整数格式


import torch

def unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    输入图像形状为 (C, H, W)，对其进行反标准化。
    """
    # 将均值和标准差扩展为 (C, 1, 1) 形状，以便广播
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    # 反标准化
    img = img * std + mean
    
    # 确保像素值在 [0, 1] 范围内
    img = torch.clamp(img, 0, 1)
    
    return img

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def plot_confusion_matrix_TR(cm,
                             target_names,
                             title='Confusion matrix',
                             cmap=None,
                             normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin\ndoğruluk={:0.4f}; hata={:0.4f}'.format(accuracy, misclass))
    plt.show()


def plot_confusion_matrix_2(cm,
                            target_names,
                            title='Confusion matrix',
                            cmap=None,
                            normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:4.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel("Predicted label")
    plt.show()


def plot_confusion_matrix_2_and_save(cm,
                                     target_names,
                                     path,
                                     title='Confusion matrix',
                                     save_dpi=600,
                                     cmap=None,
                                     normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 7), dpi=save_dpi)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:4.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel("Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))
    plt.show()
    plt.savefig(path, dpi=save_dpi)


def weighted_random_sampler(dataset):
    class_sample_counts = [0 for x in range(dataset.number_of_class)]
    train_targets = []

    for x in dataset:
        train_targets.append(x[1])
        class_sample_counts[x[1]] += 1

    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[train_targets]

    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights,
                                                     num_samples=len(samples_weights))
    return sampler

from timm import create_model

def initialize_model(args,model_name, pretrained, num_classes):
    import torch
    import torchvision.models as models

    model = None

    if model_name == "VGG16_bn":
        if pretrained:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16_bn()
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)

    elif model_name == "ResNet18":
        if pretrained:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_name == "ResNet50":
        if pretrained:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_name == "ResNet152":
        if pretrained:
            model = models.resnet152(pretrained=True)
        else:
            model = models.resnet152()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_name == "DenseNet121":
        if pretrained:
            model = models.densenet121(pretrained=True)
        else:
            model = models.densenet121()
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)

    elif model_name == "Inception_v3":
        if pretrained:
            model = models.inception_v3(pretrained=True, transform_input=False)
        else:
            model = models.inception_v3(transform_input=False)
        in_features = model.fc.in_features
        aux_in_features = model.AuxLogits.fc.in_features

        model.AuxLogits.fc = torch.nn.Linear(aux_in_features, num_classes)
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        if pretrained:
            model = models.mobilenet_v3_large(pretrained=True)
        else:
            model = models.mobilenet_v3_large()
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    
    elif model_name == "DeiT":
        # 加载预训练模型
        model = create_model("deit_base_distilled_patch16_224", pretrained=False, num_classes=4)

        # 调整分类头
        if pretrained:
            in_features = model.head.in_features  # 获取分类头的输入维度
            model.head = torch.nn.Linear(in_features, num_classes)  # 替换分类头
    elif model_name == "DeiT-384":
        model = create_model("deit_base_distilled_patch16_384",pretrained=True, num_classes=4)
        
        # 调整分类头
        if pretrained:
            in_features = model.head.in_features  # 获取分类头的输入维度
            model.head = torch.nn.Linear(in_features, num_classes)  # 替换分类头
    elif model_name == "Deit3_huge":
        model = create_model("deit3_huge_patch14_224",pretrained=True, num_classes=4)
        
        # 调整分类头
        if pretrained:
            in_features = model.head.in_features  # 获取分类头的输入维度
            model.head = torch.nn.Linear(in_features, num_classes)  # 替换分类头
    elif model_name == "coatnet_2":
        model = create_model("coatnet_2_rw_224",pretrained=True, num_classes=4)
        
        # 调整分类头
        if pretrained:
            in_features = model.head.in_features  # 获取分类头的输入维度
            model.head = torch.nn.Linear(in_features, num_classes)  # 替换分类头

    elif model_name == 'overlock_b':
        model = create_model("overlock_b",pretrained=False, num_classes=4)

        # 调整分类头
        if pretrained:
            in_features = model.head.in_features  # 获取分类头的输入维度
            model.head = torch.nn.Linear(in_features, num_classes)  # 替换分类头
    else:
        print("Invalid model name!")
        exit()

    return model

def initialize_corn_model(model_name, pretrained, num_classes):
    import torch
    import torchvision.models as models

    model = None

    if model_name == "VGG16_bn":
        if pretrained:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16_bn()
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "ResNet18":
        if pretrained:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "ResNet50":
        if pretrained:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "ResNet152":
        if pretrained:
            model = models.resnet152(pretrained=True)
        else:
            model = models.resnet152()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "DenseNet121":
        if pretrained:
            model = models.densenet121(pretrained=True)
        else:
            model = models.densenet121()
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "Inception_v3":
        if pretrained:
            model = models.inception_v3(pretrained=True, transform_input=False)
        else:
            model = models.inception_v3(transform_input=False)
        in_features = model.fc.in_features
        aux_in_features = model.AuxLogits.fc.in_features

        model.AuxLogits.fc = torch.nn.Linear(aux_in_features, num_classes-1)
        model.fc = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "mobilenet_v3_large":
        if pretrained:
            model = models.mobilenet_v3_large(pretrained=True)
        else:
            model = models.mobilenet_v3_large()
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes-1)

    else:
        print("Invalid model name!")
        exit()

    return model

def label_from_logits_corn(logits):
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels

def get_remission_test_results(model, data_loader, device):
    model.eval()

    y_true = []
    y_probs = []
    y_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            y_true.append(target.item())

            output = model(data)
            output_prob = output.sigmoid().item()

            y_probs.append(output_prob)
            prediction = round(output_prob)
            y_pred.append(prediction)

    return y_true, y_probs, y_pred

# 获取测试集结果
def get_test_results_classification(model, data_loader, device, calculate_remission=True,
                                    nonremission_scores=[2, 3]):
    model.eval()

    y_true = []
    y_probs = []
    y_pred = []

    r_true = []
    r_probs = []
    r_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_true.append(target.item())

            output = model(data)

            y_probs.append(output.softmax(1).tolist()[0])
            prediction = output.argmax(dim=1, keepdim=True)[0][0].item()
            y_pred.append(prediction)

            if calculate_remission:
                # Remission calculation
                if target.item() in nonremission_scores:
                    r_true.append(1)
                else:
                    r_true.append(0)

                if prediction in nonremission_scores:
                    r_pred.append(1)
                else:
                    r_pred.append(0)

                r_probs.append(output.softmax(1).squeeze(0)[nonremission_scores].sum().item())

    if calculate_remission:
        return y_true, y_probs, y_pred, r_true, r_probs, r_pred
    else:
        return y_true, y_probs, y_pred

def get_test_results_classification_for_corn_loss_model(model, data_loader, device, calculate_remission=True,
                                    nonremission_scores=[2, 3]):
    model.eval()

    y_true = []
    y_probs = []
    y_pred = []

    r_true = []
    r_probs = []
    r_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_true.append(target.item())

            output = model(data)

            # TODO fix here, I may need probabilities later
            # probas = torch.sigmoid(output)
            # probas = torch.cumprod(probas, dim=1)
            # y_probs.append(probas.tolist()[0])

            prediction = label_from_logits_corn(output)
            y_pred.append(prediction.item())

            if calculate_remission:
                # Remission calculation
                if target.item() in nonremission_scores:
                    r_true.append(1)
                else:
                    r_true.append(0)

                if prediction in nonremission_scores:
                    r_pred.append(1)
                else:
                    r_pred.append(0)

                # TODO fix here, I may need probabilities later
                # r_probs.append(probas.squeeze(0)[nonremission_scores].sum().item())

    if calculate_remission:
        return y_true, y_probs, y_pred, r_true, r_probs, r_pred
    else:
        return y_true, y_probs, y_pred

def get_test_results_regression(model, data_loader, device, boundaries, nonremission_scores=[2, 3]):
    model.eval()

    y_true = []
    y_pred = []
    r_true = []
    r_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_true.append(target.item())

            output = model(data)
            output.squeeze_(1)
            prediction = get_regression_accuracy_with_boundaries(output, target, boundaries)
            y_pred.append(int(prediction.item()))

            # Remission calculation
            if target.item() in nonremission_scores:
                r_true.append(1)
            else:
                r_true.append(0)

            if prediction in nonremission_scores:
                r_pred.append(1)
            else:
                r_pred.append(0)

    return y_true, y_pred, r_true, r_pred


def get_regression_accuracy_with_boundaries(output, target, boundaries):
    output_classified = torch.zeros_like(output)
    for output_index in range(len(output)):
        for i in range(len(boundaries)):
            if i == 0:
                if output[output_index] < boundaries[i]:
                    output_classified[output_index] = 0
                    break
            elif i == len(boundaries) - 1:
                if boundaries[i] < output[output_index]:
                    output_classified[output_index] = i + 1
                else:
                    output_classified[output_index] = i
                break
            elif boundaries[i - 1] < output[output_index] and output[output_index] < boundaries[i]:
                output_classified[output_index] = i
                break

    return output_classified


def mixup_data(x, y, device, alpha=1.0, ):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_averaged_featuremap(channels: np.ndarray, shape: tuple):
    featuremap = channels.sum(0)
    featuremap = featuremap - np.min(featuremap)
    featuremap = featuremap / np.max(featuremap)
    featuremap_image = np.uint8(255 * featuremap)
    featuremap_image = cv2.resize(featuremap_image, (shape[1], shape[0]))

    return featuremap_image


def get_CAM(channels: np.ndarray, weights: np.ndarray, shape: tuple, id: int):
    class_weights = weights[id, :]
    class_weights = class_weights[:, np.newaxis, np.newaxis]
    CAM = channels * class_weights
    CAM = CAM.sum(0)
    CAM = CAM - np.min(CAM)
    CAM_img = CAM / (1e-7 + np.max(CAM))
    CAM_img = np.float32(cv2.resize(CAM_img, (shape[1], shape[0])))
    output_CAM = np.uint8(255 * CAM_img)

    return output_CAM


def get_CAM_with_bias(channels: np.ndarray, last_fc: torch.nn.Linear, shape: tuple, id: int):
    weights = last_fc.weight.data.cpu().numpy()
    bias = last_fc.bias.data.cpu().numpy()

    class_weights = weights[id, :]
    bias = bias[id]

    class_weights = class_weights[:, np.newaxis, np.newaxis]

    CAM = channels * class_weights + bias
    CAM = CAM.sum(0)
    CAM = CAM - np.min(CAM)
    CAM_img = CAM / (1e-7 + np.max(CAM))
    CAM_img = np.float32(cv2.resize(CAM_img, (shape[1], shape[0])))
    output_CAM = np.uint8(255 * CAM_img)

    return output_CAM


def get_CAM_clip_results(channels: np.ndarray, last_fc: torch.nn.Linear, shape: tuple, id: int, use_bias=False):
    weights = last_fc.weight.data.cpu().numpy()
    if use_bias:
        bias = last_fc.bias.data.cpu().numpy()

    class_weights = weights[id, :]
    if use_bias:
        bias = bias[id]

    class_weights = class_weights[:, np.newaxis, np.newaxis]

    if use_bias:
        CAM = channels * class_weights + bias
    else:
        CAM = channels * class_weights

    CAM = CAM.sum(0)
    CAM = np.clip(CAM, 0, CAM.max())
    CAM = CAM - np.min(CAM)
    CAM_img = CAM / (1e-7 + np.max(CAM))
    CAM_img = np.float32(cv2.resize(CAM_img, (shape[1], shape[0])))
    output_CAM = np.uint8(255 * CAM_img)

    return output_CAM


def get_CAM_clip_weights(channels: np.ndarray, last_fc: torch.nn.Linear, shape: tuple, id: int, use_bias=False):
    weights = last_fc.weight.data.cpu().numpy()

    if use_bias:
        bias = last_fc.bias.data.cpu().numpy()

    class_weights = weights[id, :]
    if use_bias:
        bias = bias[id]

    class_weights = class_weights[:, np.newaxis, np.newaxis]
    class_weights = np.clip(class_weights, 0, class_weights.max())

    if use_bias:
        CAM = channels * class_weights + bias
    else:
        CAM = channels * class_weights
    CAM = CAM.sum(0)
    CAM = CAM - np.min(CAM)
    CAM_img = CAM / (1e-7 + np.max(CAM))
    CAM_img = np.float32(cv2.resize(CAM_img, (shape[1], shape[0])))
    output_CAM = np.uint8(255 * CAM_img)

    return output_CAM


def setup_reproducability(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_dataset_mean_and_std(trainingSet):
    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in trainingSet:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum(image[:, :, 0])
        G_total = G_total + np.sum(image[:, :, 1])
        B_total = B_total + np.sum(image[:, :, 2])

    R_mean = R_total / total_count
    G_mean = G_total / total_count
    B_mean = B_total / total_count

    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in trainingSet:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum((image[:, :, 0] - R_mean) ** 2)
        G_total = G_total + np.sum((image[:, :, 1] - G_mean) ** 2)
        B_total = B_total + np.sum((image[:, :, 2] - B_mean) ** 2)

    R_std = sqrt(R_total / total_count)
    G_std = sqrt(G_total / total_count)
    B_std = sqrt(B_total / total_count)

    return [R_mean / 255, G_mean / 255, B_mean / 255], [R_std / 255, G_std / 255, B_std / 255]


def write_metric_results_to_file(wandb_rund_dir, accuracies=None, kappa_scores=None, weighted_kappa_scores=None,
                                 sensitivities=None, specificities=None, macro_precisions=None,
                                 macro_recalls=None, macro_f1s=None, class_precisions=None, class_recalls=None,
                                 class_f1s=None,
                                 accuracies_r=None, kappa_scores_r=None, sensitivities_r=None, specificities_r=None,
                                 precisions_r=None, recalls_r=None, f1s_r=None):
    results = []
    results.append("\n------------- 4-class Score -------------\n")
    if accuracies is not None:
        results.append("Accuracies: " + str(accuracies))
        results.append("Accuracy mean: " + str(mean(accuracies)))
        results.append("Accuracy stddev: " + str(pstdev(accuracies)))
        results.append("")
    if kappa_scores is not None:
        results.append("Kappas: " + str(kappa_scores))
        results.append("kappa mean: " + str(mean(kappa_scores)))
        results.append("kappa stddev: " + str(pstdev(kappa_scores)))
        results.append("")
    if weighted_kappa_scores is not None:
        results.append("QWK: " + str(weighted_kappa_scores))
        results.append("QWK mean: " + str(mean(weighted_kappa_scores)))
        results.append("QWK stddev: " + str(pstdev(weighted_kappa_scores)))
        results.append("")
    if sensitivities is not None:
        results.append("Sensitivities: " + str(sensitivities))
        results.append("Sensitivity mean: " + str(mean(sensitivities)))
        results.append("Sensitivity stddev: " + str(pstdev(sensitivities)))
        results.append("")
    if specificities is not None:
        results.append("Specificities: " + str(specificities))
        results.append("Specificity mean: " + str(mean(specificities)))
        results.append("Specificity stddev: " + str(pstdev(specificities)))
        results.append("")
    if macro_precisions is not None:
        results.append("Macro precision: " + str(macro_precisions))
        results.append("Macro precision mean: " + str(mean(macro_precisions)))
        results.append("Macro precision stddev: " + str(pstdev(macro_precisions)))
        results.append("")
    if macro_recalls is not None:
        results.append("Macro Recall: " + str(macro_recalls))
        results.append("Macro Recall mean: " + str(mean(macro_recalls)))
        results.append("Macro Recall stddev: " + str(pstdev(macro_recalls)))
        results.append("")
    if macro_f1s is not None:
        results.append("Macro f1: " + str(macro_f1s))
        results.append("Macro f1 mean: " + str(mean(macro_f1s)))
        results.append("Macro f1 stddev: " + str(pstdev(macro_f1s)))
        results.append("")
    if class_precisions is not None:
        results.append("Mayo-0 precision: " + str(class_precisions[:, 0]))
        results.append("Mayo-0 precision mean: " + str(mean(class_precisions[:, 0])))
        results.append("Mayo-0 precision stddev: " + str(pstdev(class_precisions[:, 0])))
        results.append("")
    if class_recalls is not None:
        results.append("Mayo-0 recall: " + str(class_recalls[:, 0]))
        results.append("Mayo-0 recall mean: " + str(mean(class_recalls[:, 0])))
        results.append("Mayo-0 recall stddev: " + str(pstdev(class_recalls[:, 0])))
        results.append("")
    if class_f1s is not None:
        results.append("Mayo-0 f1: " + str(class_f1s[:, 0]))
        results.append("Mayo-0 f1 mean: " + str(mean(class_f1s[:, 0])))
        results.append("Mayo-0 f1 stddev: " + str(pstdev(class_f1s[:, 0])))
        results.append("")
    if class_precisions is not None:
        results.append("Mayo-1 precision: " + str(class_precisions[:, 1]))
        results.append("Mayo-1 precision mean: " + str(mean(class_precisions[:, 1])))
        results.append("Mayo-1 precision stddev: " + str(pstdev(class_precisions[:, 1])))
        results.append("")
    if class_recalls is not None:
        results.append("Mayo-1 recall: " + str(class_recalls[:, 1]))
        results.append("Mayo-1 recall mean: " + str(mean(class_recalls[:, 1])))
        results.append("Mayo-1 recall stddev: " + str(pstdev(class_recalls[:, 1])))
        results.append("")
    if class_f1s is not None:
        results.append("Mayo-1 f1: " + str(class_f1s[:, 1]))
        results.append("Mayo-1 f1 mean: " + str(mean(class_f1s[:, 1])))
        results.append("Mayo-1 f1 stddev: " + str(pstdev(class_f1s[:, 1])))
        results.append("")
    if class_precisions is not None:
        results.append("Mayo-2 precision: " + str(class_precisions[:, 2]))
        results.append("Mayo-2 precision mean: " + str(mean(class_precisions[:, 2])))
        results.append("Mayo-2 precision stddev: " + str(pstdev(class_precisions[:, 2])))
        results.append("")
    if class_recalls is not None:
        results.append("Mayo-2 recall: " + str(class_recalls[:, 2]))
        results.append("Mayo-2 recall mean: " + str(mean(class_recalls[:, 2])))
        results.append("Mayo-2 recall stddev: " + str(pstdev(class_recalls[:, 2])))
        results.append("")
    if class_f1s is not None:
        results.append("Mayo-2 f1: " + str(class_f1s[:, 2]))
        results.append("Mayo-2 f1 mean: " + str(mean(class_f1s[:, 2])))
        results.append("Mayo-2 f1 stddev: " + str(pstdev(class_f1s[:, 2])))
        results.append("")
    if class_precisions is not None:
        results.append("Mayo-3 precision: " + str(class_precisions[:, 3]))
        results.append("Mayo-3 precision mean: " + str(mean(class_precisions[:, 3])))
        results.append("Mayo-3 precision stddev: " + str(pstdev(class_precisions[:, 3])))
        results.append("")
    if class_recalls is not None:
        results.append("Mayo-3 recall: " + str(class_recalls[:, 3]))
        results.append("Mayo-3 recall mean: " + str(mean(class_recalls[:, 3])))
        results.append("Mayo-3 recall stddev: " + str(pstdev(class_recalls[:, 3])))
        results.append("")
    if class_f1s is not None:
        results.append("Mayo-3 f1: " + str(class_f1s[:, 3]))
        results.append("Mayo-3 f1 mean: " + str(mean(class_f1s[:, 3])))
        results.append("Mayo-3 f1 stddev: " + str(pstdev(class_f1s[:, 3])))
        results.append("")

    results.append("\n------------- Remission -------------\n")
    if accuracies_r is not None:
        results.append("Accuracies_r: " + str(accuracies_r))
        results.append("Accuracies_r mean: " + str(mean(accuracies_r)))
        results.append("Accuracies_r stddev: " + str(pstdev(accuracies_r)))
        results.append("")
    if kappa_scores_r is not None:
        results.append("kappa_r: " + str(kappa_scores_r))
        results.append("kappa_r mean: " + str(mean(kappa_scores_r)))
        results.append("kappa_r stddev: " + str(pstdev(kappa_scores_r)))
        results.append("")
    if sensitivities_r is not None:
        results.append("sensitivities_r: " + str(sensitivities_r))
        results.append("sensitivities_r mean: " + str(mean(sensitivities_r)))
        results.append("sensitivities_r stddev: " + str(pstdev(sensitivities_r)))
        results.append("")
    if specificities_r is not None:
        results.append("specificities_r: " + str(specificities_r))
        results.append("specificities_r mean: " + str(mean(specificities_r)))
        results.append("specificities_r stddev: " + str(pstdev(specificities_r)))
        results.append("")
    if precisions_r is not None:
        results.append("precisions_r: " + str(precisions_r))
        results.append("precisions_r mean: " + str(mean(precisions_r)))
        results.append("precisions_r stddev: " + str(pstdev(precisions_r)))
        results.append("")
    if recalls_r is not None:
        results.append("recalls_r: " + str(recalls_r))
        results.append("recalls_r mean: " + str(mean(recalls_r)))
        results.append("recalls_r stddev: " + str(pstdev(recalls_r)))
        results.append("")
    if f1s_r is not None:
        results.append("f1s_r: " + str(f1s_r))
        results.append("f1s_r mean: " + str(mean(f1s_r)))
        results.append("f1s_r stddev: " + str(pstdev(f1s_r)))

    results_separated = map(lambda x: x + "\n", results)
    file = open(os.path.join(wandb_rund_dir, "results.txt"), "w")
    file.writelines(results_separated)
    file.close()

    
#=======================================================================================================================================
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, precision_recall_fscore_support
from metrics import get_mean_sensitivity_specificity, classification_report
import wandb

kappa_scores = []                   # Cohen's Kappa 系数
weighted_kappa_scores = []          # 加权Cohen's Kappa 系数
ROCAUC_scores = []
accuracies = []
sensitivities = []            # 平均敏感性
specificities = []            # 特异性

macro_precisions = []         # 宏平均的精确率
macro_recalls = []            # 召回率
macro_f1s = []                # F1得分

''' 缓解分析指标 '''
precisions_r = []      # 精确率
recalls_r = []         # 召回率
f1s_r = []             # F1分数

kappa_scores_r = []    # Kappa系数
accuracies_r = []      # 准确率
sensitivities_r = []   # 敏感性
specificities_r = []   # 特异性


'''
指标计算与日志记录————通过测试集返回的结果进行计算
'''
def index_calculation(args, *,y_true, y_probs, y_pred, r_true, r_probs, r_pred, i=0, group_id):
    print(f"-----------------------------------i={i}--------------------------------")
    if i == 11:
        id_ = i
        i = 0
    else:
        id_ = i
        
    class_precisions = np.zeros([10, args.nb_classes])  # 每个类的精确率、召回率、F1分数
    class_recalls = np.zeros([10, args.nb_classes])     
    class_f1s = np.zeros([10, args.nb_classes])
    
#     # 初始化W&B
#     if args.enable_wandb:
#         wandb.init(project="ulcerative-colitis-classification",
#                    group=args.model_name + "_CV_C" + "_" + group_id,
#                    save_code=True,
#                    reinit=True)
#         wandb.run.name = "epoch_" + str(id)
#         wandb.run.save()

#         config = wandb.config
#         config.exp = os.path.basename(__file__)[:-3]
#         config.model = args.model_name
#         config.dataset = "final_dataset"
#         config.lr = args.lr
#         config.wd = args.weight_decay
#         config.bs = args.batch_size
#         config.num_worker = args.num_workers
#         config.optimizer = args.opt
    
    
    # 针对4类分类任务的类别级别指标
    prf1_4classes = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    # 针对缓解分析任务的二分类指标
    prf1_remission = precision_recall_fscore_support(r_true, r_pred, average="binary")

    # 混淆矩阵
    cm_4class = confusion_matrix(y_true, y_pred)       
    cm_remission = confusion_matrix(r_true, r_pred)

    '''
    存储类别级别指标
    '''

    # 宏平均精度
    class_precisions[i] = prf1_4classes[0]            # 当前折中每个类别的精度
    macro_precision = prf1_4classes[0].mean()         # 当前折中所有类别精度的平均值——即宏平均精度（宏精度不考虑类别数量差异）
    macro_precisions.append(macro_precision)          # 把当前折的宏平均精度添加到列表中，macro_precisions用于存储每一折的宏平均精度

    # 召回率
    class_recalls[i] = prf1_4classes[1]              
    macro_recall = prf1_4classes[1].mean()
    macro_recalls.append(macro_recall)

    # F1分数
    class_f1s[i] = prf1_4classes[2]
    macro_f1 = prf1_4classes[2].mean()
    macro_f1s.append(macro_f1)

    """
    计算总体指标
    """

    '''4-class analysis ————4类分类任务的总体指标'''

    # Kappa系数，并存储
    all_kappa_score = cohen_kappa_score(y_true, y_pred)  # 本折中的测试结果Kappa系数
    kappa_scores.append(all_kappa_score)                 # 添加到kappa_score列表中


    # 加权Kappa系数，并存储
    all_kappa_score_weighted = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    weighted_kappa_scores.append(all_kappa_score_weighted)

    # 本次测试的准确度，并存储
    accuracy = accuracy_score(y_true, y_pred)
    accuracies.append(accuracy)

    # 平均敏感性及特异性，并存储
    mean_sensitivity, mean_specificity = get_mean_sensitivity_specificity(y_true, y_pred)
    sensitivities.append(mean_sensitivity), specificities.append(mean_specificity)

    ''' Remission analysis————缓解分析任务的特定指标 '''

    # Kappa系数，并存储
    remission_kappa_score = cohen_kappa_score(r_true, r_pred)
    kappa_scores_r.append(remission_kappa_score)

    # 本次测试的准确度，并存储
    accuracy_r = accuracy_score(r_true, r_pred)
    accuracies_r.append(accuracy_r)

    precisions_r.append(prf1_remission[0])
    recalls_r.append(prf1_remission[1])
    f1s_r.append(prf1_remission[2])

    cr_r = classification_report(r_true, r_pred, output_dict=True)
    sensitivities_r.append(cr_r["0"]["recall"]), specificities_r.append(cr_r["1"]["recall"])
    
    if id_ >= 0:
        print(f"Results will be saved to: {wandb.run.dir}")
        write_metric_results_to_file(wandb.run.dir, accuracies, kappa_scores, weighted_kappa_scores,
                                     sensitivities, specificities,
                                     macro_precisions,
                                     macro_recalls, macro_f1s, class_precisions, class_recalls, class_f1s,
                                     accuracies_r, kappa_scores_r, sensitivities_r, specificities_r,
                                     precisions_r,
                                     recalls_r, f1s_r)

#=======================================================================================================================================

# 假设 model 是通过 timm.create_model 创建的
def check_and_adjust_pos_embed(model, checkpoint_model, new_input_size):
    # 获取模型的补丁数量和嵌入维度
    if hasattr(model, 'patch_embed'):
        num_patches = model.patch_embed.num_patches
        print(f"------------------num_patches:{num_patches}-----------------------")
        embedding_size = model.pos_embed.shape[-1]
        print(f"++++++++++++++++++++++embedding_size : {embedding_size}++++++++++++++++++")
    else:
        # 假设模型使用卷积层生成补丁嵌入
        input_size = new_input_size
        patch_size = 16  # 根据模型配置确定
        num_patches = (input_size // patch_size) ** 2
        embedding_size = 768

    num_extra_tokens = 1

    # 初始化或提取位置嵌入
    if 'pos_embed' in checkpoint_model:
        print("============================++++++++++++++++++++++++++++")
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        print(f"+++++++++++++++++++++pos_embed_checkpoint shape: {pos_embed_checkpoint.shape}+++++++++++++++++++++++")
    elif hasattr(model, 'pos_embed'):
        pos_embed_checkpoint = model.pos_embed
    else:
        # 手动初始化位置嵌入
        pos_embed_checkpoint = torch.randn(1, num_patches + num_extra_tokens, embedding_size)
        print(f"+++++++++++++++++++++pos_embed_checkpoint shape: {pos_embed_checkpoint.shape}+++++++++++++++++++++++")

        

    # 插值位置嵌入
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(new_input_size // patch_size)
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

    # 更新模型的位置嵌入
    model.pos_embed = torch.nn.Parameter(new_pos_embed)
#     print(model.state_dict().keys())                   # 包含pos_embed 键

