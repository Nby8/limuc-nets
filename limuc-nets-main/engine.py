# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from sklearn.metrics import cohen_kappa_score

#==============================================
from provider import get_dataset_mean_and_std,get_test_results_classification
from ucmayo4 import UCMayo4
import provider
from provider import check_and_adjust_pos_embed
from typing import Callable
import os
from gradcam import *
#==============================================


def train_one_epoch(model: torch.nn.Module, criterion: Callable,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, 
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value: .4f}'))                              # 添加精度指标
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
#             print(f"-----------type(outputs)-----------------")
#             print(f"----------------{type(outputs)}-----------------")
#             print(f"-----------outputs-------------------------")
#             print(f"-----------------{outputs.keys()}-----------------")
            
            if isinstance(outputs, dict):
                main_output = outputs['main']          # 主输出
                aux_output = outputs['aux']           # 辅助输出
            
                # 计算主输出的损失
                # main_sm = main_output.softmax(dim=1)
                main_loss = criterion(main_output, targets)

                # 计算辅助输出的损失
                aux_loss = criterion(aux_output,targets)

                # 计算总损失
                total_loss = main_loss + 0.4 * aux_loss

                loss = total_loss
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        
        _, predicted = torch.max(outputs['main'], dim=1)    #获取预测类别，返回最大值及最大值对应的索引，outputs 为二维数组
        correct = (predicted == targets).sum().item()   # 计算正确预测的数量
        accuracy = correct / targets.size(0)     #计算索引

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc=accuracy)        # 更新精度指标
                            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    '''
    {
    'loss': 0.4567,  # 全局平均损失
    'lr': 0.001234   # 当前学习率的全局平均值
    }
    '''
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()   # 评估阶段不需要计算梯度
def evaluate(data_loader, model, device,args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")   # 用于打印和计算评估过程中的指标（损失值，准确率等）
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    ##
    all_targets = []
    all_preds = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)   # non_blocking = True —— 启动异步数据传输
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():      # 启动自动混合精度（AMP）
            output = model(images)
            loss = criterion(output, target)

        ## 获取预测结果
        pred = torch.argmax(output,dim=1)
        
        ## 将当前批次的真实标签和预测结果添加到列表中
        all_targets.extend(target.cpu().numpy())   # 转换为Numpy 数组并保存
        all_preds.extend(pred.cpu().numpy())
        
        ''' 计算 Top-1 、Top-5 准确率
        Top-1 acc: 预测概率最高的类别是否正确
        Top-5 acc: 预测概率最高的5个类别是否包含正确类别
        ''' 
        acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.nb_classes)))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    ## 计算QWK
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} QWK {qwk: .3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, qwk=qwk))

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()}, **{'qwk': qwk}}


'''
测试集结果获取——加载测试集，并获取模型在测试集上的预测结果
id: 当前实验的编号
test_dir: 测试集所在的目录路径
normalize: 数据标准化的转换操作
'''
import torchvision.transforms as transforms
def get_test_set_results(args,device, id_, dataset_train, test_dir, model_name,cms1,cms2):
    
    channel_means = (0.485, 0.456, 0.406)
    channel_stds = (0.229, 0.224, 0.225)
    
    # 对数据进行标准化处理
    normalize = transforms.Normalize(mean=channel_means,
                                     std=channel_stds)

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(model_name)     # coatnet
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    ''' 数据预处理方式 '''
    if model_name == "Inception_v3":
        test_transform = transforms.Compose([transforms.Resize((299, 299)),
                                             transforms.ToTensor(),
                                             normalize])
    else:
        test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             normalize])
    # 加载测试集
    test_dataset = UCMayo4(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                              pin_memory=True)
    
    print("========================================test set data number================================================")
    print(len(test_dataset))
    print("============================================================================================================")

    # 初始化模型
    model = provider.initialize_model(args,model_name, False, 4)
    
    # 加载权重文件
    if args.CV:
#         checkpoint = torch.load(str(args.output_dir) + "/weights/best_acc_" + args.model_name + '_' + str(id_) + '.pth.tar')
        checkpoint_path = str(args.output_dir) + "/weights/best_acc_" + args.model_name + '_' + str(id_) + '.pth.tar'
        checkpoint = torch.load(checkpoint_path,weights_only=False)
        
        '''
        检查并处理预训练模型的权重，确保与当前模型的结构兼容
        '''
        checkpoint_model = checkpoint['model']
                        
        check_and_adjust_pos_embed(model,checkpoint_model,224)
        
        '''
        加载权重到模型
        '''
        model.load_state_dict(checkpoint_model)

    else:
        checkpoint = torch.load("../save_log/bestAcc_checkpoint.pth")
        model_state_dict = checkpoint['model']
        
        check_and_adjust_pos_embed(model,model_state_dict,224)
        
        model.load_state_dict(model_state_dict,weights_only=False,strict=False)
    model.to(device)

    # 获取测试结果
    y_true, y_probs, y_pred, r_true, r_probs, r_pred = get_test_results_classification(model, test_loader, device,
                                                                                       True, [2, 3])

    # 绘制混淆矩阵
    cms1 = plot_cm(args, y_true, y_pred, id_, cms1, False)     # 4- score
    cms2 = plot_cm(args, r_true, r_pred, id_, cms2, True)     # remission
    
    # 绘制类激活图
#     plot_cam(args, test_dataset, model, id_, device)
    
    # 绘制ROC曲线
    plot_roc(args, y_true, y_probs, id_, False)
    plot_roc(args, r_true, r_probs, id_, True)
    
    return y_true, y_probs, y_pred, r_true, r_probs, r_pred


#_-----------------------------------------------------------------------

def plot_cm(args, y_true, y_pred, id_, cms, is_remission):
    from provider import save_confusion_matrix
    from sklearn.metrics import confusion_matrix
    
    if is_remission:
        root_path = f'../saves/save_image/cm/remission/'
        class_names = ['remission', 'no_remission']
    else:
        root_path = f'../saves/save_image/cm/4_score/'
        class_names = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']
    
    os.makedirs(root_path, exist_ok=True)
    
    save_path = os.path.join(root_path, f'{args.model_name}_fold_{id_}.png')

    cm = confusion_matrix(y_true, y_pred)
    
    cms.append(cm)
    
    save_confusion_matrix(args, cm, class_names,save_path, id_, is_remission=is_remission)
    
    return cms
                          
    
    
def plot_cam(args, test_dataset, model, id_, device):
    
    import random
    from torch.utils.data import Subset
    from provider import gen_cam
    
    CNN_layer1       = model.stages[0].blocks[-1].conv2_kxk
    CNN_layer2       = model.stages[1].blocks[-1].conv2_kxk
    
    transform_layer1 = model.stages[2].blocks[-1].norm2
    transform_layer2 = model.stages[3].blocks[-1].norm2  
    
    # 我们只想选取部分图像作为 Grad-CAM 输入
    # 方法 1: 固定索引
#     selected_indices = list(range(100))  # 选择前 100 张图像

    # 方法 2: 随机选择
    num_samples = 20  # 选择 5 张随机图像
    selected_indices = random.sample(range(len(test_dataset)), num_samples)
    
    # 创建子集
    subset_test_dataset = Subset(test_dataset, selected_indices)
    # 创建 DataLoader
    sub_loader = torch.utils.data.DataLoader(
        subset_test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    for i, (image, label) in enumerate(sub_loader):
        
        img = image.to(device)
#         image = img.cpu()
#         image = unnormalize(image)
        
#         image = image.squeeze(0)
#         r_channel = image[0, :, :] 
#         g_channel = image[1, :, :] 
#         b_channel = image[2, :, :] 
#         rgb_image = np.stack([r_channel, g_channel, b_channel], axis=-1)       
        
#         plt.imshow(rgb_image)  
#         plt.axis('off')  
#         plt.savefig(f'../saves/save_image/orin/orin{i}.png')
         
        
        grad_cam_C1 = GradCam(model, CNN_layer1)
        grad_cam_C2 = GradCam(model, CNN_layer2)
        grad_cam_T1 = GradCam(model, transform_layer1)
        grad_cam_T2 = GradCam(model, transform_layer2)
        
        mask_C1 = grad_cam_C1(img)
        mask_C2 = grad_cam_C2(img)
        mask_T1 = grad_cam_T1(img)
        mask_T2 = grad_cam_T2(img)
        
        result_C1 = gen_cam(img, mask_C1, i, id_)
        result_C2 = gen_cam(img, mask_C2, i, id_)
        result_T1 = gen_cam(img, mask_T1, i, id_)
        result_T2 = gen_cam(img, mask_T2, i, id_)
        
        root_path_C1 = f'../saves/save_image/cam/CNN/fold_{id_}/stage_1/'
        root_path_C2 = f'../saves/save_image/cam/CNN/fold_{id_}/stage_2/'
        root_path_T1 = f'../saves/save_image/cam/transform/fold_{id_}/stage_1/'
        root_path_T2 = f'../saves/save_image/cam/transform/fold_{id_}/stage_2/'
        
        os.makedirs(root_path_C1, exist_ok=True)
        os.makedirs(root_path_C2, exist_ok=True)
        os.makedirs(root_path_T1, exist_ok=True)
        os.makedirs(root_path_T2, exist_ok=True)
        
        path_C1 = os.path.join(root_path_C1,f'result_C1_{i}.png')
        path_C2 = os.path.join(root_path_C2,f'result_C2_{i}.png')
        path_T1 = os.path.join(root_path_T1,f'result_T1_{i}.png')
        path_T2 = os.path.join(root_path_T2,f'result_T2_{i}.png')
        
        cv2.imwrite(path_C1, result_C1)
        cv2.imwrite(path_C2, result_C2)
        cv2.imwrite(path_T1, result_T1)
        cv2.imwrite(path_T2, result_T2)

    
def plot_roc(args, y_true, y_probs,id_ , is_remission):
    from sklearn.metrics import roc_curve, auc, RocCurveDisplay
    from sklearn.preprocessing import label_binarize
    import numpy as np
    import matplotlib.pyplot as plt
    
#     plt.rcParams["font.weight"] = "bold"
#     plt.rcParams["axes.labelweight"] = "bold"
#     plt.rcParams["axes.titleweight"] = "bold"
#     plt.rcParams['font.family'] = 'Times New Roman'
    
    y_onehot = label_binarize(y_true, classes=[0, 1, 2, 3])
    
    y_probs_ = np.array(y_probs)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    #对每个类别计算ROC曲线和AUC值
    classes = list(range(args.nb_classes))
    
    if is_remission:
        root_path = f'../saves/save_image/roc/remission/'
        n_classes = 1
    else:
        root_path = f'../saves/save_image/roc/4_score/'
        n_classes = y_onehot.shape[1]
    
    os.makedirs(root_path, exist_ok=True)
    
    path = os.path.join(root_path, f'roc_{args.model_name}_fold_{id_}.png')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess (AUC = 0.5)')
    
    # 计算每个类别的ROC曲线
    plt.figure(figsize=(10, 10))
    
    for i in range(n_classes):
        if not is_remission:
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:,i], y_probs_[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            print(f"Class {i}:")
            print("y_onehot[:, i]:", y_onehot[:, i])
            print("y_probs_[:, i]:", y_probs_[:, i])
            
            # 绘制ROC曲线
            display = RocCurveDisplay(fpr=fpr[i], tpr=tpr[i], roc_auc=roc_auc[i],
                                     estimator_name=f'Mayo {classes[i]}')
            display.plot(ax=plt.gca(), linestyle='-', linewidth=2, label=f'Mayo {classes[i]} (AUC={roc_auc[i]:.2f})')
        
        else:
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:,1-i], y_probs_)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 绘制ROC曲线
            display = RocCurveDisplay(fpr=fpr[i], tpr=tpr[i], roc_auc=roc_auc[i],
                                     estimator_name=f'remission')
            display.plot(ax=plt.gca(), linestyle='-', linewidth=2, label=f'remission (AUC={roc_auc[i]:.2f})')
            
   
    
    if not is_remission:

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.nb_classes)]))

        # 插值所有类别的TPR到统一的FPR网格上
        mean_tpr = np.zeros_like(all_fpr)
        for j in range(args.nb_classes):
            mean_tpr += np.interp(all_fpr, fpr[j], tpr[j])

        # 计算平均TPR
        mean_tpr /= args.nb_classes

        # 计算宏平均的AUC
        macro_auc = auc(all_fpr, mean_tpr)

        # 绘制宏平均的ROC曲线
        plt.plot(all_fpr, mean_tpr, label=f'Macro-average ROC (AUC = {macro_auc:.2f})',
                 color='navy', linestyle='--', linewidth=2)


    # 绘制参考线
    plt.plot([0,1], [0,1], "k--", label="Random guess (AUC = 0.5)")
    

    plt.xlabel("False Positive Rate", fontsize=38)
    plt.ylabel("True Positive Rate", fontsize=38)
    
    # 调整 x 轴刻度标签（xticks）与 xlabel 的间距
    plt.tick_params(axis='x', which='both', pad=15)  # x 轴方向增加 padding，使 xlabel 离刻度更远

    # 调整 y 轴刻度标签（yticks）与 ylabel 的间距
    plt.tick_params(axis='y', which='both', pad=15)  # y 轴方向增加 padding
    
    plt.tick_params(axis='both', which='major', labelsize=38)
    
#     plt.title(f"{args.model_name} ROC curve for Fold {id_}", fontsize=22,pad=20)
#     plt.legend(loc="lower right")
    if is_remission:
        plt.legend(loc="lower right", prop={'size': 30})
    else:
        plt.legend(loc="lower right", prop={'size': 23})
        
    plt.axis("square")
    plt.grid(True)
    
    plt.tick_params(axis='x', which='both', pad=10)  # x 轴方向增加 padding，使 xlabel 离刻度更远
    plt.tick_params(axis='y', which='both', pad=10)  # y 轴方向增加 padding
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.15)
    plt.savefig(path)