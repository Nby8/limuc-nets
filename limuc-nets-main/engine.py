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
        
        if isinstance(outputs, dict):
            _, predicted = torch.max(outputs['main'], dim=1)    #获取预测类别，返回最大值及最大值对应的索引，outputs 为二维数组
        else:
            _, predicted = torch.max(outputs, dim=1)    #获取预测类别，返回最大值及最大值对应的索引，outputs 为二维数组
        
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
def get_test_set_results(args,device, id_, dataset_train, test_dir, model_name):
    
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
                        
        check_and_adjust_pos_embed(args,model,checkpoint_model,224)
        
        '''
        加载权重到模型
        '''
        model.load_state_dict(checkpoint_model)

    else:
        checkpoint = torch.load("../save_log/bestAcc_checkpoint.pth")
        model_state_dict = checkpoint['model']
        
        check_and_adjust_pos_embed(args,model,model_state_dict,224)
        
        model.load_state_dict(model_state_dict,weights_only=False,strict=False)
    model.to(device)

    # 获取测试结果
    y_true, y_probs, y_pred, r_true, r_probs, r_pred = get_test_results_classification(model, test_loader, device,
                                                                                       True, [2, 3])

    return y_true, y_probs, y_pred, r_true, r_probs, r_pred
