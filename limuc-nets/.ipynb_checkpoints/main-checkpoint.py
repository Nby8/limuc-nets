# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import wandb
import os

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from datasets import read_split_data_CV
from datasets import create_class_to_index_mapping
from datasets import build_transform
from datasets import MyDataSet
from sklearn.model_selection import StratifiedKFold

#==============================================
from engine import get_test_set_results
from provider import index_calculation,check_and_adjust_pos_embed
from losses import ClassDistanceWeightedLoss
from augment import read_and_augment
#==============================================

from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

import models
import models_v2

import utils


def get_args_parser():
    parser = argparse.ArgumentParser('CoAtNet training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--nb-classes', default=4, type=int)
    parser.add_argument('--model-name', default='DeiT', type=str, help='The abbreviation of model name')
    parser.add_argument("--enable_wandb", choices=["True", "False"], default="True",
                    help="if True, logs training details into wandb platform. Wandb settings in the OS should be performed before using this option.")

    # Model parameters
    parser.add_argument('--model', default='deit_base_distilled_patch16_224 ', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Model Exponential Moving Average(EMA) 模型指数移动平均——稳定训练过程
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')     # 模型EMA的衰减率
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')  # 强制将模型EMA的计算放在CPU上

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',    # 优化器类型
                        help='Optimizer (default: "adamw"')                        
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',   # 优化器epsilon ε参数
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',  # 优化器的betas βs参数
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',     #设置梯度裁剪的阈值
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',          # SGD优化器的动量参数
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.005,                  #权重衰减（L2正则化）的系数
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',     # 学习率调度器类型
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',                 #初始学习率
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',    #学习率噪声的开启和关闭时机
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',    # 学习率噪声的最大扰动百分比
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',      # 学习率噪声的标准差
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',            # 预热阶段的学习率
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',               # 学习率的最小值
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',            # 学习率衰减的周期
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',              # 预热阶段的epoch数
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',            # 冷却阶段的epoch数
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',            # 设置 Plateau 调度器的耐心值（以 epoch 为单位）
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',   # 设置学习率衰减的速率
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',          # 设置颜色抖动的强度
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',   #指定自动数据增强策略（随机增强策略）
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')   # 设置标签平滑的强度
    parser.add_argument('--CDW', action='store_true')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',                            # 设置图像缩放时的插值方法
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')                               # 启用重复增强
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')      # 禁用重复增强
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')                                 # 启用训练模式
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')          # 禁用训练模式
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment                     三种数据增强策略的组合
    
    parser.add_argument('--src', action='store_true') #simple random crop                     启用简单随机裁剪
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',      # 设置随机擦除的概率
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',                     # 设置随机擦除的模式
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,                          # 设置每张图像中随机擦除的次数
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,           # 控制是否对第一个数据增强分支应用随机擦除
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,                       # 设置Mixup 的α 参数
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,                       # 设置CutMix 的α 参数
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,       # 设置CutMix的最小和最大裁剪比例
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,                      # 设置 Mixup或CutMix的概率
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,               # 设置从 Mixup 切换到 CutMix的概率
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')  
    parser.add_argument('--mixup-mode', type=str, default='batch',                     # 设置Mixup 或 CutMix 的应用模式
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',    # 指定教师模型的名称
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')                                 # 指定教师模型的权重文件的路径
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")     # 指定知识蒸馏的类型
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")               # 设置蒸馏损失的权重
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")                 # 设置蒸馏的温度参数
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') # 启用 Co-Substitution 技术，该技术是一种数据增强或正则化技术，通常用于提升模型的泛化能力
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')              # 指定从哪个检查点开始微调
    parser.add_argument('--attn-only', action='store_true')                                     # 仅微调注意力层
    parser.add_argument('--CV', action='store_true', help='Using 10 fold cross validation')
    
    # Dataset parameters
    parser.add_argument('--data-path', default='../LIMUC/train_and_validation_sets/', type=str,      # 指定数据集的路径
                        help='dataset path')
    parser.add_argument('--test-path',default='../LIMUC/test_set/',type=str,help='test set path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19','LIMUC'], type=str, help='Image Net dataset path')    # 指定数据集的名称
                        
    parser.add_argument('--inat-category', default='name',                                              # 指定iNaturalist 数据集的分类粒度
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='../saves/save_log/',                                                     # 指定输出目录的路径
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',                                                     # 指定训练和测试使用的设备
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)                                                  # 设置随机种子
    parser.add_argument('--resume', default='', help='resume from checkpoint')                          # 从指定的检查点恢复训练
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',                              # 设置起始的训练时间
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')                  # 仅执行评估，不执行训练
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")       # 设置评估时的裁剪比例
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')     # 启用分布式评估
    parser.add_argument('--num_workers', default=10, type=int)                                          # 设置数据加载的子进程数
    parser.add_argument('--pin-mem', action='store_true',                                               # 启用内存锁定
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',                           # 禁用内存锁定
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')  # 启用分布式训练
    parser.add_argument('--world_size', default=1, type=int,                                                         # 设置分布式训练的进程总数
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')             # 设置分布式训练的初始化URL
    
    parser.add_argument("-est", "--early_stopping_threshold", type=int, default=25,
                    help="early stopping threshold to terminate training.")
    
    parser.add_argument("-alpha",type=float,default=5.0, help="alpha for CDW-CE loss")
    return parser

#==========================================================run===============================================================================

best_threshold = 0.0001                   # 判断验证集性能是否显著提升的阈值

def run_model(args,dataset_val,data_loader_train, data_loader_val,device,fold=0):
    # 创建模型
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    """
    加载预训练权重并进行微调
    """                
    checkpoint = dict([])
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        elif args.finetune == "pretrained":
            print("----------------pretrained-----------------")
            checkpoint['model'] = model.state_dict()
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        '''
        检查并处理预训练模型的权重，确保与当前模型的结构兼容
        '''
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

#         '''
#         插值位置编码
#         '''
#         print("++++++++++++++++++++++++++++++++++++")
#         print(checkpoint_model.keys())
#         print("++++++++++++++++++++++++++++++++++++")
#         
#         # interpolate position embedding
#         pos_embed_checkpoint = checkpoint_model['pos_embed']
#         embedding_size = pos_embed_checkpoint.shape[-1]
#         num_patches = model.patch_embed.num_patches
#         num_extra_tokens = model.pos_embed.shape[-2] - num_patches
#         # height (== width) for the checkpoint position embedding
#         orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
#         # height (== width) for the new position embedding
#         new_size = int(num_patches ** 0.5)
#         # class_token and dist_token are kept unchanged
#         extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
#         # only the position tokens are interpolated
#         pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
#         pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
#         pos_tokens = torch.nn.functional.interpolate(
#             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
#         pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
#         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
#         checkpoint_model['pos_embed'] = new_pos_embed


        check_and_adjust_pos_embed(model,checkpoint_model,224)
    
        '''
        加载权重到模型
        '''
        model.load_state_dict(checkpoint_model, strict=False)
        
        
    '''
    模型微调
    '''
#     print("_______________________________________________________________________________")
#     param_names = [name for name, _ in model.named_parameters()]
#     print(param_names)
#     print("_______________________________________________________________________________")
    for name_p, p in model.named_parameters():
        if 'stem.' or 'stages.' in name_p:
            p.requires_grad = False
        else:
            p.requires_grad = True
        
        if 'stages.3' in name_p or 'stages.2' in name_p or 'stages.1' in name_p:
            p.requires_grad = True
        
    try:
        model.head.fc.weight.reqires_grad = True
        model.head.fc.bias.reequires_gras = True
    except:
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
    try:
        model.pos_embed.requires_grad = True
    except:
        print('no position encoding')
    try:
        for p in models.pach_embed.parameters():
            p.requires_grad = False
    except:
        print('no patch embed')

    """
    仅微调注意力层，冻结其他层的参数
    """
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

#     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#     '''判断是否所有的模型权重都是可更新的'''
#     for name, param in model.named_parameters():
#         print(f"Parameter:{name}, Requires Grad:{param.requires_grad}")
    
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
#     print(f"Tranable parameters: {trainable_params}")
#     print(f"Non-trainable parameters:{non_trainable_params}")
          
#     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    
    '''
    初始化模型指数移动平均
    '''
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
     
    '''
    若启用了分布式训练,则使用DistributedDataParallel 包装模型
    '''
    model_without_ddp = model   # 保存未包装的模型，用于优化器和学习率调度器的创建
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)     # 计算模型参数量
    print('number of params:', n_parameters)
    
    '''
    调整学习率
    '''
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    '''
    创建优化器
    '''
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()          #初始化混合精度训练的损失缩放器

    '''
    创建学习率调度器
    '''
    lr_scheduler, _ = create_scheduler(args, optimizer)


    '''
    设置损失函数
    '''
    criterion = LabelSmoothingCrossEntropy()
    
    # 图像标签融合
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.CDW:
        print("-------------------------------------using CDW loss--------------------------------------")
        criterion = ClassDistanceWeightedLoss(4,args.alpha)
    elif mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    '''
    创建并初始化教师模型
    '''
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'], strict=False)
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    '''
    从检查点恢复训练
    '''
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

        lr_scheduler.step(args.start_epoch)  # 更新学习率调度器的状态
        
    '''
    只使用评估模式：使用验证集对模型进行评估
    '''
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    '''
    开始训练
    '''
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_qwk = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        ''' 如果启用分布式训练 '''
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        ''' eg.
        {
            'loss': 0.4567,  # 全局平均损失
            'lr': 0.001234   # 当前学习率的全局平均值
        }
        '''
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        lr_scheduler.step(epoch)
        
        '''每个epoch保存检查点'''
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'fold': fold,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             
        ''' 训练阶段同样需要验证评估模型 '''
        test_stats = evaluate(data_loader_val, model, device,args)
        
        
        if args.enable_wandb:
            wandb.log(
                    {"epoch"     : epoch + 1,
                     "lr"        : optimizer.param_groups[0]['lr'],
                     'train loss': train_stats["loss"],
                     'val loss'  : test_stats["loss"],
                     'train acc' : train_stats["acc"],
                     'val acc'   : test_stats["acc1"]})
        
        
        if args.CV:
            print(f"***************************************************Fold {fold+1}******************************************************")
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        ''' 在测试过程中保存模型的最佳检查点'''
        if (max_accuracy * (1 + best_threshold)) < test_stats["acc1"]:
            early_stop_counter = 0
            max_accuracy = test_stats["acc1"]
            
            if args.enable_wandb:
                wandb.run.summary["best accuracy"] = max_accuracy
                
            if args.output_dir:
                if args.CV:
                    checkpoint_paths = [str(output_dir)+ "/weights/best_acc_" + args.model_name + '_' + str(fold) + '.pth.tar']
                else:
                    checkpoint_paths = [output_dir / 'bestAcc_checkpoint.pth']
                
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'fold': fold,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
        else:
            early_stop_counter += 1
            
        
#         if early_stop_counter >= args.early_stopping_threshold:
#             print("Early stopint at:" + str(epoch))
#             break
                    
        if max_qwk < test_stats["qwk"]:
            max_qwk = test_stats["qwk"]
#             if args.output_dir:
#                 if args.CV:
#                     checkpoint_paths = [str(output_dir)+ "/weights/best_qwk_" + args.model_name + '_' + str(fold) + '.pth.tar']
#                 else:
#                     checkpoint_paths = [output_dir / 'bestQWK_checkpoint.pth']
#                 for checkpoint_path in checkpoint_paths:
#                     utils.save_on_master({
#                         'model': model_without_ddp.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'lr_scheduler': lr_scheduler.state_dict(),
#                         'epoch': epoch,
#                         'fold': fold,
#                         'model_ema': get_state_dict(model_ema),
#                         'scaler': loss_scaler.state_dict(),
#                         'args': args,
#                     }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%, max qwk: {max_qwk: .3f}')
        
        if args.CV:
            print(f"***************************************************Fold {fold+1} end******************************************************")

        ###统计：statistics
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        ''' 输出日志 '''
        if args.output_dir and utils.is_main_process() and args.CV:
            with (output_dir / f"Fold{fold}_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        else:
            with (output_dir / f"No_CV_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
    ''' 总用时计算 '''
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.CV:
        print('Fold {} Training time {}'.format(fold,total_time_str))
    else:
        print('No CV Training time {}'.format(total_time_str))


def load_data(args,dataset_train, dataset_val):
    
    # 对训练集进行采样
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:             # 启用数据增强
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:                # 启用分布式评估
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,                          # 内存锁定
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,                          # 内存锁定
        drop_last=False
    )
    
    return data_loader_train, data_loader_val

#=================================================================run end==============================================================================


def main(args):
    utils.init_distributed_mode(args)

    print(args)      # 打印所有命令行参数


    # 不支持同时启用蒸馏和微调
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()  # 根据进程排名调整种子值
    torch.manual_seed(seed)              # 设置 PyTorch 的随机数生成器的种子
    np.random.seed(seed)                 # 设置NumPy的随机数生成器的种子
    # random.seed(seed)

    cudnn.benchmark = True               # 启用CuDNN的自动优化功能
    
#============================================================    
    test_dir = args.test_path
    
    if args.enable_wandb:
        group_id = wandb.util.generate_id()         # 生成唯一的分组ID：group_id，用于将多个实验归类到同一个分组中
#============================================================

#----------------------------------------------------------------------------------------------------------------------------------------
    if not args.CV:
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(is_train=False, args=args)
        data_loader_train, data_loader_val = load_data(args,dataset_train, dataset_val)
        run_model(args,dataset_val,data_loader_train, data_loader_val,device)
                        
        # 每一个fold 进行一次测试集的测试，返回各个结果
        y_true, y_probs, y_pred, r_true, r_probs, r_pred = get_test_set_results(args, device, 11, dataset_train, test_dir, args.model_name)
        
        print("================================y_true===================================")
        print(y_true)
        
        print("================================y_pred===================================")
        print(y_pred)
        
        print("=========================================================================")
                        
        index_calculation(args, y_true=y_true, y_probs=y_probs, y_pred=y_pred, r_true=r_true, r_probs=r_probs, r_pred=r_pred, i=11, group_id = group_id)
        
    else:
        all_images_path, all_images_label,args.nb_classes = read_split_data_CV(args.data_path)
        
#         # 使用数据增强
#         save_dir = "../LILMUC/saves/augmented_images"
#         all_augmented_paths = read_and_augment(all_images_path,save_dir)
        
        
        skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
        
        fold_results = []
        
        for fold,(train_index, val_index) in enumerate(skf.split(all_images_path,all_images_label)):
            print(f'=============================================Fold {fold+1}==============================================')
            
            if args.enable_wandb:
                wandb.init(project="ulcerative-colitis-classification-coatnet_final",
                           group=args.model_name + "final_v1" + "_" + group_id,
                           save_code=True,
                           reinit=True)
                wandb.run.name = "fold_" + str(fold)
                wandb.run.save()

                config = wandb.config
                config.exp = os.path.basename(__file__)[:-3]
                config.model = args.model_name
                config.dataset = "final_dataset"
                config.lr = args.lr
                config.wd = args.weight_decay
                config.bs = args.batch_size
                config.num_worker = args.num_workers
                config.optimizer = args.opt
            
            #根据索引划分训练集与验证集
            train_images_path = [all_images_path[i] for i in train_index]
#             save_dir = f"../LIMUC/saves/augmented_images/fold_{fold}"
#             train_images_path = read_and_augment(train_images_path,save_dir)
            
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(len(train_images_path))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
            train_images_label = [all_images_label[i] for i in train_index]
            
            val_images_path = [all_images_path[i] for i in val_index]
            val_images_label = [all_images_label[i] for i in val_index]
            
            
            #创建类别名称到索引的映射
            class_to_index = create_class_to_index_mapping(train_images_label + val_images_label)
            
            #构建数据集
            dataset_train = MyDataSet(
                images_path=train_images_path,
                images_class=train_images_label,
                transform=build_transform(is_train=True,args=args),
                class_to_index=class_to_index
            )
            
            dataset_val = MyDataSet(
                images_path=val_images_path,
                images_class=val_images_label,
                transform=build_transform(is_train=False,args=args),
                class_to_index=class_to_index
            )
        
            data_loader_train, data_loader_val = load_data(args,dataset_train, dataset_val)
            run_model(args,dataset_val,data_loader_train, data_loader_val,device,fold=fold)
                        
            # 每一个fold 进行一次测试集的测试，返回各个结果
            y_true, y_probs, y_pred, r_true, r_probs, r_pred = get_test_set_results(args,device,fold, dataset_train, test_dir, args.model_name)

            index_calculation(args,y_true=y_true, y_probs=y_probs, y_pred=y_pred, r_true=r_true, r_probs=r_probs, r_pred=r_pred, i=fold, group_id = group_id)
            
#--------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
