'''
Author: nby 2403050046@qq.com
Date: 2025-02-23 10:38:38
LastEditors: nby 2403050046@qq.com
LastEditTime: 2025-02-23 20:33:33
FilePath: \deit-main\losses.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion   # 基础损失函数，用于计算学生模型与实际标签之间的损失
        self.teacher_model = teacher_model      # 教师模型，用于生成软标签（硬标签），提供给学生模型作为额外的监督信号
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type      # 蒸馏类型
        self.alpha = alpha                       # 权重系数，用于平衡基础损失与蒸馏损失的贡献。α 越大，蒸馏损失影响越大
        self.tau = tau          # 蒸馏的温度超参数，温度（T) 控制教师模型输出的平滑度，“软度”

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model  
                    输入图像，传入教师模型和学生模型
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
                学生模型的输出，可能是一个包含两个个元素的元组，第一个元组是学生模型的原始输出（通常是分类概率），第二个元素是学生模型的蒸馏输出（用于与教师模型输出进行比较）。
            labels: the labels for the base criterion
                    目标标签用于计算基础损失
        """
        outputs_kd = None  # 学生模型用于知识蒸馏的输出（来自dist_token)
        
        '''
        检查输出类型
        '''
        if not isinstance(outputs, torch.Tensor):    # 检查outputs 是否是torch.Tensor 类型
            # assume that the model outputs a tuple of [outputs, outputs_kd]
#             outputs, outputs_kd = outputs
            pass

        '''
        计算基础损失——student loss
        '''
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        # 只是蒸馏过程必须存在学生模型的蒸馏输出
        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        
        '''
        获取教师模型的输出
        '''
        # don't backprop throught the teacher——不需要反向传播，因为通常是强大的预训练模型
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        """
        计算蒸馏损失——distillation loss
        """

        '''
        如果是软蒸馏
        '''
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(      # 计算学生和教师模型之间的KL散度
                F.log_softmax(outputs_kd / T, dim=1),  # 学生模型的log 概率分布（outputs_kd—— 学生模型的logits）
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),  # 教师模型的 log 概率分布（teacher_outputs——教师模型的logits）
                reduction='sum',       # 如何计算损失值，‘sum’——对所有样本和类别的损失求和
                log_target=True        # 
            ) * (T * T) / outputs_kd.numel()      # T*T —— 缩放因子，outputs_kd.numel() 返回outputs_kd 张量中元素的总数
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        
        ### 如果是硬蒸馏——直接通过计算学生模型的输出ouputs_kd 与教师模型的预测标签之间的交叉熵损失来进行蒸馏
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        """
        计算总损失——total loss
        """
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


#===============================================================================================================
import torch
from torch import Tensor


class ClassDistanceWeightedLoss(torch.nn.Module):
    """
    Instead of calculating the confidence of true class, this class takes into account the confidences of
    non-ground-truth classes and scales them with the neighboring distance.
    Paper: "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation" (https://arxiv.org/abs/2202.05167)
    It is advised to experiment with different power terms. When searching for new power term, linearly increasing
    it works the best due to its exponential effect.

    """

    def __init__(self, class_size: int, power: float = 5.0, reduction: str = "mean"):
        super(ClassDistanceWeightedLoss, self).__init__()
        self.class_size = class_size
        self.power = power
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_sm = input.softmax(dim=1)

        weight_matrix = torch.zeros_like(input_sm)
        
        for i, target_item in enumerate(target):
            weight_matrix[i] = torch.tensor([abs(k - target_item.item()) for k in range(self.class_size)], dtype=torch.float32)

        weight_matrix.pow_(self.power)

        # TODO check here, stop here if a nan value and debug it
        reverse_probs = (1 - input_sm).clamp_(min=1e-4)

        log_loss = -torch.log(reverse_probs)
        if torch.sum(torch.isnan(log_loss) == True) > 0:
            print("nan detected in forward pass")

        loss = log_loss * weight_matrix
        loss_sum = torch.sum(loss, dim=1)

        if self.reduction == "mean":
            loss_reduced = torch.mean(loss_sum)
        elif self.reduction == "sum":
            loss_reduced = torch.sum(loss_sum)
        else:
            raise Exception("Undefined reduction type: " + self.reduction)

        return loss_reduced