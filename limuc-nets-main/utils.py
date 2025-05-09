# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

"""
动态记录和计算一系列的统计信息
"""
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:            # 格式化字符串，用于输出统计信息（中位数、全局平局值）
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)    # 双端队列里，自动存储最近的window_size 值
        self.total = 0.0       # 所有值的总和
        self.count = 0
        self.fmt = fmt

    # 更新值
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    # 多进程同步
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    # 返回滑动窗口的中位数
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    # 返回滑动窗口中的平均值
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    # 返回全局平均值
    @property
    def global_avg(self):
        return self.total / self.count

    # 返回滑动窗口中的最大值
    @property
    def max(self):
        return max(self.deque)

    # 返回滑动窗口中的最新值
    @property
    def value(self):
        return self.deque[-1]

    # 用于打印日志
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

"""
记录和管理评估或训练过程中的指标（如损失值，准确率等），结合日志记录、进度条显示以及进程同步功能
"""
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)      # 平滑记录指标值（平均值、总和等）
        self.delimiter = delimiter                # 日志输出时的定界符

    # 更新指定的指标值
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    # 实现动态属性访问
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    # 返回当前所有指标的字符串表示
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    # 多进程同步
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    # 添加自定义指标
    def add_meter(self, name, meter):
        self.meters[name] = meter

    # 日志记录器
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        '''
        输出格式
        '''
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]

        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)

        MB = 1024.0 * 1024.0

        '''
        生成器部分
        '''
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__      # Python 的内建print函数存储在__builtin__模块中
    builtin_print = __builtin__.print

    def print(*args, **kwargs):         # 自定义print 函数
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print           # 覆盖


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():   # 获取当期那进程参与分布式训练的总进程数
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():       # 获取当前进程在分布式训练中的rank
    if not is_dist_avail_and_initialized():   # 检查是否启用了分布式训练
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])  # 当前进程在分布式环境中的标识符，通常是进程的唯一ID
        args.world_size = int(os.environ['WORLD_SIZE'])  # 参与分布式训练的进程总数（包括所有节点和GPU)
        args.gpu = int(os.environ['LOCAL_RANK'])         # 当前节点中GPU的标识符，帮助为当前进程分配正确的GPU
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])          # 使用SLURM集群管理系统时的一个标识符，唯一标识当前SLURM作业中的进程
        args.gpu = args.rank % torch.cuda.device_count()     # 确保每个进程使用不同的GPU
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)   # 设置当前进程使用的GPU设备
    args.dist_backend = 'nccl'        # 设置分布式训练的通信后端，‘nccl’—— NVDIA Collective Communication Library
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()       # 同步点，确保在初始化完成后所有进程同步执行
    setup_for_distributed(args.rank == 0)     # 外部函数，只有主进程执行打印输出

    
# -*- coding: utf-8 -*-
"""
@author: Adnan-Sadi
"""
import os
import numpy as np
import configparser

"""
Function gettign the number of images per class from
a particular directory

Args:
    path: data directory
   
Returns:
    img_per_cls: a list of image counts per class
"""
def get_img_num_per_cls(path):
    folders = os.listdir(path)
    img_per_cls = []

    for folder in folders:
        img_path = path +'/'+ folder
        img_count = len(os.listdir(img_path))
        img_per_cls.append(img_count)

    return img_per_cls


"""
Function getting the m_list value of the LDAM loss. The m_list
values are used as class weights in this study.  

Args:
    cls_num_list: list of image counts per class
    max_m: 'maximum margin' hyper parameter used in lDAM loss, 
            defaults to 0.5 
    
Returns:
    m_list: m_list(class_weights/margins) value
"""
def get_mlist(cls_num_list, max_m=0.5):
    m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
    m_list = m_list * (max_m / np.max(m_list))
    
    return m_list