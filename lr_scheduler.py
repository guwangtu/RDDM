import torch
from torch.optim.lr_scheduler import _LRScheduler

import math
 
class CosineAnnealingLRWarmup(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, last_epoch=-1):
        self.T_max = T_max            # 余弦退火的总周期数
        self.T_warmup = T_warmup      # warmup 阶段的步数
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            # warmup 阶段：线性增长
            warmup_lr = [base_lr * float(self.last_epoch + 1) / self.T_warmup for base_lr in self.base_lrs]
            return warmup_lr
        else:
            # 余弦退火阶段
            # 计算已经经过的 epoch 数（扣除 warmup）
            epoch_since_warmup = self.last_epoch - self.T_warmup
            cosine_lr = [
                base_lr * 0.5 * (1 + math.cos(math.pi * epoch_since_warmup / (self.T_max - self.T_warmup)))
                for base_lr in self.base_lrs
            ]
            return cosine_lr
 