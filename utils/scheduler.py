"""学习率调度器（cosine + warmup）。"""

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> LambdaLR:
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            ratio = current_step / max(1, warmup_steps)
            lr = min_lr + (base_lr - min_lr) * ratio
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        return lr / base_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
