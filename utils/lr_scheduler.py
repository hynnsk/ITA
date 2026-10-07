"""Cosine learning rate with the original linear warmup schedule.

Adapted from https://github.com/mzhaoshuai/Divide-and-Co-training.
"""

import math


class CosineWarmupScheduler:
    def __init__(self, init_lr, total_steps, warmup_steps, min_lr=1e-8):
        self.init_lr = init_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, optimizer, global_step):
        if self.warmup_steps > 0 and global_step <= self.warmup_steps:
            lr = global_step / self.warmup_steps * (self.init_lr - self.min_lr)
            lr = min(lr + self.min_lr, self.init_lr)
        else:
            progress = (global_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = 0.5 * self.init_lr * (1.0 + math.cos(progress * math.pi))
        for group in optimizer.param_groups:
            group["lr"] = max(lr, self.min_lr)
