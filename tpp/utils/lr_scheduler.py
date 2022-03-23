import torch
import torch.optim as optim

from torch.optim.lr_scheduler import _LRScheduler


def create_lr_scheduler(optimizer,
        lr_scheduler: str,
        lr_scheduler_patience: int = 1,
        lr_scheduler_milestones: float = 1.0,
        lr_rate_init: float = 1.0,
        lr_scheduler_step_size: int = 1,
        max_epochs: int = 1,
        max_steps: int = 1,
        lr_scheduler_gamma: float = 1.0,
        lr_scheduler_warmup: int = 1
    ):
    if not isinstance(optimizer, optim.Optimizer):
        optimizer = optimizer.optimizer

    if lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            threshold=0,
            patience=lr_scheduler_patience,
            factor=lr_scheduler_gamma)
    elif lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, lr_scheduler_step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
    elif lr_scheduler == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_epochs)
    elif lr_scheduler == 'milestones':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_scheduler_milestones, gamma=lr_scheduler_gamma)
    elif lr_scheduler == 'findlr':
        lr_scheduler = FindLR(optimizer, max_steps)
    elif lr_scheduler == 'noam':
        lr_scheduler = NoamLR(optimizer, warmup_steps=lr_scheduler_warmup)
    elif lr_scheduler == "clr":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr_rate_init * 1.e-2,
            max_lr=lr_rate_init,
            step_size_up=lr_scheduler_step_size,
            step_size_down=lr_scheduler_step_size,
            mode="exp_range",
            cycle_momentum=False,
            gamma=lr_scheduler_gamma)
    elif lr_scheduler == 'calr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_scheduler_step_size,
            eta_min=lr_rate_init * lr_scheduler_gamma)
    else:
        raise NotImplementedError("unknown lr_scheduler " + lr_scheduler)
    return lr_scheduler


class FindLR(_LRScheduler):
    """
    inspired by fast.ai @https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    def __init__(self, optimizer, max_steps, max_lr=10):
        self.max_steps = max_steps
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * ((self.max_lr / base_lr) ** (
                self.last_epoch / (self.max_steps - 1)))
                for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(
            last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
