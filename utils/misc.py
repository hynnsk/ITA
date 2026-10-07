import os
import random
import shutil

import numpy as np
import torch


def save_checkpoint(state, is_best, model_dir, filename="checkpoint.pth.tar"):
    filename = os.path.join(model_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("pth.tar", "best.pth.tar"))


def set_random_seed(seed=None):
    """set random seeds for pytorch, random, and numpy.random"""
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
