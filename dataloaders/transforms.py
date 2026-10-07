import torch
import torchvision


class TensorNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return torchvision.transforms.functional.normalize(
            tensor, mean=self.mean, std=self.std, inplace=True
        )


class GroupToTensorBCHW:
    """Convert decoded RGB arrays to a (T, C, H, W) float tensor in [0, 1]."""

    def __call__(self, sample):
        frames = [
            torch.from_numpy(frame).permute(2, 0, 1).contiguous().float().div_(255)
            for frame in sample
        ]
        return torch.stack(frames)
