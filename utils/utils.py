import numpy as np
import torch.nn as nn
from torchvision.transforms.functional import normalize


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def color_map(dataset):
    return monusac_cmap()


def monusac_cmap():
    return np.array(
        [[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255]],
        dtype=np.uint8,
    )


class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]


def convert_bn2gn(module):
    mod = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        num_features = module.num_features
        num_groups = num_features // 16
        mod = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    for name, child in module.named_children():
        mod.add_module(name, convert_bn2gn(child))
    del module
    return mod
