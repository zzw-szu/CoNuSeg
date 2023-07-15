from functools import partial, reduce

import torch
import torch.nn as nn
import torch.nn.functional as functional

from net.denseunet_2d import DenseUnet_2d


def make_model_v2(opts, classes=None):
    ##network == 'denseunet':
    body = DenseUnet_2d()
    head_channels = 64
    model = DenseUNetSegmentationModule(
        body,
        head_channels,
        classes=classes,
    )
    return model


class DenseUNetSegmentationModule(nn.Module):
    def __init__(
        self,
        body,
        head_channels,
        classes,
    ):
        super(DenseUNetSegmentationModule, self).__init__()
        self.body = body
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(
            classes, list
        ), "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = nn.ModuleList(
            [nn.Conv2d(head_channels, c, kernel_size=1, padding=0) for c in classes]
        )
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):
        x_b = self.body(x)
        x_pl = x_b[0]
        x_b = x_b[1]
        out = []
        for mod in self.cls:
            out.append(mod(x_pl))
        x_o = torch.cat(out, dim=1)

        if ret_intermediate:
            return x_o, x_b, x_pl
        return x_o

    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = bkg_bias - bias_diff

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate)

        sem_logits = out[0] if ret_intermediate else out

        sem_logits = functional.interpolate(
            sem_logits, size=out_size, mode="bilinear", align_corners=False
        )

        if ret_intermediate:
            return sem_logits, {"body": out[1], "pre_logits": out[2]}

        return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
