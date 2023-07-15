import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction="mean", ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets, mask=None):
        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = (
            torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den
        )  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(
            dim=1
        )  # B, N, H, W    p(N_i)

        # Following line was fixed more recently in:
        # https://github.com/fcdl94/MiB/commit/1c589833ce5c1a7446469d4602ceab2cdeac1b0e
        # and added to my repo the 04 August 2020 at 10PM
        labels = targets.clone()  # B, H, W

        labels[
            targets < old_cl
        ] = 0  # just to be sure that all labels old belongs to zero

        if mask is not None:
            labels[mask] = self.ignore_index
        loss = F.nll_loss(
            outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction
        )

        return loss


class PrototypeWiseRelationDistillationLoss(nn.Module):
    def __init__(
        self,
        device,
        num_classes,
        old_classes,
        feat_dim,
        current_temp=0.2,
        past_temp=0.01,
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.old_classes = old_classes
        self.feat_dim = feat_dim
        self.current_temp = current_temp
        self.past_temp = past_temp

    def forward(self, features, features_old, labels):
        loss = torch.tensor(0.0, device=self.device)

        cl_present = torch.unique(input=labels)
        features_local_mean = torch.zeros(
            [self.num_classes, self.feat_dim], device=self.device
        )
        features_local_mean_old = torch.zeros(
            [self.num_classes, self.feat_dim], device=self.device
        )
        for cl in cl_present:
            features_cl = features[
                (labels == cl).expand(-1, features.shape[1], -1, -1)
            ].view(features.shape[1], -1)
            features_cl_old = features_old[
                (labels == cl).expand(-1, features.shape[1], -1, -1)
            ].view(features.shape[1], -1)
            features_cl = F.normalize(features_cl, p=2, dim=0)
            features_cl_old = F.normalize(features_cl_old, p=2, dim=0)
            features_local_mean[cl] = torch.mean(features_cl, dim=-1)
            features_local_mean_old[cl] = torch.mean(features_cl_old, dim=-1)

        features_local_mean = features_local_mean[: self.old_classes]
        features_local_mean_old = features_local_mean_old[: self.old_classes]

        features1_sim = torch.div(
            torch.matmul(features_local_mean, features_local_mean.T), self.current_temp
        )
        logits_mask = torch.scatter(
            torch.ones_like(features1_sim),
            1,
            torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
            0,
        )
        logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
        features1_sim = features1_sim - logits_max1.detach()
        row_size = features1_sim.size(0)
        logits1 = torch.exp(
            features1_sim[logits_mask.bool()].view(row_size, -1)
        ) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(
            dim=1, keepdim=True
        )

        features2_sim = torch.div(
            torch.matmul(features_local_mean_old, features_local_mean_old.T),
            self.past_temp,
        )
        logits_max2, _ = torch.max(features2_sim * logits_mask, dim=1, keepdim=True)
        features2_sim = features2_sim - logits_max2.detach()
        logits2 = torch.exp(
            features2_sim[logits_mask.bool()].view(row_size, -1)
        ) / torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(
            dim=1, keepdim=True
        )

        loss = (-logits2 * torch.log(logits1)).sum(1).mean()

        return loss


class PrototypeWiseContrastiveLoss(nn.Module):
    def __init__(
        self, device, num_classes, old_classes, feat_dim, hard=False, temp=0.1
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.old_classes = old_classes
        self.feat_dim = feat_dim
        self.temp = temp
        self.hard = hard

    def forward(self, features, prototypes, outputs, labels):
        loss = torch.tensor(0.0, device=self.device)

        max_probs, pseudo_labels = F.softmax(outputs, dim=1).max(dim=1, keepdim=True)

        cl_present = torch.unique(input=labels)
        features_local_mean = torch.zeros(
            [self.num_classes, self.feat_dim], device=self.device
        )
        for cl in cl_present:
            if cl >= self.old_classes and self.hard:
                position1 = labels == cl
                position2 = pseudo_labels == cl
                position_easy = (position1 & position2).expand(
                    -1, features.shape[1], -1, -1
                )
                position_hard = (position1 & (~position2)).expand(
                    -1, features.shape[1], -1, -1
                )
                weight_cl_easy = 1 - max_probs[position1 & position2]
                features_cl_easy = (
                    features[position_easy].view(features.shape[1], -1) * weight_cl_easy
                )
                features_cl_hard = features[position_hard].view(features.shape[1], -1)
                features_cl = torch.cat([features_cl_easy, features_cl_hard], dim=1)
            else:
                features_cl = features[
                    (labels == cl).expand(-1, features.shape[1], -1, -1)
                ].view(features.shape[1], -1)
            features_cl = F.normalize(features_cl, p=2, dim=0)
            features_local_mean[cl] = torch.mean(features_cl, dim=-1)

        logits1 = torch.div(torch.matmul(features_local_mean, prototypes.T), self.temp)
        mask = torch.zeros_like(logits1)
        for cl in cl_present[1:]:
            mask[cl, cl] = 1
        logits_max, _ = torch.max(logits1, dim=1, keepdim=True)
        logits1 = logits1 - logits_max
        logits2 = torch.exp(logits1[cl_present[1:], 1:]).sum(1)

        logits1 = torch.exp(logits1[mask.bool()])
        loss = -torch.log(logits1 / logits2).mean()

        return loss


class PseudoUnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction="mean", ignore_index=255, temp=0.1):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl
        self.temp = temp
        self.loss_unce = UnbiasedCrossEntropy(
            self.old_cl, self.reduction, self.ignore_index
        )

    def forward(self, outputs, max_probs, labels, thresholds):
        mask_valid_pseudo = max_probs > thresholds[labels]
        mask_background = labels < self.old_cl

        loss_not_pseudo = self.loss_unce(
            outputs, labels, mask=mask_background & mask_valid_pseudo
        )
        _labels = labels.clone()
        _labels[~(mask_background & mask_valid_pseudo)] = 255
        _labels[mask_background & mask_valid_pseudo] = labels[
            mask_background & mask_valid_pseudo
        ]
        loss_pseudo = F.cross_entropy(
            outputs, _labels, ignore_index=255, reduction="none"
        )
        loss = loss_pseudo + loss_not_pseudo

        return loss
