import time
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import (
    PrototypeWiseContrastiveLoss,
    PrototypeWiseRelationDistillationLoss,
    PseudoUnbiasedCrossEntropy,
    UnbiasedCrossEntropy,
)
from utils.run_utils import *


class Trainer:
    def __init__(
        self,
        model,
        model_old,
        device,
        opts,
        trainer_state=None,
        classes=None,
        logdir=None,
    ):
        self.model_old = model_old
        self.model = model
        self.device = device
        self.step = opts.step
        self.no_mask = (
            opts.no_mask
        )  # if True sequential dataset from https://arxiv.org/abs/1907.13372
        self.overlap = opts.overlap
        self.num_classes = sum(classes) if classes is not None else 0
        self.feat_dim = opts.feat_dim

        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = tot_classes
            self.nb_new_classes = new_classes
        else:
            self.old_classes = 0

        # Select the Loss Type
        reduction = "none"

        self.unce = opts.unce
        self.puce = opts.puce
        self.hard = opts.hard
        self.threshold = opts.base_threshold

        if opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(
                old_cl=self.old_classes, ignore_index=255, reduction=reduction
            )
        elif self.puce and self.old_classes != 0:
            self.criterion = PseudoUnbiasedCrossEntropy(
                old_cl=self.old_classes, ignore_index=255, reduction=reduction
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        self.lprd = opts.loss_prd
        self.lprd_flag = self.lprd > 0.0
        self.lprd_loss = PrototypeWiseRelationDistillationLoss(
            device=self.device,
            num_classes=self.num_classes,
            old_classes=self.old_classes,
            feat_dim=self.feat_dim,
        )

        self.lpcon = opts.loss_pcon
        self.lpcon_flag = self.lpcon > 0.0
        self.lpcon_loss = PrototypeWiseContrastiveLoss(
            device=self.device,
            num_classes=self.num_classes,
            old_classes=self.old_classes,
            feat_dim=self.feat_dim,
            hard=self.hard,
        )

        self.ret_intermediate = self.puce or self.lprd or self.lpcon or self.unce

    def train(
        self,
        cur_epoch,
        optim,
        train_loader,
        world_size,
        scheduler=None,
        print_int=10,
        logger=None,
        prototypes=None,
        count_features=None,
    ):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch + 1, optim.param_groups[0]["lr"]))

        device = self.device
        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lprd = torch.tensor(0.0)
        lpcon = torch.tensor(0.0)

        train_loader.sampler.set_epoch(cur_epoch)

        model.train()
        start_time = time.time()
        start_epoch_time = time.time()
        for cur_step, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            if (
                self.lprd_flag or self.lpcon_flag or self.puce
            ) and self.model_old is not None:
                with torch.no_grad():
                    outputs_old, features_old = self.model_old(
                        images, ret_intermediate=self.ret_intermediate
                    )

            optim.zero_grad()
            outputs, features = model(images, ret_intermediate=self.ret_intermediate)

            if self.lpcon_flag or self.unce:
                feature_size = features["body"].shape[2:]
                prototypes, count_features = self._update_running_stats(
                    (
                        F.interpolate(
                            input=labels.unsqueeze(dim=1).double(),
                            size=feature_size,
                            mode="nearest",
                        )
                    ).long(),
                    features["body"],
                    self.no_mask,
                    self.overlap,
                    self.step,
                    prototypes,
                    count_features,
                )

            # get feature_local_mean and pseudo_labels
            if self.puce or self.lprd_flag or self.lpcon_flag:
                mask = labels != 0
                max_probs, pseudo_labels = torch.softmax(outputs_old, dim=1).max(dim=1)
                pseudo_labels = labels * mask.long() + pseudo_labels * (~mask).long()
                pseudo_labels = pseudo_labels.unsqueeze(1)
                pseudo_labels_down = (
                    F.interpolate(
                        input=pseudo_labels.double(),
                        size=(features["body"].shape[2], features["body"].shape[3]),
                        mode="nearest",
                    )
                ).long()

            if self.puce:
                loss = criterion(
                    outputs, max_probs, pseudo_labels.squeeze(1), self.thresholds
                )
            else:
                loss = criterion(outputs, labels)  # B x H x W

            loss = loss.mean()  # scalar

            # prototype-wise relation distillation loss
            if self.lprd_flag:
                lprd = self.lprd_loss(
                    features=features["body"],
                    features_old=features_old["body"],
                    labels=pseudo_labels_down,
                )
                lprd *= self.lprd

            # Contrastive loss between local and global prototypes
            if self.lpcon_flag:
                feature_size = features["body"].shape[2:]
                lpcon = self.lpcon_loss(
                    features=features["body"],
                    prototypes=prototypes,
                    outputs=(
                        F.interpolate(input=outputs, size=feature_size, mode="nearest")
                    ),
                    labels=pseudo_labels_down,
                )
                lpcon *= self.lpcon

            # xxx first backprop of previous loss (compute the gradients for regularization methods)
            loss_tot = loss + lprd + lpcon

            loss_tot.backward()

            optim.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            reg_loss += +lprd.item() + lpcon.item()
            interval_loss += loss.item() + lprd.item() + lpcon.item()

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(
                    f"Epoch {cur_epoch + 1}, Batch {cur_step + 1}/{len(train_loader)},"
                    f" Loss={interval_loss}, Time taken={time.time() - start_time}"
                )
                logger.debug(
                    f"Loss made of: CE {loss}, " f"lprd {lprd}, " f"lpcon {lpcon}, "
                )
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar("Losses/interval_loss", interval_loss, x)

                interval_loss = 0.0
                start_time = time.time()

        logger.info(
            f"END OF EPOCH {cur_epoch + 1}, TOTAL TIME={time.time() - start_epoch_time}"
        )

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        epoch_loss = epoch_loss / world_size / len(train_loader)
        reg_loss = reg_loss / world_size / len(train_loader)

        logger.info(
            f"Epoch {cur_epoch + 1}, Class Loss={epoch_loss}, Reg Loss={reg_loss}"
        )

        return (epoch_loss, reg_loss), prototypes, count_features

    def validate(
        self,
        loader,
        metrics,
        world_size,
        ret_samples_ids=None,
        logger=None,
        vis_dir=None,
        label2color=None,
        denorm=None,
    ):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        class_loss = 0.0
        reg_loss = 0.0
        lprd = torch.tensor(0.0)
        lpcon = torch.tensor(0.0)

        ret_samples = []
        prototypes = []
        prototypes_label = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if (
                    self.lprd_flag or self.lpcon_flag or self.puce
                ) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(
                            images, ret_intermediate=True
                        )

                outputs, features = model(images, ret_intermediate=True)

                if self.puce:
                    loss = torch.tensor(0.0, device=self.device)
                else:
                    loss = criterion(outputs, labels)  # B x H x W

                loss = loss.mean()  # scalar

                class_loss += loss.item()
                reg_loss += lprd.item() + lpcon.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append(
                        (images[0].detach().cpu().numpy(), labels[0], prediction[0])
                    )

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            class_loss = class_loss / world_size / len(loader)
            reg_loss = reg_loss / world_size / len(loader)

            if logger is not None:
                logger.info(
                    f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)"
                )

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])

    def _update_running_stats(
        self,
        labels_down,
        features,
        sequential,
        overlapped,
        incremental_step,
        prototypes,
        count_features,
    ):
        cl_present = torch.unique(input=labels_down)

        # if overlapped: exclude background as we could not have a reliable statistics
        # if disjoint (not overlapped) and step is > 0: exclude bgr as could contain old classes
        if overlapped or ((not sequential) and incremental_step > 0):
            cl_present = cl_present[1:]

        # if cl_present[-1] == 255:
        #     cl_present = cl_present[:-1]

        features_local_mean = torch.zeros(
            [self.num_classes, self.feat_dim], device=self.device
        )

        for cl in cl_present:
            if len(features.shape) == 5:
                features_cl = (
                    features[
                        (labels_down == cl).expand(-1, features.shape[1], -1, -1, -1)
                    ]
                    .view(features.shape[1], -1)
                    .detach()
                )
            else:
                features_cl = (
                    features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)]
                    .view(features.shape[1], -1)
                    .detach()
                )
                features_cl = F.normalize(features_cl, p=2, dim=0)
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            # cumulative moving average for each feature vector
            # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
            features_running_mean_tot_cl = (
                features_cl_sum + count_features.detach()[cl] * prototypes.detach()[cl]
            ) / (count_features.detach()[cl] + features_cl.shape[-1])
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl

        return prototypes, count_features

    def find_median(self, train_loader, device, logger):
        """Find the median prediction score per class with the old model.

        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.

        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        max_value = 1.0
        nb_bins = 20  # Bins of 0.05 on a range [0, 1]

        histograms = (
            torch.zeros(self.nb_current_classes, nb_bins).long().to(self.device)
        )

        for cur_step, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs_old, features_old = self.model_old(images, ret_intermediate=False)

            mask_bg = labels == 0
            probas = torch.softmax(outputs_old, dim=1)
            max_probas, pseudo_labels = probas.max(dim=1)

            values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
                accumulate=True,
            )

            if cur_step % 10 == 0:
                logger.info(f"Median computing {cur_step}/{len(train_loader)}.")

        thresholds = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
            self.device
        )  # zeros or ones? If old_model never predict a class it may be important

        logger.info("Approximating median")
        for c in range(self.nb_current_classes):
            total = histograms[c].sum()
            if total <= 0.0:
                continue

            half = total / 2
            running_sum = 0.0
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (
                    running_sum + histograms[c, bin_index]
                ):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + (
                (half - running_sum) / histograms[c, bin_index].sum()
            ) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = self.threshold

        for c in range(len(thresholds)):
            thresholds[c] = min(thresholds[c], base_threshold)
        logger.info(f"Finished computing median {thresholds}")
        return thresholds.to(device), max_value

    def get_median(self, train_loader, logger):
        logger.info("Find median score")
        self.thresholds, self.max_entropy = self.find_median(
            train_loader, self.device, logger
        )
