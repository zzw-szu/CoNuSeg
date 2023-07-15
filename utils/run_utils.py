import os
import random

import numpy as np
import torch
from torch import distributed

import tasks
from dataset import (
    CoNSePSegmentationIncremental,
    MoNuSACSegmentationIncremental,
    transform,
)
from utils.logger import Logger


def save_ckpt(
    path,
    model,
    trainer,
    optimizer,
    scheduler,
    epoch,
    best_score,
    prototypes,
    count_features,
):
    """save current model"""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict(),
        "prototypes": prototypes,
        "count_features": count_features,
    }
    torch.save(state, path)


def get_dataset(opts, rank):
    """Dataset And Augmentation"""
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]

    if opts.dataset == "monusac":
        data_mean = [0.67038155, 0.5370589, 0.7303036]
        data_std = [0.18143843, 0.23448963, 0.19455737]
    elif opts.dataset == "consep":
        data_mean = [0.82759047, 0.7016048, 0.84708214]
        data_std = [0.15911889, 0.1978274, 0.12712607]

    train_transform = transform.Compose(
        [
            # transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=data_mean, std=data_std),
        ]
    )

    if opts.crop_val:
        val_transform = transform.Compose(
            [
                transform.Resize(size=opts.crop_size),
                transform.CenterCrop(size=opts.crop_size),
                transform.ToTensor(),
                transform.Normalize(mean=data_mean, std=data_std),
            ]
        )
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose(
            [
                transform.ToTensor(),
                transform.Normalize(mean=data_mean, std=data_std),
            ]
        )

    labels, labels_old, path_base = tasks.get_task_labels(
        opts.dataset, opts.task, opts.step
    )
    labels_cum = labels_old + labels

    if opts.dataset == "monusac":
        dataset = MoNuSACSegmentationIncremental
    elif opts.dataset == "consep":
        dataset = CoNSePSegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if opts.no_mask:
        path_base += "-oldclICCVW2019"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(
        root=opts.data_root,
        train="train",
        transform=train_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        idxs_path=path_base + f"/train-{opts.step}.npy",
        masking=not opts.no_mask,
        overlap=opts.overlap,
        rank=rank,
    )

    # if not opts.no_cross_val:  # if opts.cross_val:
    #     train_len = int(0.8 * len(train_dst))
    #     val_len = len(train_dst)-train_len
    #     train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    # else:  # don't use cross_val
    val_dst = dataset(
        root=opts.data_root,
        train="val",
        transform=val_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        idxs_path=path_base + f"/val-{opts.step}.npy",
        masking=not opts.no_mask,
        overlap=True,
    )

    image_set = "train" if opts.val_on_trainset else "val"
    test_dst = dataset(
        root=opts.data_root,
        train="test",
        transform=val_transform,
        labels=list(labels_cum),
        idxs_path=path_base + f"/test-{opts.step}.npy",
    )

    return train_dst, val_dst, test_dst, len(labels_cum)


def define_distrib_training(opts, logdir_full):
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    torch.cuda.set_device(device_id)
    rank, world_size = 0, 1

    if rank == 0:
        logger = Logger(
            logdir_full,
            rank=rank,
            debug=opts.debug,
            summary=opts.visualize,
            step=opts.step,
        )
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    return device, rank, world_size, logger


def setup_random_seeds(opts):
    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
