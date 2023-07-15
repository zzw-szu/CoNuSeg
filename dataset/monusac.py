import os

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed, unique, zeros_like

from .utils import Subset, filter_images

classes = {
    0: "Background",
    1: "Neutrophil",
    2: "Macrophage",
    3: "Lymphocyte",
    4: "Epithelial",
}


class MoNuSACSegmentation(data.Dataset):
    def __init__(self, root, image_set="train", is_aug=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.image_set = image_set
        base_dir = "MoNuSAC"
        monusac_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(monusac_root, "splits")

        if not os.path.isdir(monusac_root):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"'
            )

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(" ") for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [
            (os.path.join(monusac_root, x[0][1:]), os.path.join(monusac_root, x[1][1:]))
            for x in file_names
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert("RGB")
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class MoNuSACSegmentationIncremental(data.Dataset):
    def __init__(
        self,
        root,
        train="train",
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        masking=True,
        overlap=True,
        rank=0,
    ):
        full_monusac = MoNuSACSegmentation(root, train, is_aug=True, transform=None)

        self.labels = []
        self.labels_old = []

        self.rank = rank

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old

            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_monusac, labels, labels_old, overlap=overlap)

                if idxs_path is not None and self.rank == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if train:
                masking_value = 0
            else:
                masking_value = 255

            self.inverted_order = {
                label: self.order.index(label) for label in self.order
            }
            self.inverted_order[255] = masking_value

            reorder_transform = self.tmp_funct1

            if masking:
                target_transform = self.tmp_funct3
            else:
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_monusac, idxs, transform, target_transform)
        else:
            self.dataset = full_monusac

    def tmp_funct1(self, x):
        tmp = zeros_like(x)
        for value in unique(x):
            if value in self.inverted_order:
                new_value = self.inverted_order[value.item()]
            else:
                new_value = self.inverted_order[255]  # i.e. masking value
            tmp[x == value] = new_value
        return tmp

    def tmp_funct3(self, x):
        tmp = zeros_like(x)
        for value in unique(x):
            if value in self.labels + [255]:
                new_value = self.inverted_order[value.item()]
            else:
                new_value = self.inverted_order[255]  # i.e. masking value
            tmp[x == value] = new_value
        return tmp

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
