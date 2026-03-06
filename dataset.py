import itertools
import os
import random

import PIL.Image as Image
from PIL import ImageOps, ImageFilter, ImageEnhance
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)


def img_aug_identity(img, scale=None):
    return img


def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.5, 1.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Contrast(img).enhance(v)


def img_aug_brightness(img, scale=[0.5, 1.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_sharpness(img, scale=[0.5, 2.0]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.solarize(img, v)


class RandomResizedCrop(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image, label):
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.8, 1.2), ratio=(1.0, 1.0))
        image = F.resized_crop(image, i, j, h, w, size=(self.img_size, self.img_size))
        label = F.resized_crop(label, i, j, h, w, size=(self.img_size, self.img_size))
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        image = F.hflip(image)
        label = F.hflip(label)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        image = F.vflip(image)
        label = F.vflip(label)
        return image, label


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        return image, label


def get_strong_aug_list():
    op_list = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.75, 1.25]),
        (img_aug_contrast, [0.75, 1.25]),
        (img_aug_brightness, [0.75, 1.25]),
        (img_aug_sharpness, [0.75, 1.25]),
        (img_aug_posterize, [4, 8]),
        (img_aug_solarize, [1, 256]),
    ]
    return op_list


class StrongAugmentation:
    def __init__(self, num_augs, randn_strong_augs=False):
        self.num_augs = num_augs
        self.augment_list = get_strong_aug_list()
        self.randn_strong_augs = randn_strong_augs
        self.total_ops = len(self.augment_list)

    def __call__(self, img):
        if self.randn_strong_augs:
            max_num = np.random.randint(1, high=self.total_ops + 1)
        else:
            max_num = self.num_augs
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return img


class WeakAugmentation(object):
    def __init__(self, args, num_augs, randn_weak_augs=False):
        self.num_augs = num_augs
        self.randn_weak_augs = randn_weak_augs
        self.augment_list = [RandomResizedCrop(args.img_size), RandomHorizontalFlip(), RandomVerticalFlip(), Identity()]
        self.total_ops = len(self.augment_list)

    def __call__(self, image, label):
        if self.randn_weak_augs:
            max_num = np.random.randint(1, high=self.total_ops + 1)
        else:
            max_num = self.num_augs
        ops = random.choices(self.augment_list, k=max_num)
        for op in ops:
            image, label = op(image, label)
        return image, label


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        # assert len(self.primary_indices) >= self.primary_batch_size > 0
        # assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_eternally(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size)
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class LNDataset(Dataset):
    def __init__(self, args, image_names, primary_indices, secondary_indices):
        self.args = args
        self.image_names = image_names
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.strong_aug = StrongAugmentation(
            num_augs=self.args.num_strong_augs, randn_strong_augs=self.args.randn_strong_augs
        )
        self.weak_aug = WeakAugmentation(
            args, num_augs=self.args.num_weak_augs, randn_weak_augs=self.args.randn_weak_augs
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(f"../data/{self.args.dataset}/images", self.image_names[idx])
        label_path = os.path.join(f"../data/{self.args.dataset}/labels", self.image_names[idx])

        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        # Resize first, then apply transform
        resize = T.Resize((self.args.img_size, self.args.img_size))
        image = resize(image)
        label = resize(label)

        # Apply augmentation based on index
        if idx in self.primary_indices:
            if self.args.strong_augs and self.args.weak_augs:
                if random.random() < 0.5:
                    image = self.strong_aug(image)
                else:
                    image, label = self.weak_aug(image, label)
            elif self.args.strong_augs:
                image = self.strong_aug(image)
            elif self.args.weak_augs:
                image, label = self.weak_aug(image, label)
        elif idx in self.secondary_indices and self.args.weak_augs:
            image, label = self.weak_aug(image, label)

        # To tensor whether transform is applied or not
        to_tensor = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.float32)])
        image = to_tensor(image)
        label = to_tensor(label)

        return image, label, self.image_names[idx]


class DataModule:
    def __init__(self, args):
        self.args = args

        # Load shuffled image names from txt files
        train_image_names, val_image_names, test_image_names = [], [], []
        with open(f"../data/{self.args.dataset}/train.txt", "r") as f:
            train_image_names = f.read().splitlines()
        with open(f"../data/{self.args.dataset}/val.txt", "r") as f:
            val_image_names = f.read().splitlines()
        with open(f"../data/{self.args.dataset}/test.txt", "r") as f:
            test_image_names = f.read().splitlines()

        train_len = len(train_image_names)
        num_labeled = int(self.args.labeled_ratio * train_len)
        labeled_indices = list(range(num_labeled))
        unlabeled_indices = list(range(num_labeled, train_len))

        # Initialize datasets with appropriate indices
        self.train_dataset = LNDataset(self.args, train_image_names, labeled_indices, unlabeled_indices)
        self.val_dataset = LNDataset(self.args, val_image_names, [], [])
        self.test_dataset = LNDataset(self.args, test_image_names, [], [])

        # If all data is labeled, then don't use TwoStreamBatchSampler
        if self.args.labeled_ratio == 1.0:
            self.batch_sampler = None
        else:
            self.batch_sampler = TwoStreamBatchSampler(
                primary_indices=labeled_indices,
                secondary_indices=unlabeled_indices,
                batch_size=self.args.batch_size,
                secondary_batch_size=self.args.batch_size - self.args.labeled_bs,
            )

    def train_dataloader(self):
        if self.batch_sampler is None:
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                persistent_workers=True,
                shuffle=True,
                drop_last=True,
            )
        else:
            dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=self.batch_sampler,
                num_workers=self.args.num_workers,
                persistent_workers=True,
            )
        return dataloader

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )
