# Copyright 2022-present, Sebastian Dziadzio.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from argparse import Namespace
from pathlib import Path
from typing import Union

import numpy as np
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.io import read_image

from backbone.ResNet18 import resnet18
from datasets.utils.continual_dataset import ContinualDataset


class FileDataset(Dataset):
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)
        factors = np.load(self.path / "factors.npz", allow_pickle=True)
        self.targets = factors["shape_id"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = self.path / f"sample_{idx}.png"
        img = read_image(str(img_path))
        label = self.targets[idx]
        return img, label


class ContinualBenchmarkDisk:
    def __init__(
        self,
        path: Union[Path, str],
        accumulate_test_set: bool = False,
    ):
        """Initialize the continual learning benchmark.
        Args:
            path: The path to the dataset.
            accumulate_test_set: Whether to accumulate the test set over tasks.
        """
        self.path = Path(path)
        self.accumulate_test_set = accumulate_test_set
        if self.accumulate_test_set:
            self.test_sets = []

    def __iter__(self):
        for task_dir in sorted(
            self.path.glob("task_*"), key=lambda x: int(x.stem.split("_")[-1])
        ):
            train = FileDataset(task_dir / "train")
            test = FileDataset(task_dir / "test")
            if self.accumulate_test_set:
                self.test_sets.append(test)
                accumulated_test = ConcatDataset(self.test_sets)
                yield train, accumulated_test
            else:
                yield train, test


class IDSprites(ContinualDataset):
    NAME = "infinite-dsprites"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 10
    N_TASKS = 1000
    IMG_SIZE = 256
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.benchmark = ContinualBenchmarkDisk(args.data_path)
        self.iter_benchmark = iter(self.benchmark)

    def get_data_loaders(self):
        """Get the data loaders for the benchmark."""
        self.i += self.N_CLASSES_PER_TASK
        train_dataset, test_dataset = next(self.iter_benchmark)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True,
        )
        self.train_loader = train_loader
        self.test_loaders.append(test_loader)
        return train_loader, test_loader

    @staticmethod
    def get_backbone():
        return resnet18(IDSprites.N_CLASSES_PER_TASK * IDSprites.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 5

    @staticmethod
    def get_batch_size():
        return 64

    @staticmethod
    def get_minibatch_size():
        return IDSprites.get_batch_size()
