# Copyright 2022-present, Sebastian Dziadzio.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from argparse import Namespace

import torch.nn.functional as F
from codis.data.continual_benchmark import ContinualBenchmark
from codis.data.infinite_dsprites import InfiniteDSprites, Latents
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from backbone.ResNet18 import resnet18
from datasets.utils.continual_dataset import ContinualDataset


class IDSprites(ContinualDataset):
    NAME = "infinite-dsprites"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 10
    N_TASKS = 200
    IMG_SIZE = 224

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.benchmark = None

    def initialize_benchmark(self):
        shapes = [
            InfiniteDSprites().generate_shape()
            for _ in range(IDSprites.N_TASKS * IDSprites.N_CLASSES_PER_TASK)
        ]
        exemplars = self.generate_canonical_images(shapes, IDSprites.IMG_SIZE)
        cfg = {
            "dataset": {
                "factor_resolution": 8,
                "img_size": IDSprites.IMG_SIZE,
                "shapes_per_task": IDSprites.N_CLASSES_PER_TASK,
                "tasks": IDSprites.N_TASKS,
                "train_split": 0.98,
                "val_split": 0.01,
                "test_split": 0.01,
            }
        }
        cfg = OmegaConf.create(cfg)
        self.benchmark = ContinualBenchmark(
            cfg, shapes, exemplars, accumulate_test=False
        )

    def generate_canonical_images(self, shapes, img_size: int):
        """Generate a batch of exemplars for training and visualization."""
        dataset = InfiniteDSprites(
            img_size=img_size,
        )
        return [
            dataset.draw(
                Latents(
                    color=(1.0, 1.0, 1.0),
                    shape=shape,
                    shape_id=None,
                    scale=1.0,
                    orientation=0.0,
                    position_x=0.5,
                    position_y=0.5,
                )
            )
            for shape in shapes
        ]

    def get_data_loaders(self):
        """Get the data loaders for the benchmark."""
        if self.benchmark is None:
            self.initialize_benchmark()
            self.iter_benchmark = iter(self.benchmark)

        self.i += self.N_CLASSES_PER_TASK
        datasets, _ = next(self.iter_benchmark)
        train_dataset, _, test_dataset = datasets
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )
        self.train_loader = train_loader
        self.test_loaders.append(test_loader)
        return train_loader, test_loader

    @staticmethod
    def get_backbone():
        # return torchvision.models.resnet18(
        #    num_classes=IDSprites.N_CLASSES_PER_TASK * IDSprites.N_TASKS, weights=None
        # )
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
        return 256

    @staticmethod
    def get_minibatch_size():
        return IDSprites.get_batch_size()
