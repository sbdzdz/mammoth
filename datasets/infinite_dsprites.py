# Copyright 2022-present, Sebastian Dziadzio.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18

from datasets.utils.continual_dataset import ContinualDataset


class InfinitedSprites(ContinualDataset):
    NAME = "infinite-dsprites"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 10
    N_TASKS = 200

    def get_data_loaders(self):
        raise NotImplementedError

    @staticmethod
    def get_backbone():
        return resnet18(InfinitedSprites.N_CLASSES_PER_TASK * InfinitedSprites.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        return transforms.ToPILImage()

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 256

    @staticmethod
    def get_minibatch_size():
        return InfinitedSprites.get_batch_size()
