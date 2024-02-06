# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_dataset
from torch.optim import SGD

from models.utils.continual_model import ContinualModel
from utils.args import (
    add_management_args,
    add_experiment_args,
    add_rehearsal_args,
    ArgumentParser,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Continual learning via" " Learning without Forgetting."
    )
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--alpha", type=float, default=0.5, help="Penalty weight.")
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=2,
        help="Temperature of the softmax function.",
    )
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(ContinualModel):
    NAME = "lwf"
    COMPATIBILITY = ["class-il", "task-il"]

    def __init__(self, backbone, loss, args, transform):
        super(Lwf, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.dataset = get_dataset(args)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.nc = get_dataset(args).N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones(self.nc, self.nc)).bool()

    def get_lower_triangular_row(self, n):
        """
        Generates the n-th row of a lower triangular matrix of ones of size (nc, nc).

        Args:
        n (int): The row index (0-based).
        nc (int): The number of columns (and rows) in the matrix.

        Returns:
        torch.Tensor: The n-th row of the lower triangular matrix.
        """
        if n >= self.nc or n < 0:
            raise ValueError("Row index out of bounds.")
        row = torch.zeros(self.nc).bool()
        row[: n + 1] = True
        return row

    def begin_task(self, dataset):
        self.net.eval()
        if self.current_task > 0:
            # warm-up
            opt = SGD(self.net.classifier.parameters(), lr=self.args.lr)
            for epoch in range(self.args.n_epochs):
                for i, data in enumerate(dataset.train_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)
                    opt.zero_grad()
                    with torch.no_grad():
                        feats = self.net(inputs, returnt="features")
                    mask = (
                        self.eye[(self.current_task + 1) * self.cpt - 1]
                        ^ self.eye[self.current_task * self.cpt - 1]
                    ).to(self.device)
                    # mask = self.get_lower_triangular_row(
                    #    (self.current_task + 1) * self.cpt - 1
                    # ) ^ self.get_lower_triangular_row(self.current_task * self.cpt - 1)
                    outputs = self.net.classifier(feats)[:, mask]
                    loss = self.loss(outputs, labels - self.current_task * self.cpt)
                    loss.backward()
                    opt.step()

            logits = []
            with torch.no_grad():
                for i in range(
                    0, dataset.train_loader.dataset.data.shape[0], self.args.batch_size
                ):
                    inputs = torch.stack(
                        [
                            dataset.train_loader.dataset.__getitem__(j)[2]
                            for j in range(
                                i,
                                min(
                                    i + self.args.batch_size,
                                    len(dataset.train_loader.dataset),
                                ),
                            )
                        ]
                    )
                    log = self.net(inputs.to(self.device)).cpu()
                    logits.append(log)
            setattr(dataset.train_loader.dataset, "logits", torch.cat(logits))
        self.net.train()

        self.current_task += 1

    def observe(self, inputs, labels, not_aug_inputs, logits=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)

        mask = self.eye[self.current_task * self.cpt - 1].to(self.device)
        # mask = self.get_lower_triangular_row(self.current_task * self.cpt - 1)
        loss = self.loss(outputs[:, mask], labels)
        if logits is not None:
            mask = self.eye[(self.current_task - 1) * self.cpt - 1].to(self.device)
            # mask = self.get_lower_triangular_row((self.current_task - 1) * self.cpt - 1)
            loss += self.args.alpha * modified_kl_div(
                smooth(self.soft(logits[:, mask]).to(self.device), 2, 1),
                smooth(self.soft(outputs[:, mask]), 2, 1),
            )

        loss.backward()
        self.opt.step()

        return loss.item()
