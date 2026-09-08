"""Distributed batches that never mix incompatible datasets."""

from __future__ import annotations

import math
import random

import torch.distributed as dist
from torch.utils.data import ConcatDataset, Sampler


class HomogeneousDistributedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: ConcatDataset, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _all_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches = []
        offset = 0
        for child in self.dataset.datasets:
            indices = list(range(offset, offset + len(child)))
            if self.shuffle:
                rng.shuffle(indices)
            usable = len(indices) - len(indices) % self.batch_size
            batches.extend(indices[i:i + self.batch_size] for i in range(0, usable, self.batch_size))
            offset += len(child)
        if self.shuffle:
            rng.shuffle(batches)
        usable = len(batches) - len(batches) % self.world_size
        return batches[:usable]

    def __iter__(self):
        return iter(self._all_batches()[self.rank::self.world_size])

    def __len__(self) -> int:
        total = sum(len(d) // self.batch_size for d in self.dataset.datasets)
        return math.floor(total / self.world_size)
