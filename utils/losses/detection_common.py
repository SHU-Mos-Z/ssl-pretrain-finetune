"""Shared helpers for dense detection criteria."""

from __future__ import annotations

import torch
import torch.distributed as dist


def distributed_normalizer(local_count: int | torch.Tensor, device: torch.device) -> torch.Tensor:
    count = torch.as_tensor(local_count, dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        count /= dist.get_world_size()
    return count.clamp(min=1.0)


def target_to_device(target: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }
