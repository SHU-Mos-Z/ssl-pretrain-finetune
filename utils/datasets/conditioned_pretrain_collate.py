from __future__ import annotations

import torch


def conditioned_pretrain_collate(samples: list[dict]) -> dict:
    tensor_keys = [k for k, v in samples[0].items() if isinstance(v, torch.Tensor)]
    for sample in samples[1:]:
        for key in tensor_keys:
            if sample[key].shape != samples[0][key].shape:
                raise ValueError(f"heterogeneous batch for {key}: {sample[key].shape} vs {samples[0][key].shape}")
    output = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    output["stem"] = [sample["stem"] for sample in samples]
    output["dataset_id"] = [sample["dataset_id"] for sample in samples]
    return output
