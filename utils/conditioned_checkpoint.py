from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel


def save_training_checkpoint(path, model, optimizer, scheduler, scaler, epoch, config, args):
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save({
        "model": inner.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "model_config": asdict(config), "args": vars(args),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
    }, path)


def resume_training_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    inner.load_state_dict(state.get("model", state))
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None and name in state:
            obj.load_state_dict(state[name])
    rng = state.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(state.get("epoch", 0)), state
