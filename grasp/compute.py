"""Explicit policy device selection; physics workers remain on CPU."""
import torch


def resolve_device(requested="cuda:0"):
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu, cuda, or cuda:N")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {requested}, but CUDA is unavailable. Install requirements-lock.txt or explicitly use --device cpu.")
        device = torch.device("cuda", index)
        # Fail early for incompatible driver/runtime/kernel combinations.
        torch.ones(1, device=device).square().sum().item()
    return device


def compute_info(device):
    return dict(device=str(device), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
