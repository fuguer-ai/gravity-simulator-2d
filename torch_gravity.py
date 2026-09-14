"""Optional memory-bounded exact force reference in native PyTorch.

Runs on the input tensor's CPU/CUDA device. No N-by-N allocation: scratch is
O(block_size**2), work remains O(N**2). This is a diagnostic/reference backend;
the app's fast CUDA path uses the fused NVIDIA Warp kernel.
"""
import math


def accelerations(positions, masses, G=1.0, softening=3.0, block_size=512):
    import torch
    if positions.ndim != 2 or positions.shape[1] != 2 or masses.shape != positions.shape[:1]:
        raise ValueError("Expected positions [N,2] and masses [N]")
    if positions.device != masses.device or positions.dtype != masses.dtype:
        raise ValueError("Matching devices and dtypes required")
    if positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("FP32 or FP64 required")
    if not math.isfinite(G) or not math.isfinite(softening) or softening <= 0:
        raise ValueError("Finite G and positive softening required")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("Positive integer block_size required")
    if not torch.isfinite(positions).all() or not torch.isfinite(masses).all() or (masses < 0).any():
        raise ValueError("Finite positions and nonnegative finite masses required")
    with torch.no_grad():
        n = len(positions)
        result = torch.zeros_like(positions)
        for i in range(0, n, block_size):
            p = positions[i:i+block_size]
            for j in range(0, n, block_size):
                d = positions[j:j+block_size][None] - p[:, None]
                r2 = (d*d).sum(-1) + softening*softening
                if i == j:
                    r2.fill_diagonal_(float('inf'))
                weight = masses[j:j+block_size][None] * r2.rsqrt().pow(3)
                result[i:i+len(p)] += G*(d*weight[..., None]).sum(1)
        return result
