import torch
from einops import einsum, reduce, repeat
from jaxtyping import Float
from torch import Tensor

from ..types import BatchedExample


def inverse_normalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean


def normalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean) / std


def apply_normalize_shim(
    batch: BatchedExample,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> BatchedExample:
    """
    Applies normalization to batch["context"]["image"].

    Returns:
        BatchedExample:
            The input batch with its 'context.image' field normalized.

    Example:
        >>> batch = {"target": BatchedViews(...), "context": BatchedViews(image=..., extrinsics=..., ), "scene": list[str]}
        >>> normalized_batch = apply_normalize_shim(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    """
    batch["context"]["image"] = normalize_image(batch["context"]["image"], mean, std)
    return batch
