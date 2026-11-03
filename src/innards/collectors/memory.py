"""Memory collectors: process RSS (psutil) and the weights / KV / activations split."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from innards.schema import MemoryInfo

__all__ = ["rss_sampler", "split_memory", "tensor_bytes", "weights_bytes"]

log = logging.getLogger("innards.collectors")


def rss_sampler() -> Callable[[], int | None] | None:
    """Return a callable reading this process's RSS, or ``None`` if psutil is not installed."""
    try:
        import psutil
    except ImportError:
        log.debug("psutil not installed; CPU memory will be reported as null")
        return None
    process = psutil.Process()

    def sample() -> int | None:
        return int(process.memory_info().rss)

    return sample


def tensor_bytes(tensor: Any) -> int:
    """Bytes a tensor really holds on its device (metadata only: no values are read).

    Quantized weights can be tensor subclasses that report their logical dtype, so
    ``numel * element_size`` would count them at the unpacked size. Their inner tensors (packed data
    and scales) are counted instead.
    """
    if hasattr(type(tensor), "__tensor_flatten__"):  # plain tensors and Parameters do not have it
        try:
            names, _ctx = tensor.__tensor_flatten__()
            return sum(tensor_bytes(getattr(tensor, name)) for name in names)
        except Exception:
            log.debug("could not unwrap %s; counting its logical size", type(tensor).__name__, exc_info=True)
    return int(tensor.numel() * tensor.element_size())


def weights_bytes(model: Any) -> int:
    """Bytes of a model's parameters and buffers, each tensor counted once."""
    seen: set[int] = set()
    total = 0
    for tensor in [*model.parameters(), *model.buffers()]:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor_bytes(tensor)
    return total


def split_memory(
    *,
    weights_bytes: int | None,
    kv_bytes: int | None,
    peak_bytes: int | None,
    source: str | None,
    allocator_only: bool,
    allocated_bytes: int | None = None,
    budget_bytes: int | None = None,
) -> MemoryInfo:
    """Build MemoryInfo. ``activations = peak - weights - kv`` only when the peak counts tensors alone.

    Process RSS also counts the Python runtime, libraries and allocator caches, so on
    CPU the activation share is left ``None`` rather than guessed.
    """
    activations: int | None = None
    if allocator_only and peak_bytes is not None and weights_bytes is not None and kv_bytes is not None:
        activations = max(peak_bytes - weights_bytes - kv_bytes, 0)
    return MemoryInfo(
        weights_bytes=weights_bytes,
        kv_bytes=kv_bytes,
        activations_bytes=activations,
        peak_bytes=peak_bytes,
        allocated_bytes=allocated_bytes,
        budget_bytes=budget_bytes,
        source=source,
    )
