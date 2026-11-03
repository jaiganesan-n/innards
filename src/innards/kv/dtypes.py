"""KV element types and their sizes in bytes."""

from __future__ import annotations

__all__ = ["DTYPE_BYTES", "dtype_bytes", "normalize_dtype"]

#: Canonical dtype name -> bytes per element.
DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8": 1,
}

_ALIASES: dict[str, str] = {
    "fp32": "float32",
    "f32": "float32",
    "float": "float32",
    "single": "float32",
    "fp16": "float16",
    "f16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "fp8": "float8",
    "f8": "float8",
    "float8_e4m3": "float8",
    "float8_e4m3fn": "float8",
    "float8_e4m3fnuz": "float8",
    "float8_e5m2": "float8",
    "float8_e5m2fnuz": "float8",
    "fp8_e4m3": "float8",
    "fp8_e5m2": "float8",
    "e4m3": "float8",
    "e5m2": "float8",
}


def normalize_dtype(dtype: object) -> str:
    """Return the canonical name (``float32``, ``float16``, ``bfloat16`` or ``float8``).

    Accepts strings such as ``"fp16"`` or ``"torch.bfloat16"`` and objects whose
    ``str()`` gives one (for example a ``torch.dtype``).
    """
    name = str(dtype).strip().lower()
    if name.startswith("torch."):
        name = name[len("torch.") :]
    name = _ALIASES.get(name, name)
    if name not in DTYPE_BYTES:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"Unsupported KV dtype {dtype!r}; use one of: {supported} (or fp32/fp16/bf16/fp8)")
    return name


def dtype_bytes(dtype: object) -> int:
    """Bytes per element for a dtype name or alias."""
    return DTYPE_BYTES[normalize_dtype(dtype)]
