"""KV-cache calculator: architectural KV bytes from a model config (pure Python).

>>> from innards.kv import predict
>>> p = predict({"model_type": "gpt2", "n_layer": 12, "n_head": 12, "n_embd": 768}, 1024)
>>> p.bytes_per_token
73728
"""

from innards.kv.calculator import KVPrediction, layer_bytes, predict, resolve_spec
from innards.kv.config import ConfigSource, LayerKV, ModelSpec, load_config, spec_from_config
from innards.kv.dtypes import DTYPE_BYTES, dtype_bytes, normalize_dtype

__all__ = [
    "DTYPE_BYTES",
    "ConfigSource",
    "KVPrediction",
    "LayerKV",
    "ModelSpec",
    "dtype_bytes",
    "layer_bytes",
    "load_config",
    "normalize_dtype",
    "predict",
    "resolve_spec",
    "spec_from_config",
]
