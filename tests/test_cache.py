"""Cache inspector on fake cache objects (no torch needed), plus the no-device-read rule."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from innards.observer.cache import measure_snapshot, snapshot_cache

SRC = Path(__file__).parents[1] / "src" / "innards"


@dataclass
class DType:
    itemsize: int


class FakeTensor:
    def __init__(self, *shape: int, itemsize: int = 2) -> None:
        self.shape = shape
        self.dtype = DType(itemsize)

    def element_size(self) -> int:
        return self.dtype.itemsize


class Layer:
    is_sliding = False

    def __init__(self, seq: int, heads: int = 2, dim: int = 8) -> None:
        self.keys = FakeTensor(1, heads, seq, dim)
        self.values = FakeTensor(1, heads, seq, dim)
        self.is_initialized = True


class SlidingLayer(Layer):
    is_sliding = True

    def __init__(self, seq: int) -> None:
        super().__init__(seq)
        self.cumulative_length = 100
        self._sliding_window_tensor = FakeTensor(1)


class LinearLayer:
    def __init__(self) -> None:
        self.conv_states = {0: FakeTensor(1, 32, 4)}
        self.recurrent_states = {0: FakeTensor(1, 4, 8, 8)}
        self.is_conv_states_initialized = {0: True}


class Cache:
    def __init__(self, layers: list[Any]) -> None:
        self.layers = layers


def test_layers_api() -> None:
    cache = Cache([Layer(10), SlidingLayer(7), LinearLayer(), Layer(10)])
    snap = snapshot_cache(cache)
    assert snap is not None
    assert snap.layout == "layers"
    assert [layer.kind for layer in snap.layers] == ["global", "sliding", "recurrent", "global"]
    m = measure_snapshot(snap)
    kv10 = 2 * (1 * 2 * 10 * 8) * 2
    kv7 = 2 * (1 * 2 * 7 * 8) * 2
    state = (32 * 4 + 4 * 8 * 8) * 2
    assert m.per_layer_bytes == (kv10, kv7, state, kv10)
    assert m.by_layer_type == {"global": 2 * kv10, "sliding": kv7, "recurrent": state}
    assert m.layer_tokens == (10, 7, None, 10)
    assert m.total_bytes == 2 * kv10 + kv7 + state


class AddressedTensor(FakeTensor):
    """A fake tensor with a storage address, so aliases can be told apart from copies."""

    _next = 0

    def __init__(self, *shape: int, itemsize: int = 2) -> None:
        super().__init__(*shape, itemsize=itemsize)
        AddressedTensor._next += 4096
        self._address = AddressedTensor._next

    def data_ptr(self) -> int:
        return self._address


class CompressedLayer:
    """DeepSeek-V4 CSA/HCA-style layer: shared K=V window plus dict-held compressed entries."""

    is_sliding = True

    def __init__(self) -> None:
        self.dtype = None
        self.device = None
        self.keys = AddressedTensor(1, 1, 7, 8)
        self.values = self.keys  # shared-KV MQA: one tensor, two names
        self.compressed_kv: dict[str, Any] = {"main": AddressedTensor(1, 3, 8), "indexer": None}
        self.buffer_kv = {"main": AddressedTensor(1, 2, 16)}
        self.entry_count = {"main": 3}
        self.cumulative_length = 100


def test_aliases_counted_once_and_dict_held_tensors_counted() -> None:
    layer = CompressedLayer()
    m = measure_snapshot(snapshot_cache(Cache([layer, Layer(4)])))  # type: ignore[arg-type]
    compressed = (7 * 8 + 3 * 8 + 2 * 16) * 2
    assert m.per_layer_bytes[0] == compressed
    assert m.by_layer_type["sliding"] == 7 * 8 * 2  # the shared K=V window, counted once
    assert m.by_layer_type["compressed"] == 3 * 8 * 2  # pooled entries
    assert m.by_layer_type["pending"] == 2 * 16 * 2  # tokens waiting to fill the next window
    assert m.layer_tokens[0] == 7  # read from K, not from a compressed buffer


def test_legacy_lists_and_hybrid_placeholders() -> None:
    class ListCache:
        def __init__(self) -> None:
            self.key_cache = [FakeTensor(1, 2, 5, 4), FakeTensor(1, 0)]
            self.value_cache = [FakeTensor(1, 2, 5, 4), FakeTensor(1, 0)]
            self.conv_states = [FakeTensor(1, 0), FakeTensor(1, 16, 4)]
            self.ssm_states = [FakeTensor(1, 0), FakeTensor(1, 2, 4, 4)]

    snap = snapshot_cache(ListCache())
    assert snap is not None
    assert snap.layout == "key_value_lists"
    assert [layer.kind for layer in snap.layers] == ["global", "recurrent"]


def test_legacy_tuples() -> None:
    cache = ((FakeTensor(1, 2, 3, 4), FakeTensor(1, 2, 3, 4)),)
    snap = snapshot_cache(cache)
    assert snap is not None
    assert snap.layout == "legacy_tuples"
    assert measure_snapshot(snap).total_bytes == 2 * 24 * 2


def test_none_cache() -> None:
    assert snapshot_cache(None) is None


def test_measurement_modules_never_read_tensor_values() -> None:
    """Static rule: no .cpu(), .item(), .tolist(), .numpy() or .to() in measurement code."""
    forbidden = {"cpu", "item", "tolist", "numpy", "to", "synchronize"}
    for rel in ("observer/cache.py", "collectors/memory.py", "collectors/timing.py", "kv/calculator.py"):
        tree = ast.parse((SRC / rel).read_text())
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not (calls & forbidden), f"{rel} calls {calls & forbidden}"
