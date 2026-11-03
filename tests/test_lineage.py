from __future__ import annotations

import json

import pytest

from conftest import CONFIGS, load_fixture_config
from innards.kv import spec_from_config
from innards.lineage import LineageEntry, chain, explain, families, get

REQUIRED = ("predecessor", "change", "problem_solved", "trade_off")


def test_every_family_has_a_complete_entry() -> None:
    assert {"mha", "mqa", "gqa", "mla", "sliding_window", "hybrid", "ssm", "fp8_kv"} <= set(families())
    for family in families():
        entry = get(family)
        for key in REQUIRED[1:]:
            assert getattr(entry, key), f"{family}.{key} is empty"
        assert entry.reference.get("url", "").startswith("https://")


def test_predecessors_resolve() -> None:
    for family in families():
        predecessor = get(family).predecessor
        if predecessor is not None:
            assert predecessor in families()


@pytest.mark.parametrize(
    ("alias", "family"),
    [("MHA", "mha"), ("grouped-query", "gqa"), ("swa", "sliding_window"), ("fp8", "fp8_kv"), ("mamba", "ssm")],
)
def test_aliases(alias: str, family: str) -> None:
    assert explain(alias).family == family


def test_chain_walks_back_to_mha() -> None:
    assert [e.family for e in chain("mla")] == ["mla", "gqa", "mha"]


@pytest.mark.parametrize(
    ("name", "family", "detail"),
    [
        ("gpt2", "mha", "12 KV heads, one per query head"),
        ("llama-3.2-3b", "gqa", "8 KV heads shared by 24 query heads (3 per group)"),
        ("deepseek-v2-lite", "mla", "512-dim latent + 64-dim RoPE key"),
        ("gemma-3-1b-it", "sliding_window", "22 of 26 layers keep the last 511 tokens; 4 global layers grow"),
        ("granite-4.0-h-tiny", "hybrid", "4 of 40 layers use attention; 36 carry fixed recurrent state"),
    ],
)
def test_explain_model(name: str, family: str, detail: str) -> None:
    entry = explain(CONFIGS / f"{name}.json")
    assert isinstance(entry, LineageEntry)
    assert entry.family == family
    assert entry.detail is not None
    assert detail in entry.detail
    assert entry.summary()["change"] == entry.detail


def test_explain_accepts_spec_and_dict() -> None:
    cfg = load_fixture_config("qwen3-0.6b")
    assert explain(cfg).family == "gqa"
    assert explain(spec_from_config(cfg, model_id="Qwen/Qwen3-0.6B")).model_id == "Qwen/Qwen3-0.6B"


def test_str_and_dict() -> None:
    entry = explain(CONFIGS / "deepseek-v2-lite.json")
    text = str(entry)
    assert "Multi-head latent attention" in text
    assert "Trade-off" in text
    json.dumps(entry.to_dict())


def test_unknown_family() -> None:
    with pytest.raises(KeyError):
        get("transformer-xl")
