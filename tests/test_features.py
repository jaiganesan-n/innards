"""The feature-analyzer registry, detection from configs, and the Analysis-tab sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import load_fixture_config
from innards.features import KNOWN_FEATURES, analyzers, detect, feature_status, get
from innards.features.kv import explain_gap


def names(config: dict[str, Any]) -> set[str]:
    return {a.name for a, _ in detect(config)}


def test_registry() -> None:
    assert {a.name for a in analyzers()} == {
        "kv",
        "gated_deltanet",
        "gated_attention",
        "moe",
        "mhc",
        "csa_hca",
        "dsa",
        "mla",
    }
    assert [a.light for a in analyzers()][:4] == [True, True, True, True]  # light analyzers first
    assert {f.id for f in KNOWN_FEATURES} == {"mamba2", "short_conv", "mtp"}
    with pytest.raises(KeyError, match="No analyzer"):
        get("nope")


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("qwen3-0.6b", {"kv"}),
        ("qwen3-next-80b-a3b-instruct", {"kv", "gated_deltanet", "gated_attention", "moe"}),
        ("deepseek-v4-flash", {"kv", "mhc", "moe", "csa_hca"}),
        ("deepseek-v3.2", {"kv", "mla", "dsa", "moe"}),
        ("deepseek-v2-lite", {"kv", "mla", "moe"}),
        ("nemotron-3-super-120b-a12b", {"kv", "moe"}),
        ("gpt2", {"kv"}),
    ],
)
def test_detection_from_configs(fixture: str, expected: set[str]) -> None:
    assert names(load_fixture_config(fixture)) == expected


def test_feature_status() -> None:
    status = feature_status(load_fixture_config("nemotron-3-super-120b-a12b"), active=["kv", "moe"])
    assert status["moe"]["status"] == "analyzed"
    assert status["mhc"]["status"] == "absent"
    assert status["mamba2"]["status"] == "not_analyzed"
    assert status["mtp"]["status"] == "not_run"
    assert feature_status(load_fixture_config("deepseek-v4-flash"))["mhc"]["status"] == "detected"


def test_gap_explanations() -> None:
    notes = explain_gap({"sliding": 200, "pending": 50, "compressed": 10}, {"sliding": 100, "compressed": 10}, None)
    assert any("compression window" in n for n in notes)
    assert any("separate K and V" in n for n in notes)
    assert explain_gap({"global": 700}, {"global": 100}, "deepseek_v32")[0].startswith("transformers caches")


@pytest.mark.hf
@pytest.mark.parametrize("kind", ["qwen3_5", "nemotron_h", "deepseek_v4", "deepseek_v32"])
def test_conversation_sections_are_json_charts(kind: str, tmp_path: Path) -> None:
    import innards
    from innards.store import RunStore
    from innards.view.analysis import analysis
    from replicas import chat_ids, replica, tokenizer

    model = replica(kind)
    watch = innards.observe(model, tokenizer(), run="r", root=str(tmp_path))
    history: list[dict[str, str]] = []
    for message in ("remember my name is priya", "what is my name"):
        history.append({"role": "user", "content": message})
        out = model.generate(**chat_ids(history), max_new_tokens=4, min_new_tokens=4, do_sample=False)
        history.append({"role": "assistant", "content": tokenizer().decode(out[0, -4:], skip_special_tokens=True)})
    conversation = watch.conversation
    watch.close()
    data = analysis(RunStore(tmp_path), "r", conversation)
    assert data["sections"]
    for section in data["sections"]:
        assert section["findings"], section["id"]
        for chart in section["charts"]:
            assert chart["type"] in {"line", "bar", "heatmap"}
    json.dumps(data)
