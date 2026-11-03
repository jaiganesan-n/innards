from __future__ import annotations

import json

import pytest

from innards.schema import (
    SCHEMA_VERSION,
    ConversationRecord,
    KVInfo,
    ModelInfo,
    ObserverInfo,
    ProbeHealth,
    RuntimeInfo,
    TurnRecord,
)


def make_record(**kv: object) -> TurnRecord:
    return TurnRecord(
        conversation="c-1",
        run="study",
        turn=1,
        timestamp="2026-09-26T00:00:00Z",
        model=ModelInfo(id="gpt2", architecture="mha"),
        runtime=RuntimeInfo(backend="observe"),
        kv=KVInfo(**kv),  # type: ignore[arg-type]
    )


def test_schema_version_and_order() -> None:
    data = make_record().to_dict()
    assert next(iter(data)) == "schema_version"
    assert data["schema_version"] == SCHEMA_VERSION == "1.0"


def test_missing_measurements_are_null_not_zero() -> None:
    data = make_record().to_dict()
    assert data["kv"]["measured_bytes"] is None
    assert data["memory"]["peak_bytes"] is None
    assert data["timing"]["ttft_ms"] is None
    assert data["tokens"]["cached"] is None


def test_round_trip_with_feature_payloads_and_probe_health() -> None:
    record = make_record(measured_bytes=10, per_layer_bytes=[5, 5], measured_source="hf_cache")
    record.features["gated_deltanet"] = {"summary": {"beta_mean": 0.4}, "findings": ["x"], "data": {"layers": [0]}}
    record.observer = ObserverInfo(overhead_pct=0.3, probes=[ProbeHealth(feature="gated_deltanet", hooks=36)])
    again = TurnRecord.from_json(record.to_json())
    assert again == record
    assert again.observer.probes[0].hooks == 36


def test_record_keys() -> None:
    data = make_record().to_dict()
    for key in ("run", "conversation", "turn", "timestamp", "model", "runtime", "message", "tokens", "features"):
        assert key in data
    for key in ("context", "kv", "memory", "timing", "output", "observer"):
        assert key in data
    for key in ("predicted_bytes", "measured_bytes", "measured_source", "by_layer_type", "per_layer_bytes"):
        assert key in data["kv"]


def test_record_without_conversation_is_rejected() -> None:
    data = make_record().to_dict()
    del data["conversation"]
    with pytest.raises(ValueError, match="needs `conversation`"):
        TurnRecord.from_dict(data)


def test_unknown_fields_are_ignored_within_major() -> None:
    data = make_record().to_dict()
    data["schema_version"] = "1.7"
    data["kv"]["some_future_field"] = 1
    data["brand_new_group"] = {}
    assert TurnRecord.from_dict(data).schema_version == "1.7"


def test_other_major_version_is_rejected() -> None:
    data = make_record().to_dict() | {"schema_version": "2.0"}
    with pytest.raises(ValueError, match="Unsupported TurnRecord schema"):
        TurnRecord.from_dict(data)


def test_conversation_record_round_trip() -> None:
    conv = ConversationRecord(id="c-1", model=ModelInfo(id="gpt2", features=["kv"]), run="study", turns=3)
    again = ConversationRecord.from_dict(json.loads(json.dumps(conv.to_dict())))
    assert again == conv
