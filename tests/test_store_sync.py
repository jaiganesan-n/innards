"""The local run store and S3 sync (moto)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from innards.schema import ConversationRecord, ModelInfo, RuntimeInfo, TurnRecord
from innards.store import RunStore, safe_name, valid_name


def record(conversation: str, turn: int) -> TurnRecord:
    return TurnRecord(
        conversation=conversation,
        turn=turn,
        timestamp="2026-09-29T00:00:00Z",
        run="study",
        model=ModelInfo(id="m"),
        runtime=RuntimeInfo(backend="observe"),
    )


def fill(store: RunStore, run: str = "study", conversation: str = "c1", turns: int = 2) -> None:
    store.ensure_run(run, {"python": "3.x"})
    store.write_conversation(run, ConversationRecord(id=conversation, model=ModelInfo(id="m"), run=run, turns=turns))
    for t in range(1, turns + 1):
        store.append_turn(run, conversation, record(conversation, t))


def test_names() -> None:
    assert valid_name("deltanet-study")
    assert not valid_name("../etc")
    assert not valid_name(".hidden")
    assert safe_name("DeltaNet study!") == "DeltaNet-study"
    assert safe_name("../../x") == "x"
    assert safe_name("///") == "run"


def test_round_trip_and_listing(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    fill(store)
    fill(store, conversation="c2", turns=1)
    runs = store.list_runs()
    assert [r["name"] for r in runs] == ["study"]
    assert {c["id"] for c in runs[0]["conversations"]} == {"c1", "c2"}
    assert [t.turn for t in store.turn_records("study", "c1")] == [1, 2]
    assert store.find_conversation("c2") == "study"
    assert store.read_conversation("study", "nope") is None
    assert store.turns("../x", "c1") == []
    with pytest.raises(ValueError, match="Invalid"):
        store.append_turn("../x", "c1", record("c1", 1))


def test_torn_last_line_is_skipped(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    fill(store)
    path = tmp_path / "study" / "c1" / "turns.jsonl"
    path.write_text(path.read_text() + '{"turn": 3, "conv')
    assert [t["turn"] for t in store.turns("study", "c1")] == [1, 2]


def test_opening_a_single_run_folder(tmp_path: Path) -> None:
    fill(RunStore(tmp_path))
    fill(RunStore(tmp_path), run="other")
    only = RunStore(tmp_path / "study")
    assert [r["name"] for r in only.list_runs()] == ["study"]
    assert only.read_run("other") is None
    assert len(only.turns("study", "c1")) == 2


@pytest.mark.skipif(importlib.util.find_spec("moto") is None, reason="needs moto (dev dependency)")
def test_sync_up_and_down(tmp_path: Path) -> None:
    import boto3
    from moto import mock_aws

    from innards.sync import parse_s3_url, sync_down, sync_up

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bucket")
        store = RunStore(tmp_path / "aws")
        fill(store)
        first = sync_up(store, "study", "s3://bucket/innards", client=s3)
        assert "innards/study/run.json" in first.copied
        assert "innards/study/c1/turns.jsonl" in first.copied
        again = sync_up(store, "study", "s3://bucket/innards", client=s3)
        assert again.copied == []  # nothing changed
        store.append_turn("study", "c1", record("c1", 3))
        assert sync_up(store, "study", "s3://bucket/innards", client=s3).copied == ["innards/study/c1/turns.jsonl"]

        laptop = tmp_path / "laptop"
        down = sync_down("s3://bucket/innards/study", laptop, client=s3)
        assert len(down.copied) == 3
        assert [t["turn"] for t in RunStore(laptop).turns("study", "c1")] == [1, 2, 3]
        assert sync_down("s3://bucket/innards", laptop, client=s3).copied == []  # a prefix of runs, unchanged
        assert json.loads((laptop / "study" / "run.json").read_text())["name"] == "study"
    assert parse_s3_url("s3://b/p/q") == ("b", "p/q")
    with pytest.raises(ValueError, match="Not an S3 URL"):
        parse_s3_url("/tmp/x")
