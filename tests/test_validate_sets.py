"""Before-download checks and conversation sets (no network: configs and safetensors headers are local)."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from conftest import load_fixture_config
from innards.sets import SETS, count_tokens, estimate_tokens, get_set, load_document, long_document
from innards.validate import check_loaded, checkpoint_weights, published_precision, validate


def checkpoint(folder: Path, config: dict[str, Any], tensors: dict[str, tuple[str, list[int]]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config))
    header = json.dumps({name: {"dtype": d, "shape": s, "data_offsets": [0, 0]} for name, (d, s) in tensors.items()})
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header.encode())
    return folder


def by_id(report: Any) -> dict[str, Any]:
    return {c.id: c for c in report.checks}


def test_weights_from_safetensors_headers(tmp_path: Path) -> None:
    folder = checkpoint(
        tmp_path / "m", load_fixture_config("qwen3-0.6b"), {"a": ("BF16", [1000, 1000]), "b": ("F32", [10])}
    )
    total, dtypes = checkpoint_weights(str(folder))  # type: ignore[misc]
    assert total == 1000 * 1000 * 2 + 10 * 4
    assert dtypes == {"BF16": 1_000_000, "F32": 10}


@pytest.mark.hf
def test_a_small_model_is_ready(tmp_path: Path) -> None:
    folder = checkpoint(tmp_path / "qwen3", load_fixture_config("qwen3-0.6b"), {"w": ("BF16", [750_000_000])})
    report = validate(str(folder), device="cpu", conversation_set="with_context")
    checks = by_id(report)
    assert report.ok
    assert checks["architecture"].status == "ok"
    assert checks["memory"].status == "ok"
    assert checks["memory"].numbers["weights_bytes"] == 1_500_000_000
    assert checks["access"].status == "ok"


@pytest.mark.hf
def test_over_budget_is_blocked_with_the_numbers(tmp_path: Path) -> None:
    folder = checkpoint(tmp_path / "big", load_fixture_config("qwen3-0.6b"), {"w": ("BF16", [15_000_000_000])})
    report = validate(str(folder), device="cuda")
    memory = by_id(report)["memory"]
    assert report.status == "Blocked"
    assert memory.status == "blocked"
    assert memory.message.startswith("needs 34 GB, this machine allows 16 GB")
    assert memory.fix


@pytest.mark.hf
def test_fp8_checkpoint_needs_cuda() -> None:
    config = load_fixture_config("deepseek-v4-flash")
    assert published_precision(config) == "fp4/fp8"
    report = validate(
        "deepseek-ai/DeepSeek-V4-Flash", device="mps", config=config, weights_bytes=160_000_000_000, check_access=False
    )
    precision = by_id(report)["precision"]
    assert precision.status == "blocked"
    assert "FP4/FP8" in precision.message
    assert "MPS" in precision.message


@pytest.mark.hf
def test_other_precisions_are_refused(tmp_path: Path) -> None:
    folder = checkpoint(tmp_path / "q", load_fixture_config("qwen3-0.6b"), {"w": ("BF16", [10])})
    assert by_id(validate(str(folder), device="mps", precision="fp16"))["precision"].status == "blocked"
    assert by_id(validate(str(folder), device="cpu", precision="fp32"))["precision"].status == "ok"  # lossless upcast


@pytest.mark.hf
def test_unknown_architecture_and_remote_code(tmp_path: Path) -> None:
    unknown = checkpoint(
        tmp_path / "u",
        {"model_type": "deepseek_v41_text", "num_hidden_layers": 2, "num_attention_heads": 2, "hidden_size": 8},
        {"w": ("BF16", [10])},
    )
    check = by_id(validate(str(unknown), device="cpu"))["architecture"]
    assert check.status == "blocked"
    assert "doesn't support `deepseek_v41_text`" in check.message
    assert check.fix == "Upgrade transformers, or wait for support."
    remote = checkpoint(
        tmp_path / "r",
        {
            "model_type": "custom_x",
            "auto_map": {"AutoConfig": "x.Y"},
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_size": 8,
        },
        {"w": ("BF16", [10])},
    )
    assert "trust_remote_code" in by_id(validate(str(remote), device="cpu"))["architecture"].message
    native = validate(
        "nvidia/x",
        device="cuda",
        config=load_fixture_config("nemotron-3-nano-30b-a3b"),
        weights_bytes=1,
        check_access=False,
    )
    assert by_id(native)["architecture"].status == "ok"  # nemotron_h is native; its auto_map is not needed


@pytest.mark.hf
def test_gated_repo_access() -> None:
    from huggingface_hub.utils import GatedRepoError

    class Hub:
        def auth_check(self, repo: str) -> None:
            error = GatedRepoError.__new__(GatedRepoError)  # the real constructor wants an HTTP response
            Exception.__init__(error, "gated")
            raise error

    report = validate("meta/gated", device="cpu", config=load_fixture_config("qwen3-0.6b"), weights_bytes=1, hub=Hub())
    access = by_id(report)["access"]
    assert access.status == "blocked"
    assert "https://huggingface.co/meta/gated" in (access.fix or "")


@pytest.mark.hf
def test_features_without_analyzers_are_a_warning() -> None:
    report = validate(
        "nvidia/x",
        device="cuda",
        config=load_fixture_config("nemotron-3-nano-30b-a3b"),
        weights_bytes=1,
        check_access=False,
    )
    check = by_id(report)["analyzers"]
    assert check.status == "warning"
    assert "Mamba-2" in check.message
    assert report.status == "Loads with limits"


def test_loaded_memory_check() -> None:
    class Model:
        def parameters(self) -> list[Any]:
            return [Tensor(4_500_000_000)]  # 18 GB in fp32: over the 16 GB default

        def buffers(self) -> list[Any]:
            return []

    class Tensor:
        def __init__(self, n: int) -> None:
            self.n = n

        def numel(self) -> int:
            return self.n

        def element_size(self) -> int:
            return 4

    assert check_loaded(Model()).status == "blocked"
    assert check_loaded(Model(), budget_gb=20).status == "ok"


# --------------------------------------------------------------------- sets


def test_sets_and_their_token_counts() -> None:
    assert set(SETS) == {"basic_qa", "with_context", "input_docs", "long_document"}
    counts = count_tokens(SETS["with_context"], reply_tokens=10)
    assert counts["per_turn"] == sorted(counts["per_turn"])
    assert counts["max_context"] == counts["per_turn"][-1] + 10
    with pytest.raises(ValueError, match="needs a document"):
        SETS["input_docs"].messages()


def test_long_document_plants_three_facts() -> None:
    text = long_document()
    for fact in ("Aster", "Marigold", "Ilse Varga"):
        assert fact in text
    assert text.index("Aster") < text.index("Marigold") < text.index("Ilse Varga")
    assert estimate_tokens(text) > 7000
    assert "Aster" in SETS["long_document"].messages()[0]


def test_documents(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Notes\nThe budget is 42.")
    chosen = get_set("input_docs", load_document(tmp_path / "notes.md"), "notes.md")
    assert chosen.messages()[0].endswith("The budget is 42.")
    assert chosen.document_name == "notes.md"
    with pytest.raises(ValueError, match=r"\.txt, \.md or \.pdf"):
        load_document(b"x", "data.csv")
    assert load_document("héllo".encode("latin-1"), "a.txt") == "héllo"


@pytest.mark.skipif(importlib.util.find_spec("pypdf") is None, reason="needs the [docs] extra (pypdf)")
def test_pdf_without_text_is_refused() -> None:
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ValueError, match="no text layer"):
        load_document(buf.getvalue(), "scan.pdf")


@pytest.mark.hf
def test_set_over_the_context_window_is_blocked(tmp_path: Path) -> None:
    config = load_fixture_config("gpt2")  # 1,024 tokens
    report = validate(
        "gpt2", device="cpu", config=config, weights_bytes=1, conversation_set="long_document", check_access=False
    )
    assert by_id(report)["context"].status == "blocked"
