from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import CONFIGS
from innards.cli import main, parse_bytes


def run_cli(*args: str, stdin: str = "", timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "innards", *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_parse_bytes() -> None:
    assert parse_bytes("16GiB") == 16 * 2**30
    assert parse_bytes("8 GB") == 8 * 10**9
    assert parse_bytes("1024") == 1024
    with pytest.raises(Exception, match="not a size"):
        parse_bytes("lots")


def test_predict_text(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["predict", "--model", str(CONFIGS / "llama-3.2-3b.json"), "--context", "32768"]) == 0
    out = capsys.readouterr().out
    assert "114,688 B/token" in out
    assert "lineage: gqa ← mha" in out


def test_predict_json_with_memory(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["predict", "--model", str(CONFIGS / "deepseek-v2-lite.json"), "--context", "4096"]
    assert main([*args, "--kv-dtype", "fp8", "--memory", "1GiB", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["bytes_per_token"] == 27 * 576
    assert data["capacity_tokens"] == 2**30 // (27 * 576)


def test_predict_sliding(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["predict", "--model", str(CONFIGS / "gemma-3-1b-it.json"), "--context", "8192"]) == 0
    assert "22 local layers stop at 511 tokens" in capsys.readouterr().out


def test_explain(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain", "gqa"]) == 0
    assert "Grouped-query attention" in capsys.readouterr().out
    assert main(["explain", str(CONFIGS / "granite-4.0-h-tiny.json"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["family"] == "hybrid"


def test_bad_dtype_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["predict", "--model", str(CONFIGS / "gpt2.json"), "--context", "1", "--kv-dtype", "int3"]) == 2
    assert "Unsupported KV dtype" in capsys.readouterr().err


def test_version() -> None:
    result = run_cli("--version")
    assert result.returncode == 0
    assert result.stdout.startswith("innards ")


def test_lineup_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lineup"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-0.6B  →  Qwen/Qwen3.5-0.8B" in out
    assert "ibm-granite/granite-3.1-3b-a800m-instruct  →  ibm-granite/granite-swash-3b-a600m" in out
    assert "Shared expert" in out
    assert main(["lineup", "--json"]) == 0
    pairs = json.loads(capsys.readouterr().out)["pairs"]
    assert [p["where"] for p in pairs] == ["local", "local"]


def test_predict_compressed_attention(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["predict", "--model", str(CONFIGS / "deepseek-v4-flash.json"), "--context", "131072", "--kv-dtype", "bf16"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "compressed" in out
    assert "CSA layers" in out
