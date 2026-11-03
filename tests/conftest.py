"""Shared fixtures and markers.

Markers:
    hf       needs the [hf] extra (torch + transformers); skipped when it is missing.
    view     needs the [view] extra (fastapi + uvicorn); skipped when it is missing.
    network  needs the Hugging Face Hub (real checkpoints download once into the HF cache);
             skipped when it is unreachable, unless INNARDS_REQUIRE_NETWORK=1, in which
             case the test fails instead.
"""

from __future__ import annotations

import functools
import gc
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CONFIGS = FIXTURES / "configs"


def load_fixture_config(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((CONFIGS / f"{name}.json").read_text())
    return data


@functools.cache
def hf_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


@functools.cache
def view_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("fastapi", "uvicorn"))


@functools.cache
def hub_reachable() -> bool:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        import httpx

        response = httpx.head("https://huggingface.co/api/models/openai-community/gpt2", timeout=5.0)
        return response.status_code < 500
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    require_network = os.environ.get("INNARDS_REQUIRE_NETWORK") == "1"
    for item in items:
        if "hf" in item.keywords and not hf_available():
            item.add_marker(pytest.mark.skip(reason="needs the [hf] extra (torch, transformers)"))
        if "view" in item.keywords and not view_available():
            item.add_marker(pytest.mark.skip(reason="needs the [view] extra (fastapi, uvicorn)"))
        if "network" in item.keywords and not require_network and not hub_reachable():
            item.add_marker(pytest.mark.skip(reason="Hugging Face Hub is unreachable"))


@pytest.fixture(autouse=True)
def _release_models(request: pytest.FixtureRequest) -> Any:
    """Free a real checkpoint as soon as its test ends: hooks hold reference cycles, and CI runners
    have little memory left for the README examples, which load their own models in a subprocess."""
    yield
    if "network" in request.keywords:
        gc.collect()


@pytest.fixture
def fixture_config() -> Any:
    return load_fixture_config


MODELS = {
    "gpt2": "openai-community/gpt2",  # MHA, no chat template, 1,024-token context
    "qwen3": "Qwen/Qwen3-0.6B",  # GQA; ships sampling defaults that Innards must override with greedy
    "gemma3": "unsloth/gemma-3-270m-it",  # sliding window (512); byte-identical to google/gemma-3-270m-it, ungated
    "lfm2": "LiquidAI/LFM2-350M",  # short-convolution hybrid
    "qwen3_5": "Qwen/Qwen3.5-0.8B",  # Gated DeltaNet hybrid
}
"""Real checkpoints for the [hf] tests, by the architecture each one covers."""


@pytest.fixture(scope="session")
def models() -> dict[str, str]:
    return dict(MODELS)
