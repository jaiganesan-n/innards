"""Refresh the MoE models ``innards view`` offers, against the installed transformers.

::

    uv run python scripts/moe_models.py              # re-check every listed model, rewrite the facts
    uv run python scripts/moe_models.py --discover   # also print new candidates from the Hub, to review

The list lives in ``src/innards/view/moe_models.json``: original releases (no third-party fine-tunes,
merges or quantizations) of MoE checkpoints published at 16 or 32 bit that fit a 16 GB machine. Which
models are on it is a reviewed choice; this script only measures them. For each one it reads the config,
tokenizer config and safetensors metadata from the Hub (nothing goes into the Hugging Face cache, so the
view's list of cached models is untouched), then:

* runs the before-download checks of :func:`innards.validate.validate` for a 16 GB budget, and
* builds the model with the installed transformers' own class on the meta device (no weights, no
  remote code) and finds its MoE layers the way a trace does.

A model the installed transformers cannot build natively, or whose MoE layers Innards cannot find, is
reported and left out of the file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve().parents[1] / "src" / "innards" / "view" / "moe_models.json"
BUDGET_GB = 16.0
MAX_WEIGHTS = 15e9  # leaves room for the KV cache and headroom within 16 GB
BYTES = {"BF16": 2, "F16": 2, "F32": 4}
VENDORS = [  # organisations that publish original MoE checkpoints
    *("ibm-granite", "ibm-research", "allenai", "microsoft", "arcee-ai", "deepseek-ai", "Qwen", "mistralai", "google"),
    *("meta-llama", "facebook", "nvidia", "LiquidAI", "poolside", "ai21labs", "openai", "utter-project", "inclusionAI"),
    *("baidu", "tencent", "moonshotai", "zai-org", "MiniMaxAI", "stepfun-ai", "LGAI-EXAONE", "upstage", "jetmoe"),
    *("HuggingFaceTB", "Zyphra", "Snowflake", "databricks", "swiss-ai", "sarvamai", "kakaocorp", "naver-hyperclovax"),
    *("XiaomiMiMo", "CohereLabs", "ServiceNow-AI", "Motif-Technologies", "trillionlabs", "openbmb", "rednote-hilab"),
]
_TEST = re.compile(r"tiny-random|random|-test|test-|dummy|debug|mock|-dev$", re.I)


def _fetch(repo: str, name: str) -> Any:
    """A file of ``repo`` over HTTP, without the Hugging Face cache (None when it is not there)."""
    from huggingface_hub import get_session, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    response = get_session().get(hf_hub_url(repo, name), headers=build_hf_headers(), timeout=30)
    if response.status_code != 200:
        return None
    return response.json() if name.endswith(".json") else response.text


def _weights(info: Any) -> tuple[int | None, str | None]:
    params = dict(info.safetensors.parameters) if info.safetensors and info.safetensors.parameters else {}
    if not params or not set(params) <= set(BYTES):
        return None, None  # no safetensors metadata, or a quantized checkpoint
    size = sum(n * BYTES[d] for d, n in params.items())
    main = max(params, key=lambda d: params[d] * BYTES[d])
    return size, {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}[main]


def _moe_layers(config: dict[str, Any]) -> tuple[int, str]:
    """MoE layers Innards finds in the model transformers builds natively from ``config`` (on meta)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from innards.experts.routing import find_moe_layers

    cfg = AutoConfig.for_model(**{k: v for k, v in config.items() if k != "auto_map"})
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return len(find_moe_layers(model)), type(model).__name__


def check(model_id: str) -> dict[str, Any]:
    """Every fact the view shows for ``model_id``, measured now; raises ValueError when it does not qualify."""
    from huggingface_hub import HfApi
    from transformers import CONFIG_MAPPING

    from innards.experts.routing import routing_facts
    from innards.features.base import text_config
    from innards.validate import validate

    info = HfApi().model_info(model_id, expand=["safetensors", "createdAt", "gated", "tags"])
    config = _fetch(model_id, "config.json")
    if not isinstance(config, dict):
        raise ValueError("no config.json")
    model_type = str(config.get("model_type") or "")
    if model_type not in CONFIG_MAPPING:
        raise ValueError(f"transformers here doesn't support `{model_type}`")
    facts = routing_facts(text_config(config))
    if (facts.get("experts") or 0) < 2:
        raise ValueError("not an MoE checkpoint (no routed experts in its config)")
    size, precision = _weights(info)
    if size is None:
        raise ValueError("no 16/32-bit safetensors weights")
    if size > MAX_WEIGHTS:
        raise ValueError(f"{size / 1e9:.1f} GB of weights does not fit {BUDGET_GB:.0f} GB")
    report = validate(
        model_id, config=config, weights_bytes=size, device="cpu", budget_gb=BUDGET_GB, check_access=False
    )
    if report.blocked:
        raise ValueError("; ".join(f"{c.id}: {c.message}" for c in report.blocked))
    layers, cls = _moe_layers(config)
    if not layers:
        raise ValueError(f"Innards finds no MoE layers in {cls}")
    tokenizer = _fetch(model_id, "tokenizer_config.json")
    template = bool(isinstance(tokenizer, dict) and tokenizer.get("chat_template"))
    template = template or _fetch(model_id, "chat_template.jinja") is not None
    license_ = next((t.split(":", 1)[1] for t in info.tags or [] if t.startswith("license:")), None)
    return {
        "id": model_id,
        "model_type": model_type,
        "class": cls,
        "experts": facts["experts"],
        "top_k": facts.get("top_k"),
        "moe_layers": layers,
        "weights_gb": round(size / 1e9, 1),
        "precision": precision,
        "chat_template": template,
        "released": info.created_at.date().isoformat() if info.created_at else None,
        "license": license_,
        "gated": bool(info.gated),
    }


def discover(known: set[str]) -> list[str]:
    """Original MoE releases on the Hub, from the vendors above, that are not on the list yet."""
    from huggingface_hub import HfApi
    from transformers import CONFIG_MAPPING

    found = []
    for vendor in VENDORS:
        for m in HfApi().list_models(author=vendor, limit=1000, expand=["config", "safetensors", "tags"]):
            config = m.config or {}
            if m.id in known or _TEST.search(m.id) or config.get("model_type") not in CONFIG_MAPPING:
                continue
            if not any(str(a).endswith("ForCausalLM") for a in config.get("architectures") or []):
                continue
            bases = [t.split(":", 2)[2] for t in m.tags or [] if t.startswith("base_model:") and t.count(":") >= 2]
            if any(b.split("/")[0] != vendor for b in bases):
                continue
            size, _ = _weights(m)
            if size is not None and size <= MAX_WEIGHTS:
                found.append(m.id)
    return found


def main() -> int:
    import transformers

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--discover", action="store_true", help="also list new candidates from the Hub")
    args = parser.parse_args()
    data = json.loads(FILE.read_text(encoding="utf-8"))
    kept, dropped = [], []
    for entry in data["models"]:
        try:
            kept.append(check(entry["id"]))
            print(f"ok       {entry['id']}")
        except Exception as exc:
            dropped.append(entry["id"])
            print(f"dropped  {entry['id']}: {exc}", file=sys.stderr)
    data.update(
        transformers=transformers.__version__,
        checked=datetime.date.today().isoformat(),
        budget_gb=BUDGET_GB,
        models=sorted(sorted(kept, key=lambda m: m["id"].lower()), key=lambda m: m["released"] or "", reverse=True),
    )
    FILE.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(kept)} models checked with transformers {transformers.__version__}; {len(dropped)} dropped")
    if args.discover:
        candidates = discover({m["id"] for m in kept} | set(dropped))
        print("\nCandidates to review (16/32-bit, under 15 GB, native type; MoE not yet checked):")
        for model_id in candidates:
            print(" ", model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
