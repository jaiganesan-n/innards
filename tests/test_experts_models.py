"""Expert routing on small original MoE checkpoints that fit a 16 GB machine at 16 bit.

``ibm-granite/granite-3.1-1b-a400m-instruct`` (32 experts, top-8, native transformers) runs with the
network tests. ``facebook/MobileMoE-S-Base`` ships its own modeling code, so it runs only when you
opt in with ``INNARDS_REMOTE_CODE_TESTS=1`` (it loads with its own code at a pinned revision). That code
targets transformers 4.57, so on transformers 5 the test is an expected failure until MobileMoE updates it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.hf, pytest.mark.network, pytest.mark.slow]

PROMPTS = {
    "math": ["The derivative of x squared is", "Solve for x: 3x + 5 = 20. First,"],
    "code": ["def fibonacci(n):\n    if n <", "import numpy as np\narr = np.zeros("],
}
MOBILEMOE = ("facebook/MobileMoE-S-Base", "23b6f610d688b3448705d2027ae716da0bb40d72")


def run(model: Any, tok: Any, trace: Any = None) -> list[list[int]]:
    import torch

    outputs = []
    for topic, prompts in PROMPTS.items():
        for prompt in prompts:
            inputs = tok(prompt, return_tensors="pt")
            scope = trace.topic(topic) if trace is not None else None
            with scope or torch.no_grad(), torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False, temperature=None, top_p=None)
            outputs.append(out[0].tolist())
    return outputs


def check(model: Any, tok: Any, tmp_path: Path, source: str) -> None:
    import innards

    plain = run(model, tok)
    with innards.experts.trace(model, tok, run="real", root=str(tmp_path)) as trace:
        observed = run(model, tok, trace)
        experts = trace.map()
        routing = trace.tokens()
        assert {m.source for m in trace.layers} == {source}
        assert trace.errors == 0
        assert all(m.missed == 0 for m in trace.layers)
    assert observed == plain
    for topic in PROMPTS:
        assert experts.tokens(topic) > 0
        for layer, row in zip(experts.layers, experts.counts(topic), strict=True):
            top_k = experts.layer_info[experts.layers.index(layer)]["top_k"]
            assert sum(row) == experts.tokens(topic, layer) * top_k
        chosen = experts.select(topic, coverage=0.9)
        assert 0 < chosen.kept < chosen.total
    assert experts.specialists("math")
    assert len(routing.sequences()) == 4
    assert any(t.strip() for t in routing.sequence(0)["text"])


def test_granite_moe(tmp_path: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "ibm-granite/granite-3.1-1b-a400m-instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
    check(model, tok, tmp_path, "experts_input")


@pytest.mark.skipif(os.environ.get("INNARDS_REMOTE_CODE_TESTS") != "1", reason="set INNARDS_REMOTE_CODE_TESTS=1")
@pytest.mark.xfail(
    raises=TypeError,
    strict=True,  # passes the day its code runs on transformers 5: then drop this marker
    reason="MobileMoE's published code targets transformers 4.57.6 (its model card); its forward calls "
    "create_causal_mask(input_embeds=..., cache_position=...), which transformers 5 does not take",
)
def test_mobilemoe_remote_code(tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    from innards.view.loader import load_with_own_code

    model_id, revision = MOBILEMOE
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = load_with_own_code(model_id, revision=revision)  # builds and loads; the forward is what fails
    check(model, tok, tmp_path, "topk")
