"""The web view's API (FastAPI TestClient): the MoE layers tab over a trace folder, and the Conversation tab."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from replicas import moe_replica, tokenizer

pytestmark = [pytest.mark.hf, pytest.mark.view]
POST = {"X-Innards": "1"}


def events(response: Any) -> list[tuple[str, Any]]:
    out = []
    for block in response.text.split("\n\n"):
        lines = block.strip().splitlines()
        if lines and lines[0].startswith("event: "):
            out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def client(root: Path, **kwargs: Any) -> Any:
    from fastapi.testclient import TestClient

    from innards.store import RunStore
    from innards.view.app import create_app

    return TestClient(create_app(RunStore(root), **kwargs))


@pytest.fixture
def topics_trace(tmp_path: Path) -> Path:
    import torch

    import innards

    model = moe_replica("qwen3_moe")
    tok = tokenizer()
    with innards.experts.trace(model, tok, run="topics", root=str(tmp_path), name="q") as trace:
        for topic, text in [("colours", "red green blue red"), ("numbers", "one two three four five")]:
            with trace.topic(topic), torch.no_grad():
                model.generate(**tok(text, return_tensors="pt"), max_new_tokens=3, do_sample=False)
    return tmp_path


def test_moe_layers_tab(topics_trace: Path) -> None:
    c = client(topics_trace)
    listed = c.get("/api/experts").json()
    assert [(t["run"], t["name"], t["topics"]) for t in listed] == [("topics", "q", ["colours", "numbers"])]
    assert "path" not in listed[0]
    page = c.get("/api/experts/topics/q?topic=numbers&by=lift&coverage=0.8").json()
    assert page["topic"] == "numbers"
    ids = [s["id"] for s in page["sections"]]
    assert ids[:4] == ["experts-overview", "experts-layers", "experts-topics", "experts-needed"]
    heat = page["sections"][1]["charts"][0]
    assert heat["click"] == "expert"
    assert len(heat["z"]) == 2  # two MoE layers
    chosen = c.get("/api/experts/topics/q/selection?topic=numbers&coverage=0.8")
    assert chosen.headers["content-disposition"].startswith("attachment")
    assert chosen.json()["format"] == "innards.experts.selection/1.0"
    seqs = c.get("/api/experts/topics/q/sequences?topic=colours").json()
    assert [s["tokens"] for s in seqs] == [4 + 2]
    seq = c.get(f"/api/experts/topics/q/sequence/{seqs[0]['sequence']}?layer=1").json()
    columns = list(zip(*seq["charts"][0]["z"], strict=True))
    assert len(columns) == 6
    assert all(sum(v is not None for v in col) == 2 for col in columns)  # top-2 per token
    part = c.get(f"/api/experts/topics/q/sequence/{seqs[0]['sequence']}?layer=1&first=1&last=3").json()
    assert len(part["charts"][0]["x"]) == 2  # positions 1 and 2
    assert "positions 1 to 2" in part["headline"]
    assert c.get(f"/api/experts/topics/q/sequence/{seqs[0]['sequence']}?layer=1&first=50").status_code == 404
    assert c.get(f"/api/experts/topics/q/sequence/{seqs[0]['sequence']}?layer=9").status_code == 404
    expert = c.get("/api/experts/topics/q/expert?layer=0&expert=0").json()
    assert expert["title"].startswith("Layer 0 expert 0")
    assert c.get("/api/experts/topics/q/conversations").json() == []  # traced from a script, not the page
    assert c.get("/api/experts/topics/missing").status_code == 404
    assert c.get("/api/experts/topics/q?topic=poetry").status_code == 200  # falls back to the first topic
    assert c.get("/api/experts/topics/q/selection?topic=poetry").status_code == 404
    assert c.get("/api/experts/..%2F/q").status_code == 404


def test_guards_and_two_tabs_only(topics_trace: Path) -> None:
    from fastapi.testclient import TestClient

    from innards.store import RunStore
    from innards.view.app import create_app

    c = client(topics_trace)
    assert c.get("/api/info").json()["chat"] is False
    assert c.get("/api/state").json() == {"runner": None}
    assert c.post("/api/chat", json={"message": "x"}).status_code == 403  # no X-Innards header
    no_runner = c.post("/api/chat", headers=POST, json={"message": "x"})
    assert no_runner.status_code == 503
    assert "trace.view(chat=True)" in no_runner.json()["detail"]
    for gone in ("/api/runs", "/api/lineup", "/api/sets", "/api/lineup/predicted?model=gpt2"):
        assert c.get(gone).status_code == 404, gone  # the Analysis tab, KV and the other monitors are gone
    remote = TestClient(create_app(RunStore(topics_trace)), base_url="http://example.com")
    assert remote.get("/api/experts").status_code == 403  # not a loopback host
    locked = TestClient(create_app(RunStore(topics_trace), token="s3cret"), base_url="http://example.com")
    assert locked.get("/api/experts").status_code == 401
    assert locked.get("/api/experts", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    page = c.get("/")
    assert 'data-tab="conversation"' in page.text
    assert 'data-tab="moe"' in page.text
    assert page.text.count('role="tab"') == 2
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["cache-control"] == "no-cache"
    # scripts are fetched under a hash of their content, so a browser never runs an older Innards' app.js
    script = next(part.split('"')[0] for part in page.text.split('src="')[1:] if part.startswith("/static/app.js"))
    assert script.startswith("/static/app.js?v=")
    assert 'href="/static/style.css?v=' in page.text
    served = c.get(script)
    assert served.status_code == 200
    assert served.headers["cache-control"] == "no-cache"
    assert "renderLoadBar" in served.text


def test_conversation_routes_each_token_once(tmp_path: Path) -> None:
    import innards
    from innards.view.chat import ChatRunner

    model = moe_replica("qwen3_moe")
    trace = innards.experts.trace(model, tokenizer(), run="chat", root=str(tmp_path), name="q")
    runner = ChatRunner(trace, generate={"max_new_tokens": 4, "min_new_tokens": 4, "do_sample": False})
    c = client(tmp_path, runner=runner)
    try:
        state = c.get("/api/state").json()["runner"]
        assert (state["state"], state["trace"], state["turns"]) == ("ready", {"run": "chat", "name": "q"}, 0)
        first = events(c.post("/api/chat", headers=POST, json={"message": "hello my name is priya", "topic": "intro"}))
        kinds = [e for e, _ in first]
        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        assert "token" in kinds
        assert next(d for e, d in first if e == "turn_start")["topic"] == "intro"
        one = next(d for e, d in first if e == "turn")["record"]
        second = events(c.post("/api/chat", headers=POST, json={"message": "what is my name", "topic": "recall"}))
        two = next(d for e, d in second if e == "turn")["record"]
        assert (one["turn"], two["turn"]) == (1, 2)
        assert one["tokens"]["reused"] == 0
        assert one["tokens"]["generated"] == 4
        # turn 2 reuses the cache: it routes only the tokens turn 1 did not, in the same sequence
        assert two["tokens"]["reused"] == one["positions"][1]
        assert two["positions"][0] == one["positions"][1]
        assert two["sequence"] == one["sequence"]
        routed = one["tokens"]["routed"] + two["tokens"]["routed"]
        summary = json.loads((tmp_path / "chat/experts/q/summary.json").read_text())
        assert summary["topics"] == ["intro", "recall"]
        assert [row[0] for row in summary["tokens"]] == [one["tokens"]["routed"], two["tokens"]["routed"]]
        stored = c.get("/api/experts/chat/q/conversations").json()
        assert [(s["id"], [t["turn"] for t in s["turns"]]) for s in stored] == [(one["conversation"], [1, 2])]
        assert stored[0]["turns"][1]["reply"] == two["reply"]
        a, b = two["positions"]
        part = c.get(f"/api/experts/chat/q/sequence/{two['sequence']}?layer=0&first={a}&last={b}").json()
        assert len(part["charts"][0]["x"]) == b - a == two["tokens"]["routed"]
        assert part["headline"].startswith("recall ·")
        whole = c.get(f"/api/experts/chat/q/sequence/{two['sequence']}?layer=0").json()
        assert len(whole["charts"][0]["x"]) == routed
        assert c.post("/api/stop", headers=POST).json() == {"stopped": False}
        fresh = c.post("/api/conversation", headers=POST).json()
        assert fresh["conversation"] != one["conversation"]
        assert fresh["turns"] == 0
        three = next(d for e, d in events(c.post("/api/chat", headers=POST, json={"message": "hello"})) if e == "turn")
        assert three["record"]["topic"] == "recall"  # the last topic stays until the page names another
        assert three["record"]["tokens"]["reused"] == 0
        assert three["record"]["sequence"] != one["sequence"]
        assert c.post("/api/chat", headers=POST, json={"message": "x", "topic": ""}).status_code == 422
    finally:
        trace.close()


def test_conversation_never_carries_recurrent_state(tmp_path: Path) -> None:
    """A cache with conv or SSM state is not passed to the next turn: every turn is prefilled from the start."""
    import innards
    from innards.view.chat import ChatRunner

    model = moe_replica("lfm2_moe")
    trace = innards.experts.trace(model, tokenizer(), run="chat", root=str(tmp_path), name="lfm")
    runner = ChatRunner(trace, generate={"max_new_tokens": 3, "min_new_tokens": 3, "do_sample": False})
    c = client(tmp_path, runner=runner)
    try:
        records = [
            next(d for e, d in events(c.post("/api/chat", headers=POST, json={"message": m})) if e == "turn")["record"]
            for m in ("hello my name is priya", "what is my name")
        ]
        assert [r["tokens"]["reused"] for r in records] == [0, 0]
        assert records[0]["sequence"] != records[1]["sequence"]
        assert records[1]["positions"][0] == 0
    finally:
        trace.close()


def test_chat_runner_checks_its_inputs(tmp_path: Path) -> None:
    import innards
    from innards.view.chat import ChatRunner

    model = moe_replica("qwen3_moe")
    bare = innards.experts.trace(model, None, run="r", root=str(tmp_path), name="no-tok")
    with bare, pytest.raises(ValueError, match="tokenizer"):
        ChatRunner(bare)
    trace = innards.experts.trace(model, tokenizer(), run="r", root=str(tmp_path), name="q")
    with trace, pytest.raises(ValueError, match="past_key_values"):
        ChatRunner(trace, generate={"past_key_values": None})
    memory = innards.experts.trace(model, tokenizer(), store=False)
    with memory, pytest.raises(RuntimeError, match="store"):
        memory.view(chat=True)


def _turn(c: Any, message: str, topic: str | None = None) -> dict[str, Any]:
    body = {"message": message, **({"topic": topic} if topic else {})}
    return next(d for e, d in events(c.post("/api/chat", headers=POST, json=body)) if e == "turn")["record"]


def test_text_mode_for_a_base_model(tmp_path: Path) -> None:
    """No chat template: each message is added, as typed, to the text so far, and the model continues it."""
    import innards
    from innards.view.chat import ChatRunner

    base = copy.deepcopy(tokenizer())  # tokenizer() is shared across tests
    base.chat_template = None
    trace = innards.experts.trace(moe_replica("qwen3_moe"), base, run="chat", root=str(tmp_path), name="base")
    runner = ChatRunner(trace, generate={"max_new_tokens": 4, "min_new_tokens": 4, "do_sample": False})
    c = client(tmp_path, runner=runner)
    try:
        assert c.get("/api/state").json()["runner"]["mode"] == "text"
        one = _turn(c, "hello my name is priya", "intro")
        two = _turn(c, "what is my name", "recall")
        assert (one["mode"], two["mode"]) == ("text", "text")
        assert one["tokens"]["prompt"] == len(base("hello my name is priya")["input_ids"])
        # turn 2's prompt: turn 1's whole text (prompt and reply), then the new message, nothing made up between
        added = len(base("what is my name", add_special_tokens=False)["input_ids"])
        assert two["tokens"]["prompt"] == one["positions"][1] + 1 + added
        assert two["tokens"]["reused"] == one["positions"][1]  # the cache held exactly that prefix
        assert two["sequence"] == one["sequence"]
        whole = c.get(f"/api/experts/chat/base/sequence/{two['sequence']}?layer=0").json()
        assert len(whole["charts"][0]["x"]) == one["tokens"]["routed"] + two["tokens"]["routed"]
    finally:
        trace.close()


def test_trace_view_serves_the_conversation_in_a_background_thread(tmp_path: Path) -> None:
    import httpx

    import innards

    model = moe_replica("qwen3_moe")
    trace = innards.experts.trace(model, tokenizer(), run="live", root=str(tmp_path), name="q")
    server = trace.view(port=18765, chat={"max_new_tokens": 3, "do_sample": False}, topic="greeting")
    try:
        state = httpx.get(f"{server.url}api/state", timeout=5).json()["runner"]
        assert (state["state"], state["topic"], state["trace"]["run"]) == ("ready", "greeting", "live")
        reply = httpx.post(f"{server.url}api/chat", headers=POST, json={"message": "hello"}, timeout=60)
        record = next(d for e, d in events(reply) if e == "turn")["record"]
        assert record["topic"] == "greeting"
        listed = httpx.get(f"{server.url}api/experts", timeout=5).json()
        assert listed[0]["topics"] == ["greeting"]
        assert listed[0]["tokens"]["greeting"] == record["tokens"]["routed"]
    finally:
        server.stop()
        trace.close()


def _saved(kind: str, folder: Path, moe: bool = True) -> Path:
    from replicas import replica

    model = moe_replica(kind) if moe else replica(kind)
    model.save_pretrained(folder)
    tokenizer().save_pretrained(folder)
    return folder


def _wait_for(c: Any, *states: str) -> dict[str, Any]:
    import time

    for _ in range(600):
        runner: dict[str, Any] = c.get("/api/state").json()["runner"]
        if runner["state"] in states:
            return runner
        time.sleep(0.1)
    raise AssertionError(f"runner never reached {states}: {runner}")


def test_innards_view_loads_a_model_and_chats(tmp_path: Path) -> None:
    """Independent use: the page picks a checkpoint, Innards loads and traces it, and the chat runs on it."""
    from innards.view.loader import ModelRunner

    moe = _saved("qwen3_moe", tmp_path / "models" / "tiny-moe")
    dense = _saved("qwen3", tmp_path / "models" / "tiny-dense", moe=False)
    runs = tmp_path / "runs"
    runner = ModelRunner(runs, "chat", generate={"max_new_tokens": 3, "do_sample": False}, device="cpu")
    c = client(runs, runner=runner)
    try:
        assert c.get("/api/info").json()["load"] is True
        assert isinstance(c.get("/api/models").json(), list)
        idle = c.get("/api/state").json()["runner"]
        assert (idle["state"], idle["trace"], idle["loadable"]) == ("idle", None, True)
        assert c.post("/api/chat", headers=POST, json={"message": "hello"}).status_code == 409  # nothing loaded
        assert c.post("/api/load", headers=POST, json={"model": str(moe)}).json()["state"] == "loading"
        ready = _wait_for(c, "ready", "error")
        assert ready["state"] == "ready", ready["error"]
        assert ready["trace"] == {"run": "chat", "name": "tiny-moe"}
        assert ready["device"] == "cpu"
        assert {ch["id"] for ch in ready["checks"]} >= {"memory"}
        reply = events(c.post("/api/chat", headers=POST, json={"message": "hello my name is priya", "topic": "intro"}))
        record = next(d for e, d in reply if e == "turn")["record"]
        assert record["tokens"]["generated"] == 3
        listed = c.get("/api/experts").json()
        assert [(t["name"], t["topics"]) for t in listed] == [("tiny-moe", ["intro"])]
        assert c.get("/api/experts/chat/tiny-moe/conversations").json()[0]["turns"][0]["message"].startswith("hello")
        # a dense model is refused, and the MoE model it replaced was closed (its trace is complete on disk)
        c.post("/api/load", headers=POST, json={"model": str(dense)})
        failed = _wait_for(c, "error", "ready")
        assert failed["state"] == "error"
        assert "not an MoE model" in failed["error"]
        assert failed["trace"] is None
        assert c.post("/api/chat", headers=POST, json={"message": "hello"}).status_code == 409
        c.post("/api/load", headers=POST, json={"model": str(moe)})
        again = _wait_for(c, "ready", "error")
        assert again["state"] == "ready"
        assert again["conversation"] != record["conversation"]
        assert c.post("/api/unload", headers=POST).json()["state"] == "idle"
    finally:
        runner.shutdown()


def _ships_code(folder: Path, model_type: str, files: dict[str, str], auto_map: dict[str, str]) -> Path:
    """Make a saved checkpoint name its own code in config.json, the way Hub repos that ship code do."""
    config = json.loads((folder / "config.json").read_text())
    config.update(model_type=model_type, auto_map=auto_map)
    (folder / "config.json").write_text(json.dumps(config))
    for name, code in files.items():
        (folder / name).write_text(code)
    return folder


OWN_CONFIG = """from transformers import Qwen3MoeConfig


class TinyMoeConfig(Qwen3MoeConfig):
    model_type = "innards_test_moe"
"""

OWN_MODELING = """import torch
from transformers import Qwen3MoeForCausalLM

from .configuration_tiny import TinyMoeConfig


class TinyMoeForCausalLM(Qwen3MoeForCausalLM):
    config_class = TinyMoeConfig
    _tied_weights_keys = ["lm_head.weight"]  # how transformers 4 code declared ties

    def __init__(self, config):
        super().__init__(config)
        # computed from a value in __init__, as MobileMoE's RoPE setup is: impossible on the meta device
        self.register_buffer("scale", torch.tensor([float(torch.arange(4).sum().item())]), persistent=False)

    def forward(self, *args, **kwargs):
        if getattr(self.config, "innards_fail", False):
            raise TypeError("create_causal_mask() got an unexpected keyword argument 'input_embeds'")
        return super().forward(*args, **kwargs)
"""


def _own_code(folder: Path, tie: bool = False) -> Path:
    """A tiny MoE checkpoint for an architecture transformers lacks, shipping its own code the way Hub repos do.

    ``tie``: the config ties the output embedding to the input one, and the checkpoint does not store it.
    """
    from safetensors.torch import save_file

    model = moe_replica("qwen3_moe")
    model.save_pretrained(folder)  # config and generation config
    tokenizer().save_pretrained(folder)
    for old in folder.glob("model*.safetensors*"):
        old.unlink()
    # stored under the code's own module names, as checkpoints that ship their code store them
    stored = {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()}
    if tie:
        stored.pop("lm_head.weight")
    save_file(stored, folder / "model.safetensors", metadata={"format": "pt"})
    _ships_code(
        folder,
        "innards_test_moe",
        {"configuration_tiny.py": OWN_CONFIG, "modeling_tiny.py": OWN_MODELING},
        {"AutoConfig": "configuration_tiny.TinyMoeConfig", "AutoModelForCausalLM": "modeling_tiny.TinyMoeForCausalLM"},
    )
    if tie:
        config = json.loads((folder / "config.json").read_text())
        (folder / "config.json").write_text(json.dumps({**config, "tie_word_embeddings": True}))
    return folder


def test_load_with_own_code_builds_on_real_tensors(tmp_path: Path) -> None:
    """Own code that reads values in __init__ and declares ties the transformers 4 way loads with every
    weight from the checkpoint; a weight the checkpoint lacks stops the load; its failures are explained."""
    import shutil

    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    import innards
    from innards.view.chat import ChatRunner
    from innards.view.loader import load_with_own_code
    from innards.view.runner import RunnerError

    folder = _own_code(tmp_path / "models" / "innards_test_tied_moe", tie=True)
    untied = _own_code(tmp_path / "models" / "innards_test_untied_moe")
    # what from_pretrained does with such code in transformers 5: it cannot read the list form of ties ...
    with pytest.raises(AttributeError, match="'list' object has no attribute 'keys'"):
        AutoModelForCausalLM.from_pretrained(str(folder), trust_remote_code=True)
    # ... and builds on the meta device, where a value read in __init__ does not exist
    with pytest.raises(RuntimeError, match="meta tensors"):
        AutoModelForCausalLM.from_pretrained(str(untied), trust_remote_code=True)
    model = load_with_own_code(str(folder), device="cpu")
    assert model.scale.item() == 6.0  # computed by its own __init__, on real tensors
    assert model.lm_head.weight is model.get_input_embeddings().weight  # tie_word_embeddings, as transformers 4 did
    stored = load_file(folder / "model.safetensors")
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, stored[name if name != "lm_head.weight" else "model.embed_tokens.weight"]), name
    assert not model.training

    gap = shutil.copytree(folder, tmp_path / "models" / "innards_test_gap_moe")
    stored.pop(next(k for k in stored if ".layers.0." in k and k.endswith("q_proj.weight")))
    save_file(stored, gap / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RunnerError, match="no weights for 1 tensors its code defines"):
        load_with_own_code(str(gap))

    model.config.innards_fail = True  # its forward raises, as MobileMoE's does on transformers 5
    trace = innards.experts.trace(model, copy.deepcopy(tokenizer()), run="r", root=str(tmp_path), name="own")
    runner = ChatRunner(trace, generate={"max_new_tokens": 2})
    c = client(tmp_path, runner=runner)
    try:
        error = next(d for e, d in events(c.post("/api/chat", headers=POST, json={"message": "hello"})) if e == "error")
        assert "innards_test_tied_moe's own code fails with transformers" in error["message"]
        assert "does not patch it" in error["message"]
    finally:
        trace.close()


def test_innards_view_runs_a_checkpoints_own_code_only_when_allowed(tmp_path: Path) -> None:
    """``--trust-remote-code`` loads an architecture transformers lacks, and never replaces a native class."""
    from innards.view.loader import ModelRunner

    own = _own_code(tmp_path / "models" / "innards_test_own_code_moe")
    native = _ships_code(  # a native architecture whose repo also ships code: that code must never run
        _saved("qwen3_moe", tmp_path / "models" / "native-moe"),
        "qwen3_moe",
        {"modeling_broken.py": "raise RuntimeError('the checkpoint code ran')\n"},
        {"AutoModelForCausalLM": "modeling_broken.Broken"},
    )
    runs = tmp_path / "runs"
    generate = {"max_new_tokens": 2, "do_sample": False}
    refusing = ModelRunner(runs, "chat", generate=generate, device="cpu")
    c = client(runs, runner=refusing)
    try:
        c.post("/api/load", headers=POST, json={"model": str(own)})
        refused = _wait_for(c, "error", "ready")
        assert refused["state"] == "error"
        assert "ships its own code (configuration_tiny.py, modeling_tiny.py)" in refused["error"]
        assert "innards view --trust-remote-code" in refused["error"]
        assert (refused["trust_remote_code"], refused["remote_code"]) == (False, False)
    finally:
        refusing.shutdown()
    allowing = ModelRunner(runs, "chat", generate=generate, device="cpu", trust_remote_code=True)
    c = client(runs, runner=allowing)
    try:
        c.post("/api/load", headers=POST, json={"model": str(own)})
        ready = _wait_for(c, "ready", "error")
        assert ready["state"] == "ready", ready["error"]
        assert (ready["trust_remote_code"], ready["remote_code"]) == (True, True)
        arch = next(ch for ch in ready["checks"] if ch["id"] == "architecture")
        assert arch["status"] == "warning"
        assert _turn(c, "hello my name is priya", "intro")["tokens"]["generated"] == 2
        c.post("/api/load", headers=POST, json={"model": str(native)})
        plain = _wait_for(c, "ready", "error")
        assert plain["state"] == "ready", plain["error"]  # the broken file was never imported
        assert plain["remote_code"] is False
        assert _turn(c, "hello", "intro")["tokens"]["generated"] == 2
    finally:
        allowing.shutdown()


def test_package_use_cannot_load_models(tmp_path: Path) -> None:
    import innards
    from innards.view.chat import ChatRunner

    trace = innards.experts.trace(moe_replica("qwen3_moe"), tokenizer(), run="r", root=str(tmp_path), name="q")
    c = client(tmp_path, runner=ChatRunner(trace))
    try:
        assert c.get("/api/info").json()["load"] is False
        assert c.get("/api/models").json() == []
        refused = c.post("/api/load", headers=POST, json={"model": "x"})
        assert refused.status_code == 409
        assert "innards view" in refused.json()["detail"]
    finally:
        trace.close()


def test_catalogue_lists_models_the_installed_transformers_builds() -> None:
    """Every model the view offers before download is an MoE architecture transformers here builds natively."""
    from transformers import CONFIG_MAPPING

    from innards.view.loader import catalogue

    data = catalogue()
    assert data["transformers"]
    assert data["checked"]
    assert data["budget_gb"] == 16
    ids = [m["id"] for m in data["models"]]
    assert len(ids) == len(set(ids)) > 0
    for m in data["models"]:
        assert m["model_type"] in CONFIG_MAPPING, m["id"]
        assert m["experts"] >= 2, m["id"]
        assert m["moe_layers"] > 0, m["id"]
        assert 0 < m["weights_gb"] <= 15, m["id"]
        assert m["precision"] in {"bf16", "fp16", "fp32"}, m["id"]
    released = [m["released"] for m in data["models"]]
    assert released == sorted(released, reverse=True)
    assert "facebook/MobileMoE-S-Base" not in ids  # its code targets transformers 4.57


def _cached(hub: Path, repo: str, files: dict[str, Any]) -> None:
    """A repo in a Hugging Face cache laid out as huggingface_hub writes it (blobs, snapshot links, refs)."""
    import hashlib

    folder = hub / f"models--{repo.replace('/', '--')}"
    revision = hashlib.sha1(repo.encode()).hexdigest()
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(revision)
    (folder / "blobs").mkdir()
    snapshot = folder / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name, content in files.items():
        data = content if isinstance(content, bytes) else json.dumps(content).encode()
        blob = hashlib.sha256(data).hexdigest()
        (folder / "blobs" / blob).write_bytes(data)
        (snapshot / name).symlink_to(f"../../blobs/{blob}")


def test_model_list_puts_what_runs_here_first(tmp_path: Path) -> None:
    """Only what transformers here builds natively: downloaded first, then the catalogue newest first.
    A checkpoint that runs only with its own code is left out even when downloaded, and a cached config
    without weights is not offered unless the catalogue names it."""
    from innards.view.loader import catalogue, moe_models

    hub = tmp_path / "hub"
    moe = {"model_type": "qwen3_moe", "num_experts": 8, "num_experts_per_tok": 2}
    weights = {"model.safetensors": b"\0" * 64}
    _cached(hub, "me/native-moe", {"config.json": moe, "tokenizer_config.json": {"chat_template": "x"}, **weights})
    _cached(
        hub, "me/own-code-moe", {"config.json": {**moe, "model_type": "innards_x", "auto_map": {"a": "m.C"}}, **weights}
    )
    _cached(hub, "me/dense", {"config.json": {"model_type": "qwen3"}, **weights})
    _cached(hub, "me/config-only-moe", {"config.json": moe})
    listed = catalogue()["models"][0]
    _cached(hub, listed["id"], {"config.json": {"model_type": listed["model_type"]}})  # config read, no weights

    models = moe_models(hub)
    ids = [m["id"] for m in models]
    assert ids[0] == "me/native-moe"
    assert "me/own-code-moe" not in ids  # downloaded, but transformers here cannot build it
    assert "me/dense" not in ids
    assert "me/config-only-moe" not in ids
    first = models[0]
    assert (first["downloaded"], first["chat_template"], first["experts"]) == (True, True, 8)
    offered = models[1:]
    assert [m["id"] for m in offered] == [m["id"] for m in catalogue()["models"]]  # newest release first
    assert offered[0]["id"] == listed["id"]
    assert offered[0]["downloaded"] is False
    assert {m["checked_with"] for m in offered} == {catalogue()["transformers"]}
    assert min(m["size_bytes"] for m in offered) > 0
