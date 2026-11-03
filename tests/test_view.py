"""The web view's API (FastAPI TestClient) over a run folder, with and without a runner."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from replicas import chat_ids, replica, tokenizer

pytestmark = [pytest.mark.hf, pytest.mark.view]
POST = {"X-Innards": "1"}


class FakeRunner:
    """The smallest runner: loads a replica instead of a checkpoint and streams its turns."""

    name = "fake"

    def __init__(self, root: str) -> None:
        self.root = root
        self.watch: Any = None
        self.model: Any = None
        self.history: list[dict[str, str]] = []
        self.state = "idle"

    def status(self) -> dict[str, Any]:
        conv = {"run": self.watch.run, "id": self.watch.conversation} if self.watch else None
        return {
            "name": self.name,
            "state": self.state,
            "model": {"id": "replica/qwen3_5"} if self.watch else None,
            "conversation": conv,
        }

    def models(self) -> dict[str, Any]:
        return {"lineup": [], "downloaded": []}

    def validate(self, model: str, precision: Any, conversation_set: Any, document: Any = None) -> dict[str, Any]:
        return {"model": model, "status": "Ready", "ok": True, "checks": []}

    def load(self, model: str, precision: Any, device: Any, conversation_set: Any = None, document: Any = None) -> Any:
        import innards
        from innards.view import RunnerError

        if model == "too-big":
            raise RunnerError("Blocked", detail={"report": {"status": "Blocked", "checks": []}})
        self.model = replica("qwen3_5")
        self.watch = innards.observe(
            self.model, tokenizer(), model_id="Qwen/Qwen3.5-0.8B", run="view-test", root=self.root
        )
        self.state = "ready"
        return self.status()

    def unload(self) -> dict[str, Any]:
        if self.watch:
            self.watch.close()
        self.watch, self.state = None, "idle"
        return self.status()

    def new_conversation(self, conversation_set: Any = None) -> dict[str, Any]:
        self.history = []
        self.watch.new_conversation(conversation_set=conversation_set)
        return self.status()

    def _turn(self, job: Any, message: str, step: int | None = None) -> None:
        self.history.append({"role": "user", "content": message})
        job.turn_start(self.watch.turns + 1, message, step)
        inputs = chat_ids(self.history)
        self.watch.next_turn(message)
        out = self.model.generate(**inputs, max_new_tokens=4, min_new_tokens=4, do_sample=False)
        reply = tokenizer().decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        job.token(reply, 4)
        self.history.append({"role": "assistant", "content": reply})
        job.put("turn", {"record": self.watch.last_turn().record.to_dict()})

    def chat(self, message: str) -> Any:
        from innards.view import TurnJob

        job = TurnJob("chat", message)
        threading.Thread(target=lambda: (self._turn(job, message), job.close()), daemon=True).start()
        return job

    def run_set(self, conversation_set: str, document: Any = None, document_name: Any = None, steps: Any = None) -> Any:
        from innards.sets import get_set
        from innards.view import TurnJob

        messages = get_set(conversation_set, document, document_name).messages()[: steps or 2]
        job = TurnJob("set", conversation_set, len(messages))

        def work() -> None:
            for i, message in enumerate(messages, start=1):
                self._turn(job, message, i)
            job.close()

        threading.Thread(target=work, daemon=True).start()
        return job

    def stop(self) -> bool:
        return False

    def count_tokens(self, text: str) -> int | None:
        return len(tokenizer()(text)["input_ids"])


def events(response: Any) -> list[tuple[str, Any]]:
    out = []
    for block in response.text.split("\n\n"):
        lines = block.strip().splitlines()
        if lines and lines[0].startswith("event: "):
            out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    import innards

    for kind, model_id in (("qwen3", "Qwen/Qwen3-0.6B"), ("qwen3_5", "Qwen/Qwen3.5-0.8B")):
        model = replica(kind)
        watch = innards.observe(
            model, tokenizer(), model_id=model_id, run="study", root=str(tmp_path), conversation_set="with_context"
        )
        history: list[dict[str, str]] = []
        for message in ("hello my name is priya", "what is my name"):
            history.append({"role": "user", "content": message})
            watch.next_turn(message)
            out = model.generate(**chat_ids(history), max_new_tokens=4, min_new_tokens=4, do_sample=False)
            history.append({"role": "assistant", "content": tokenizer().decode(out[0, -4:])})
        watch.close()
    return tmp_path


def client(root: Path, **kwargs: Any) -> Any:
    from fastapi.testclient import TestClient

    from innards.store import RunStore
    from innards.view.app import create_app

    return TestClient(create_app(RunStore(root), **kwargs))


def test_runs_turns_and_analysis(run_folder: Path) -> None:
    c = client(run_folder)
    runs = c.get("/api/runs").json()
    assert runs[0]["name"] == "study"
    convs = {x["model"]: x["id"] for x in runs[0]["conversations"]}
    new, old = convs["Qwen/Qwen3.5-0.8B"], convs["Qwen/Qwen3-0.6B"]
    turns = c.get(f"/api/runs/study/{new}/turns").json()
    assert [t["turn"] for t in turns] == [1, 2]
    assert "data" not in turns[0]["features"]["gated_deltanet"]  # per-token series only in the analysis
    data = c.get(f"/api/runs/study/{new}/analysis", params={"compare": f"study/{old}"}).json()
    assert [t["id"] for t in data["tiles"]] == ["context", "kv", "memory", "decode", "ttft", "overhead"]
    assert data["changed"]["previous_id"] == "Qwen/Qwen3-0.6B"
    assert {a["id"] for a in data["changed"]["added"]} >= {"gated_deltanet", "gated_attention"}
    ids = [s["id"] for s in data["sections"]]
    assert ids[:2] == ["gated_deltanet", "gated_attention"]
    assert ids[-1] == "kv"
    kv = data["sections"][-1]
    kv_chart = kv["charts"][0]
    assert {s["model"] for s in kv_chart["series"]} == {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"}
    assert data["compare"]["models"] == ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"]
    assert len(data["compare"]["rows"]) == 2
    one = c.get(f"/api/runs/study/{new}/analysis", params={"first": 2}).json()
    assert one["turns"] == 1
    assert c.get("/api/runs/study/nope/analysis").status_code == 404
    assert c.get("/api/runs/..%2Fx/c/turns").status_code == 404


def test_lineup_sets_predicted_and_validate(run_folder: Path) -> None:
    c = client(run_folder)
    lineup = c.get("/api/lineup").json()
    assert len(lineup["pairs"]) == 2
    changed = c.get("/api/lineup/changed", params={"model": "ibm-granite/granite-swash-3b-a600m"}).json()
    assert {a["id"] for a in changed["added"]} == {"shared_expert"}
    assert {r["key"]: (r["previous"], r["newer"]) for r in changed["facts"]}["experts_per_token"] == (8, 4)
    predicted = c.get("/api/lineup/predicted", params={"model": "ibm-granite/granite-swash-3b-a600m"}).json()
    assert predicted["label"] == "predicted"
    assert predicted["points"][1]["total_bytes"] == 271_036_416  # 32K tokens: 20 of 28 layers hold one window
    assert c.get("/api/lineup/predicted", params={"model": "gpt2"}).status_code == 404
    sets = c.get("/api/sets").json()
    assert [s["name"] for s in sets] == ["basic_qa", "with_context", "input_docs", "long_document"]
    report = c.post(
        "/api/validate",
        headers=POST,
        json={
            "model": "ibm-granite/granite-swash-3b-a600m",
            "device": "mps",
            "precision": "fp8",
            "set": "with_context",
        },
    ).json()
    assert report["status"] == "Blocked"  # FP8 needs CUDA; the published bf16 runs here


def test_documents_upload(run_folder: Path) -> None:
    c = client(run_folder)
    doc = c.post(
        "/api/documents",
        headers=POST,
        json={"name": "notes.txt", "data_base64": base64.b64encode(b"The budget is 42.").decode()},
    ).json()
    assert doc["characters"] == 17
    bad = c.post("/api/documents", headers=POST, json={"name": "x.exe", "data_base64": base64.b64encode(b"x").decode()})
    assert bad.status_code == 422


def test_guards(run_folder: Path) -> None:
    from fastapi.testclient import TestClient

    from innards.store import RunStore
    from innards.view.app import create_app

    c = client(run_folder)
    assert c.post("/api/validate", json={"model": "x"}).status_code == 403  # no X-Innards header
    assert c.post("/api/load", headers=POST, json={"model": "x"}).status_code == 503  # no runner
    remote = TestClient(create_app(RunStore(run_folder)), base_url="http://example.com")
    assert remote.get("/api/runs").status_code == 403  # not a loopback host
    locked = TestClient(create_app(RunStore(run_folder), token="s3cret"), base_url="http://example.com")
    assert locked.get("/api/runs").status_code == 401
    assert locked.get("/api/runs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    page = c.get("/")
    assert "innards" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert c.get("/static/app.js").status_code == 200


def test_runner_load_chat_and_set(tmp_path: Path) -> None:
    runner = FakeRunner(str(tmp_path))
    c = client(tmp_path, runner=runner)
    try:
        blocked = c.post("/api/load", headers=POST, json={"model": "too-big"})
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["report"]["status"] == "Blocked"
        assert c.post("/api/load", headers=POST, json={"model": "Qwen/Qwen3.5-0.8B"}).json()["state"] == "ready"
        state = c.get("/api/state").json()
        assert state["runner"]["state"] == "ready"
        assert state["watch"]["features"]["gated_deltanet"]["status"] == "analyzed"
        chat = events(c.post("/api/chat", headers=POST, json={"message": "hello"}))
        kinds = [e for e, _ in chat]
        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        assert "token" in kinds
        record = next(d for e, d in chat if e == "turn")["record"]
        assert record["message"] == "hello"
        assert record["kv"]["measured_bytes"] == record["kv"]["predicted_bytes"]
        c.post("/api/conversation", headers=POST, json={"set": "basic_qa"})
        run = events(c.post("/api/set/run", headers=POST, json={"set": "basic_qa", "steps": 2}))
        assert [d["step"] for e, d in run if e == "turn_start"] == [1, 2]
        assert c.get("/api/state").json()["watch"]["turns"] == 2
    finally:
        runner.unload()


def test_serve_in_a_background_thread(run_folder: Path) -> None:
    import httpx

    import innards

    model = replica("qwen3")
    watch = innards.observe(model, tokenizer(), run="live", root=str(run_folder))
    server = watch.view(port=18765)
    try:
        model.generate(**chat_ids([{"role": "user", "content": "hello"}]), max_new_tokens=3, do_sample=False)
        watch.wait()
        state = httpx.get(f"{server.url}api/state", timeout=5).json()
        assert state["watch"]["turns"] == 1
        assert state["runner"] is None
        assert any(r["name"] == "live" for r in httpx.get(f"{server.url}api/runs", timeout=5).json())
    finally:
        server.stop()
        watch.close()


def test_experts_tab(tmp_path: Path) -> None:
    import torch
    from fastapi.testclient import TestClient

    import innards
    from innards.store import RunStore
    from innards.view.app import create_app
    from replicas import moe_replica

    model = moe_replica("qwen3_moe")
    tok = tokenizer()
    with innards.experts.trace(model, tok, run="topics", root=str(tmp_path), name="q") as trace:
        for topic, text in [("colours", "red green blue red"), ("numbers", "one two three four five")]:
            with trace.topic(topic), torch.no_grad():
                model.generate(**tok(text, return_tensors="pt"), max_new_tokens=3, do_sample=False)
    client = TestClient(create_app(RunStore(str(tmp_path))))
    listed = client.get("/api/experts").json()
    assert [(t["run"], t["name"], t["topics"]) for t in listed] == [("topics", "q", ["colours", "numbers"])]
    assert "path" not in listed[0]
    page = client.get("/api/experts/topics/q?topic=numbers&by=lift&coverage=0.8").json()
    assert page["topic"] == "numbers"
    ids = [s["id"] for s in page["sections"]]
    assert ids[:4] == ["experts-overview", "experts-layers", "experts-topics", "experts-needed"]
    heat = page["sections"][1]["charts"][0]
    assert heat["click"] == "expert"
    assert len(heat["z"]) == 2  # two MoE layers
    chosen = client.get("/api/experts/topics/q/selection?topic=numbers&coverage=0.8")
    assert chosen.headers["content-disposition"].startswith("attachment")
    assert chosen.json()["format"] == "innards.experts.selection/1.0"
    seqs = client.get("/api/experts/topics/q/sequences?topic=colours").json()
    assert [s["tokens"] for s in seqs] == [4 + 2]
    seq = client.get(f"/api/experts/topics/q/sequence/{seqs[0]['sequence']}?layer=1").json()
    columns = list(zip(*seq["charts"][0]["z"], strict=True))
    assert all(sum(v is not None for v in col) == 2 for col in columns)  # top-2 per token
    expert = client.get("/api/experts/topics/q/expert?layer=0&expert=0").json()
    assert expert["title"].startswith("Layer 0 expert 0")
    assert client.get("/api/experts/topics/missing").status_code == 404
    assert client.get("/api/experts/topics/q?topic=poetry").status_code == 200  # falls back to the first topic
    assert client.get("/api/experts/topics/q/selection?topic=poetry").status_code == 404
    assert client.get("/api/experts/..%2F/q").status_code == 404
