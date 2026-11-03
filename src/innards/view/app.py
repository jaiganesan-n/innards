"""The web view's FastAPI app: a JSON API over the run folder, the live state, and the page.

Read-only (always):

    GET  /healthz                                     liveness
    GET  /api/info                                    versions, run folder, whether a runner or a watch is attached
    GET  /api/state                                   live: loaded model, turn in progress, light tier, features, health
    GET  /api/runs                                    runs and their conversations, newest first
    GET  /api/runs/{run}/{conversation}               the conversation record
    GET  /api/runs/{run}/{conversation}/turns         its TurnRecords
    GET  /api/runs/{run}/{conversation}/analysis      the Analysis tab (?compare=run/conversation&first=&last=)
    GET  /api/lineup                                  the lineup: pairs and models
    GET  /api/lineup/changed?model=                   what a lineup model changed from the version before
    GET  /api/lineup/predicted?model=                 KV predicted from the config (for pairs not run yet)
    GET  /api/sets                                    conversation sets
    GET  /api/experts                                 MoE expert traces in the run folder
    GET  /api/experts/{run}/{name}                    the Experts tab (?topic=&by=&layer=&coverage=)
    GET  /api/experts/{run}/{name}/selection          the expert set a topic needs (?topic=&by=&coverage=), JSON
    GET  /api/experts/{run}/{name}/sequences          recorded sequences (?topic=); needs torch
    GET  /api/experts/{run}/{name}/sequence/{seq}     one sequence's tokens x experts (?layer=); needs torch
    GET  /api/experts/{run}/{name}/expert             the tokens one expert takes (?layer=&expert=&topic=); needs torch
    POST /api/validate                                {model, precision, set, document} → checks (Hub metadata only)
    POST /api/documents                               {name, data_base64} → {id, characters, tokens}

With a runner (e.g. ``innards-exp serve``), which loads and runs models:

    POST /api/load        {model, precision, device}    validation first; blocked checks come back as 409
    POST /api/unload
    POST /api/conversation {set}
    POST /api/chat        {message}                     → text/event-stream
    POST /api/set/run     {set, document}               → text/event-stream, one turn after another
    POST /api/stop

Every ``POST`` needs an ``X-Innards: 1`` header (it forces a CORS preflight, so other web pages cannot
drive the view). Set ``INNARDS_VIEW_TOKEN`` to require ``Authorization: Bearer <token>`` on ``/api``;
without a token, requests addressed to anything but a loopback host are refused. On AWS, reach the
view through an SSH port forward (``ssh -L 8000:127.0.0.1:8000 ...``).
"""

from __future__ import annotations

import base64
import binascii
import hmac
import os
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from innards import __version__
from innards.store import RunStore, valid_name
from innards.view.runner import Runner, RunnerError

__all__ = ["HEADER", "create_app"]

STATIC = Path(__file__).parent / "static"
HEADER = "x-innards"
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]", "testserver"}
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
_MAX_DOCUMENT_BYTES = 20_000_000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadBody(_Strict):
    model: str = Field(min_length=1, max_length=512)
    precision: str | None = Field(default=None, pattern=r"^(auto|bf16|fp16|fp32|fp8|fp4|fp4/fp8)$")
    device: str | None = Field(default=None, pattern=r"^(cpu|mps|cuda(:\d+)?)$")
    set: str | None = Field(default=None, max_length=64)
    document: str | None = Field(default=None, max_length=64)


class ValidateBody(_Strict):
    model: str = Field(min_length=1, max_length=512)
    precision: str | None = Field(default=None, max_length=16)
    device: str | None = Field(default=None, pattern=r"^(cpu|mps|cuda(:\d+)?)$")
    set: str | None = Field(default=None, max_length=64)
    document: str | None = Field(default=None, max_length=64)


class ConversationBody(_Strict):
    set: str | None = Field(default=None, max_length=64)


class ChatBody(_Strict):
    message: str = Field(min_length=1, max_length=400_000)


class SetBody(_Strict):
    set: str = Field(min_length=1, max_length=64)
    document: str | None = Field(default=None, max_length=64)
    steps: int | None = Field(default=None, ge=1, le=100)


class DocumentBody(_Strict):
    name: str = Field(min_length=1, max_length=256)
    data_base64: str = Field(min_length=1, max_length=_MAX_DOCUMENT_BYTES * 4 // 3 + 8)


def _host(request: Request) -> str:
    host = request.headers.get("host", "")
    if host.startswith("["):
        return host.split("]")[0] + "]"
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def create_app(
    store: RunStore | None = None,
    *,
    watch: Any = None,
    runner: Runner | None = None,
    token: str | None = None,
) -> FastAPI:
    """Build the app over ``store``. ``watch`` shows a script's live model; ``runner`` enables Load/Send."""
    store = store or RunStore()
    token = token if token is not None else os.environ.get("INNARDS_VIEW_TOKEN") or None
    documents: dict[str, dict[str, Any]] = {}
    documents_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        shutdown = getattr(runner, "shutdown", None)
        if callable(shutdown):
            shutdown()

    app = FastAPI(
        title="innards view",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.state.store = store
    app.state.runner = runner
    app.state.watch = watch

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        path = request.url.path
        if path.startswith("/api") and path not in {"/api/docs", "/api/openapi.json"}:
            if token:
                header = request.headers.get("authorization", "")
                supplied = header[7:] if header.lower().startswith("bearer ") else ""
                if not hmac.compare_digest(supplied, token):
                    return JSONResponse(
                        {"detail": "Unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
                    )
            elif _host(request) not in _LOOPBACK:
                return JSONResponse(
                    {
                        "detail": "The view answers loopback hosts only; set INNARDS_VIEW_TOKEN to "
                        "serve it on another address, or use an SSH port forward."
                    },
                    status_code=403,
                )
            if request.method == "POST" and request.headers.get(HEADER) != "1":
                return JSONResponse({"detail": f"POST requests need the {HEADER}: 1 header."}, status_code=403)
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not path.startswith("/api/docs"):
            response.headers.setdefault("Content-Security-Policy", _CSP)
        if path.startswith("/static/vendor/"):
            response.headers["Cache-Control"] = "public, max-age=604800, immutable"
        elif path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def live_watch() -> Any:
        if runner is not None and runner.watch is not None:
            return runner.watch
        return watch

    def need_runner() -> Runner:
        if runner is None:
            raise HTTPException(
                503,
                "No runner is attached: this view monitors the model your code runs. "
                "Start the view with a runner (innards-exp serve) to load models here.",
            )
        return runner

    def call(fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except RunnerError as exc:
            detail: Any = str(exc) if exc.detail is None else {"message": str(exc), **exc.detail}
            raise HTTPException(exc.status, detail) from exc

    def check_names(*names: str) -> None:
        for name in names:
            if not valid_name(name):
                raise HTTPException(404, "Not found")

    # ------------------------------------------------------------------ health, info, state

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        from innards._env import environment

        return {
            "innards": __version__,
            "store": store.url,
            "runner": getattr(runner, "name", None),
            "watch": live_watch() is not None,
            "auth": bool(token),
            "environment": environment(),
        }

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        w = live_watch()
        return {
            "runner": runner.status() if runner is not None else None,
            "watch": w.status() if w is not None else None,
        }

    # ------------------------------------------------------------------ runs

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return [{k: v for k, v in m.items() if k != "environment"} for m in store.list_runs()]

    @app.get("/api/runs/{run}/{conversation}")
    def conversation_record(run: str, conversation: str) -> dict[str, Any]:
        check_names(run, conversation)
        found = store.read_conversation(run, conversation)
        if found is None:
            raise HTTPException(404, "Conversation not found")
        return {k: v for k, v in found.items() if k != "config"}

    @app.get("/api/runs/{run}/{conversation}/turns")
    def turns(run: str, conversation: str, features: bool = False) -> list[dict[str, Any]]:
        check_names(run, conversation)
        rows = store.turns(run, conversation)
        if not features:  # the conversation tab needs text and the light tier, not per-token series
            for row in rows:
                row["features"] = {
                    k: {"summary": v.get("summary"), "title": v.get("title")}
                    for k, v in (row.get("features") or {}).items()
                }
        return rows

    @app.get("/api/runs/{run}/{conversation}/analysis")
    def analysis(
        run: str,
        conversation: str,
        compare: str | None = None,
        first: int | None = Query(None, ge=1),
        last: int | None = Query(None, ge=1),
    ) -> dict[str, Any]:
        from innards.view.analysis import analysis as build

        check_names(run, conversation)
        other = None
        if compare:
            parts = compare.split("/")
            if len(parts) != 2:
                raise HTTPException(422, "compare must be run/conversation")
            check_names(*parts)
            other = (parts[0], parts[1])
        try:
            return build(store, run, conversation, compare=other, first=first, last=last)
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0] if exc.args else exc)) from exc

    # ------------------------------------------------------------------ expert routing (MoE)

    def trace_folder(run: str, name: str) -> Path:
        check_names(run, name)
        if store.only and run != store.only:
            raise HTTPException(404, "Not found")
        folder = store.root / run / "experts" / name
        if not (folder / "summary.json").is_file():
            raise HTTPException(404, "No expert trace with that name")
        return folder

    def experts_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0] if exc.args else exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(501, f"Per-token routing needs torch and safetensors here ({exc.name})") from exc

    @app.get("/api/experts")
    def expert_traces() -> list[dict[str, Any]]:
        from innards.view.experts import trace_list

        return [{k: v for k, v in t.items() if k != "path"} for t in trace_list(store.root, store.only)]

    @app.get("/api/experts/{run}/{name}")
    def experts(
        run: str,
        name: str,
        topic: str | None = Query(None, max_length=512),
        by: Literal["selection", "gate", "lift"] = "selection",
        layer: int | None = None,
        coverage: float = Query(0.9, gt=0, le=1),
    ) -> dict[str, Any]:
        from innards.view.experts import experts_page

        return dict(experts_call(experts_page, trace_folder(run, name), topic, by, layer, coverage))

    @app.get("/api/experts/{run}/{name}/selection")
    def selection(
        run: str,
        name: str,
        topic: str = Query(min_length=1, max_length=512),
        by: Literal["selection", "gate"] = "selection",
        coverage: float = Query(0.9, gt=0, le=1),
    ) -> JSONResponse:
        from innards.experts import load

        chosen = experts_call(lambda: load(trace_folder(run, name)).select(topic, coverage, by=by))
        filename = f"{name}-{''.join(c if c.isalnum() else '-' for c in topic)}-{round(coverage * 100)}.json"
        return JSONResponse(chosen.to_dict(), headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.get("/api/experts/{run}/{name}/sequences")
    def sequences(run: str, name: str, topic: str | None = Query(None, max_length=512)) -> list[dict[str, Any]]:
        from innards.experts import TokenRouting

        folder = trace_folder(run, name)
        return list(experts_call(lambda: TokenRouting(folder).sequences(topic)))

    @app.get("/api/experts/{run}/{name}/sequence/{sequence}")
    def sequence(run: str, name: str, sequence: int, layer: int) -> dict[str, Any]:
        from innards.view.experts import sequence_section

        return dict(experts_call(sequence_section, trace_folder(run, name), sequence, layer))

    @app.get("/api/experts/{run}/{name}/expert")
    def expert(
        run: str, name: str, layer: int, expert: int = Query(ge=0), topic: str | None = Query(None, max_length=512)
    ) -> dict[str, Any]:
        from innards.view.experts import expert_tokens

        return dict(experts_call(expert_tokens, trace_folder(run, name), layer, expert, topic))

    # ------------------------------------------------------------------ lineup, sets, validation

    @app.get("/api/lineup")
    def lineup() -> dict[str, Any]:
        from innards.lineage.lineup import lineup as get_lineup

        data = get_lineup().to_dict()
        if runner is not None:
            try:
                data["downloaded"] = runner.models().get("downloaded", [])
            except Exception:
                data["downloaded"] = []
        return data

    @app.get("/api/lineup/changed")
    def changed(model: str = Query(min_length=1, max_length=512)) -> dict[str, Any]:
        from innards.lineage.lineup import pair_of, what_changed

        found = pair_of(model)
        if found is None:
            raise HTTPException(404, f"{model} is not in the lineup")
        _pair, role, partner = found
        newer, previous = (model, partner) if role == "newer" else (partner, model)
        result = what_changed(newer, previous)
        result.update(newer_id=newer, previous_id=previous)
        return result

    @app.get("/api/lineup/predicted")
    def predicted(model: str = Query(min_length=1, max_length=512)) -> dict[str, Any]:
        from innards.lineage.lineup import model_config, predicted_view

        if model_config(model) is None:
            raise HTTPException(404, f"{model} is not in the lineup")
        return predicted_view(model)

    @app.get("/api/sets")
    def sets() -> list[dict[str, Any]]:
        from innards.sets import SETS

        return [s.to_dict(with_steps=True) for s in SETS.values()]

    def document_text(doc_id: str | None) -> tuple[str | None, str | None]:
        if not doc_id:
            return None, None
        with documents_lock:
            doc = documents.get(doc_id)
        if doc is None:
            raise HTTPException(404, "Document not found; upload it again")
        return str(doc["text"]), str(doc["name"])

    @app.post("/api/validate")
    def validate(body: ValidateBody) -> dict[str, Any]:
        text, name = document_text(body.document)
        if runner is not None:
            return dict(call(runner.validate, body.model, body.precision, body.set, text))
        from innards.sets import get_set
        from innards.validate import validate as run_checks

        chosen = get_set(body.set, text, name) if body.set else None
        try:
            report = run_checks(body.model, precision=body.precision, device=body.device, conversation_set=chosen)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return report.to_dict()

    @app.post("/api/documents")
    def upload(body: DocumentBody) -> dict[str, Any]:
        from innards.sets import estimate_tokens, load_document

        try:
            data = base64.b64decode(body.data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(422, "data_base64 is not valid base64") from exc
        if len(data) > _MAX_DOCUMENT_BYTES:
            raise HTTPException(413, "Documents are limited to 20 MB")
        try:
            text = load_document(data, body.name)
        except (ValueError, ImportError) as exc:
            raise HTTPException(422, str(exc)) from exc
        tokens = runner.count_tokens(text) if runner is not None else None
        doc_id = uuid.uuid4().hex[:12]
        with documents_lock:
            documents[doc_id] = {"name": body.name, "text": text}
            while len(documents) > 8:
                documents.pop(next(iter(documents)))
        return {
            "id": doc_id,
            "name": body.name,
            "characters": len(text),
            "tokens": tokens if tokens is not None else estimate_tokens(text),
            "exact": tokens is not None,
        }

    # ------------------------------------------------------------------ runner

    @app.post("/api/load")
    def load(body: LoadBody) -> dict[str, Any]:
        text, _name = document_text(body.document)
        return dict(call(need_runner().load, body.model, body.precision, body.device, body.set, text))

    @app.post("/api/unload")
    def unload() -> dict[str, Any]:
        return dict(call(need_runner().unload))

    @app.post("/api/conversation")
    def new_conversation(body: ConversationBody | None = None) -> dict[str, Any]:
        return dict(call(need_runner().new_conversation, body.set if body else None))

    def stream(job: Any) -> StreamingResponse:
        return StreamingResponse(
            job.sse(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        )

    @app.post("/api/chat")
    def chat(body: ChatBody) -> StreamingResponse:
        return stream(call(need_runner().chat, body.message))

    @app.post("/api/set/run")
    def run_set(body: SetBody) -> StreamingResponse:
        text, name = document_text(body.document)
        return stream(call(need_runner().run_set, body.set, text, name, body.steps))

    @app.post("/api/stop")
    def stop() -> dict[str, Literal[True, False]]:
        return {"stopped": bool(need_runner().stop())}

    # ------------------------------------------------------------------ page

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app
