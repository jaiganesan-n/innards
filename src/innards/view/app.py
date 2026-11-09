"""The web view's FastAPI app: two tabs, Conversation and MoE layers, over the expert traces of a run folder.

Read-only (always):

    GET  /healthz                                     liveness
    GET  /api/info                                    version, run folder, whether the page can chat
    GET  /api/state                                   the runner: model, trace, conversation, reply in progress
    GET  /api/experts                                 MoE expert traces in the run folder
    GET  /api/experts/{run}/{name}                    the MoE layers tab (?topic=&by=&layer=&coverage=)
    GET  /api/experts/{run}/{name}/selection          the expert set a topic needs (?topic=&by=&coverage=), JSON
    GET  /api/experts/{run}/{name}/conversations      the Conversation tab's stored turns
    GET  /api/experts/{run}/{name}/sequences          recorded sequences (?topic=); needs torch
    GET  /api/experts/{run}/{name}/sequence/{seq}     one sequence's tokens x experts (?layer=&first=&last=); torch
    GET  /api/experts/{run}/{name}/expert             the tokens one expert takes (?layer=&expert=&topic=); torch

With a runner, which sends messages to a traced model (``trace.view(chat=True)`` from your code, or
``innards view``, which loads the model itself):

    POST /api/conversation                            a new conversation
    POST /api/chat        {message, topic}            → text/event-stream
    POST /api/stop

With ``innards view`` (a runner that loads models):

    GET  /api/models                                  MoE checkpoints in the local Hugging Face cache
    POST /api/load        {model}                     checks, then loads in the background (see /api/state)
    POST /api/unload

Every ``POST`` needs an ``X-Innards: 1`` header (it forces a CORS preflight, so other web pages cannot
drive the view). Set ``INNARDS_VIEW_TOKEN`` to require ``Authorization: Bearer <token>`` on ``/api``;
without a token, requests addressed to anything but a loopback host are refused. On a remote machine,
reach the view through an SSH port forward (``ssh -L 8000:127.0.0.1:8000 ...``).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
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


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadBody(_Strict):
    model: str = Field(min_length=1, max_length=512)


class ChatBody(_Strict):
    message: str = Field(min_length=1, max_length=400_000)
    topic: str | None = Field(default=None, min_length=1, max_length=512)


def _host(request: Request) -> str:
    host = request.headers.get("host", "")
    if host.startswith("["):
        return host.split("]")[0] + "]"
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def create_app(store: RunStore | None = None, *, runner: Runner | None = None, token: str | None = None) -> FastAPI:
    """Build the app over ``store``. ``runner`` lets the Conversation tab send messages."""
    store = store or RunStore()
    token = token if token is not None else os.environ.get("INNARDS_VIEW_TOKEN") or None

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
        elif path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"  # revalidate: a page from an older Innards breaks
        elif path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def need_runner() -> Runner:
        if runner is None:
            raise HTTPException(
                503,
                "This view shows stored traces only. To chat with a model, run innards view, or call "
                "trace.view(chat=True) from the script that traced it.",
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
        return {
            "innards": __version__,
            "store": store.url,
            "chat": runner is not None,
            "load": callable(getattr(runner, "load", None)),
            "auth": bool(token),
        }

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return {"runner": runner.status() if runner is not None else None}

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

    @app.get("/api/experts/{run}/{name}/conversations")
    def stored_conversations(run: str, name: str) -> list[dict[str, Any]]:
        from innards.view.experts import conversations

        return conversations(trace_folder(run, name))

    @app.get("/api/experts/{run}/{name}/sequences")
    def sequences(run: str, name: str, topic: str | None = Query(None, max_length=512)) -> list[dict[str, Any]]:
        from innards.experts import TokenRouting

        folder = trace_folder(run, name)
        return list(experts_call(lambda: TokenRouting(folder).sequences(topic)))

    @app.get("/api/experts/{run}/{name}/sequence/{sequence}")
    def sequence(
        run: str,
        name: str,
        sequence: int,
        layer: int,
        first: int | None = Query(None, ge=0),
        last: int | None = Query(None, ge=1),
    ) -> dict[str, Any]:
        from innards.view.experts import sequence_section

        return dict(experts_call(sequence_section, trace_folder(run, name), sequence, layer, first, last))

    @app.get("/api/experts/{run}/{name}/expert")
    def expert(
        run: str, name: str, layer: int, expert: int = Query(ge=0), topic: str | None = Query(None, max_length=512)
    ) -> dict[str, Any]:
        from innards.view.experts import expert_tokens

        return dict(experts_call(expert_tokens, trace_folder(run, name), layer, expert, topic))

    # ------------------------------------------------------------------ conversation (runner)

    def need_loader() -> Any:
        if not callable(getattr(runner, "load", None)):
            raise HTTPException(
                409,
                "This page talks to the model your code loaded (package use). To pick a model here, run innards view.",
            )
        return runner

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        listed = getattr(runner, "models", None)
        return list(listed()) if callable(listed) else []

    @app.post("/api/load")
    def load(body: LoadBody) -> dict[str, Any]:
        return dict(call(need_loader().load, body.model))

    @app.post("/api/unload")
    def unload() -> dict[str, Any]:
        return dict(call(need_loader().unload))

    @app.post("/api/conversation")
    def new_conversation() -> dict[str, Any]:
        return dict(call(need_runner().new_conversation))

    @app.post("/api/chat")
    def chat(body: ChatBody) -> StreamingResponse:
        job = call(need_runner().chat, body.message, body.topic)
        return StreamingResponse(
            job.sse(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        )

    @app.post("/api/stop")
    def stop() -> dict[str, Literal[True, False]]:
        return {"stopped": bool(need_runner().stop())}

    # ------------------------------------------------------------------ page

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        # The script and stylesheet URLs carry a hash of their content, so a browser that cached the files
        # of an older Innards fetches the new ones instead of running old code against this page.
        page = (STATIC / "index.html").read_text(encoding="utf-8")
        for name in ("app.js", "style.css"):
            digest = hashlib.sha256((STATIC / name).read_bytes()).hexdigest()[:12]
            page = page.replace(f'"/static/{name}"', f'"/static/{name}?v={digest}"')
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    return app
