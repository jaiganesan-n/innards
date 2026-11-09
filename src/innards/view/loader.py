"""Independent use: ``innards view`` loads an MoE checkpoint itself, so the Conversation tab can talk to it.

::

    innards view                                                    # pick a model on the Conversation tab
    innards view --model ibm-granite/granite-3.1-1b-a400m-instruct   # or load one at start

The checkpoint is loaded as published (``dtype="auto"``) on the best device here (CUDA, then MPS, then
CPU), after the before-download checks of :func:`innards.validate.validate` pass. Its expert trace goes to
``<root>/<run>/experts/<model name>/``, and the turns to the :class:`~innards.view.chat.ChatRunner` beside it.
Loading another model closes this trace and frees the memory first.

Architectures the installed transformers knows load with its own classes. A checkpoint that ships its own
code for an architecture transformers does not know yet (``facebook/MobileMoE``, ``Tiiny/SmallThinker``)
loads only when the server was started with ``innards view --trust-remote-code``; even then, its code
runs only for such checkpoints, never in place of a native class, and it is built on real tensors by
:func:`load_with_own_code` rather than on the meta device.

Package use is different: your code loads the model and calls ``trace.view(chat=True)``.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from innards.view.chat import ChatRunner
from innards.view.runner import RunnerError, TurnJob

__all__ = ["ModelRunner", "catalogue", "load_with_own_code", "moe_models"]

log = logging.getLogger("innards.view")


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _free(device: str | None) -> None:
    gc.collect()
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:  # torch missing, or no such backend
        pass


def _weight_files(model_id: str, revision: str | None = None) -> list[Path]:
    """The checkpoint's safetensors files, in shard order (from the Hub cache or a local folder)."""
    from transformers.utils.hub import cached_file

    index = cached_file(
        model_id, "model.safetensors.index.json", revision=revision, _raise_exceptions_for_missing_entries=False
    )
    if index:
        names = sorted(set(json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    files = []
    for name in names:
        path = cached_file(model_id, name, revision=revision, _raise_exceptions_for_missing_entries=False)
        if not path:
            raise RunnerError(f"{model_id} has no {name}: a checkpoint's own code loads from safetensors weights only")
        files.append(Path(path))
    return files


def load_with_own_code(model_id: str, device: str = "cpu", dtype: Any = "auto", revision: str | None = None) -> Any:
    """Load a checkpoint that ships its own code for an architecture transformers lacks (trust_remote_code).

    ``from_pretrained`` in transformers 5 builds the model on the meta device, loads the stored weights,
    and then fills the buffers a checkpoint does not store (``persistent=False``, such as RoPE frequencies)
    with uninitialised memory, recomputing them only for module types transformers knows. Code written
    for a checkpoint computes those buffers in its own ``__init__``, sometimes from values it reads
    (``facebook/MobileMoE`` calls ``.item()``), which fails on meta tensors or leaves the buffer unset.

    So here the model is built on ``device`` with real tensors by its own ``__init__`` (random
    initialisation skipped), and every parameter and stored buffer is then copied from the checkpoint's
    safetensors files. The output embedding is tied to the input one only when the config says so
    (``tie_word_embeddings``); any other tensor the checkpoint lacks stops the load, so nothing runs on
    weights that were not published.

    Code written for transformers 4 declares tied weights as a list (``_tied_weights_keys =
    ["lm_head.weight"]``: tied to the input embeddings), which transformers 5 cannot read. Such a class is
    built through a subclass that declares no ties, and the tie is then made as transformers 4 made it.
    Nothing in its computation changes. Package use can call this too::

        model = load_with_own_code("facebook/MobileMoE-S-Base", device="mps")
    """
    import torch
    from transformers import AutoConfig, GenerationConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.initialization import no_init_weights

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    ref = (getattr(config, "auto_map", None) or {}).get("AutoModelForCausalLM")
    if not ref:
        raise RunnerError(f"{model_id}'s config names no AutoModelForCausalLM code (auto_map)")
    cls: Any = get_class_from_dynamic_module(ref, model_id, revision=revision)
    if isinstance(getattr(cls, "_tied_weights_keys", None), (list, tuple)):  # the transformers 4 form
        cls = type(cls.__name__, (cls,), {"_tied_weights_keys": None, "__module__": cls.__module__})
    if dtype == "auto":
        dtype = getattr(config, "dtype", None) or torch.float32
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""))
    files = _weight_files(model_id, revision)
    with torch.device(device), no_init_weights():
        model: Any = cls._from_config(config, dtype=dtype)
    wanted = model.state_dict(keep_vars=True)  # parameters and stored buffers; tied names share one tensor
    loaded: set[str] = set()
    ignored: list[str] = []
    from safetensors.torch import load_file

    with torch.no_grad():
        for path in files:
            shard = load_file(str(path))
            result = model.load_state_dict(shard, strict=False)  # copies in place, through the code's own hooks
            ignored += result.unexpected_keys
            loaded |= set(shard) - set(result.unexpected_keys)
            del shard
    have = {id(wanted[name]) for name in loaded if name in wanted}
    missing = [name for name, tensor in wanted.items() if name not in loaded and id(tensor) not in have]
    out, inp = model.get_output_embeddings(), model.get_input_embeddings()
    if missing and getattr(config, "tie_word_embeddings", False) and out is not None and inp is not None:
        name = next((n for n, t in wanted.items() if t is out.weight), None)
        if name in missing and id(inp.weight) in have:
            out.weight = inp.weight  # what tie_word_embeddings means: the output projection is the embedding
            missing.remove(name)
    if missing:
        listed = ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
        raise RunnerError(
            f"{model_id}: the checkpoint has no weights for {len(missing)} tensors its code defines ({listed}); "
            "Innards does not run a model on weights that were not published"
        )
    if ignored:
        log.info("innards: %s: %d checkpoint tensors its code does not use", model_id, len(ignored))
    with contextlib.suppress(OSError):  # no generation_config.json: the defaults derived from the config stay
        model.generation_config = GenerationConfig.from_pretrained(model_id, revision=revision or "main")
    return model.eval()


def catalogue() -> dict[str, Any]:
    """MoE checkpoints the view offers before they are downloaded (``moe_models.json``).

    Original releases at 16 or 32 bit, checked with the transformers version the file names: transformers
    builds each natively, Innards finds its MoE layers, and it passes the before-download checks for a
    16 GB machine. ``scripts/moe_models.py`` refreshes it when transformers moves on.
    """
    from importlib import resources

    data: dict[str, Any] = json.loads(resources.files(__package__).joinpath("moe_models.json").read_text("utf-8"))
    return data


def _native(model_type: str) -> bool | None:
    """Whether the installed transformers builds ``model_type`` with its own classes (None: no transformers)."""
    try:
        from transformers import CONFIG_MAPPING
    except ImportError:
        return None
    return model_type in CONFIG_MAPPING


def moe_models(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """The page's model list: MoE checkpoints that load here, downloaded ones first.

    Only architectures the installed transformers builds with its own classes are listed: a checkpoint
    that only runs with its own code (``facebook/MobileMoE``) is left out even when it is downloaded.
    Downloaded checkpoints come first, then the catalogue's not yet downloaded, newest release first. A
    cache entry without weights (only its config was read, e.g. by a before-download check) is listed
    only when the catalogue names it.
    """
    from huggingface_hub import scan_cache_dir

    from innards.experts.routing import routing_facts
    from innards.features.base import text_config

    listed = catalogue()
    out: dict[str, dict[str, Any]] = {}
    try:
        repos = scan_cache_dir(cache_dir).repos  # None: the Hugging Face cache in use
    except Exception:  # no cache yet
        repos = frozenset()
    for repo in repos:
        if repo.repo_type != "model" or not repo.revisions:
            continue
        snapshot = Path(max(repo.revisions, key=lambda r: r.last_modified).snapshot_path)
        if not any(p.suffix in {".safetensors", ".bin"} for p in snapshot.iterdir()):
            continue  # no weights here: only its config was read
        try:
            config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_type = str(config.get("model_type") or "")
        facts = routing_facts(text_config(config))
        if not facts.get("experts") or facts["experts"] < 2 or not _native(model_type):
            continue  # not MoE, or transformers here cannot build it
        template: bool | None = True if (snapshot / "chat_template.jinja").is_file() else None
        if template is None:  # None: the tokenizer files are not in the cache, so it is not known yet
            try:
                tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
                template = bool(tokenizer.get("chat_template"))
            except (OSError, json.JSONDecodeError):
                template = None
        out[repo.repo_id] = {
            "id": repo.repo_id,
            "model_type": model_type,
            "experts": facts["experts"],
            "top_k": facts.get("top_k"),
            "chat_template": template,
            "downloaded": True,
            "size_bytes": repo.size_on_disk,
            "released": None,
        }
    for entry in listed["models"]:
        if entry["id"] in out:
            out[entry["id"]]["released"] = entry.get("released")
            continue
        if not _native(entry["model_type"]):  # checked with a newer transformers than the one here
            continue
        out[entry["id"]] = {
            "id": entry["id"],
            "model_type": entry["model_type"],
            "experts": entry["experts"],
            "top_k": entry.get("top_k"),
            "chat_template": entry.get("chat_template"),
            "downloaded": False,
            "size_bytes": int(entry["weights_gb"] * 1e9),
            "released": entry.get("released"),
            "checked_with": listed.get("transformers"),
        }

    def order(m: dict[str, Any]) -> tuple[Any, ...]:
        if m["downloaded"]:  # ready to load: instruct checkpoints first
            return (0, m["chat_template"] is not True, m["id"].lower())
        return (1, -int((m["released"] or "0000-00-00").replace("-", "")), m["id"].lower())  # newest first

    return sorted(out.values(), key=order)


class ModelRunner:
    """Loads one MoE checkpoint at a time and talks to it through a :class:`ChatRunner`."""

    name = "innards"

    def __init__(
        self,
        root: str | Path,
        run: str = "chat",
        *,
        topic: str = "chat",
        generate: Mapping[str, Any] | None = None,
        device: str | None = None,
        trust_remote_code: bool = False,
        budget_gb: float | None = None,
    ) -> None:
        self.root = Path(root)
        self.run = run
        self.topic = topic
        self.generate = dict(generate or {})
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.budget_gb = budget_gb
        self._lock = threading.Lock()
        self._chat: ChatRunner | None = None
        self._state = "idle"
        self._model: str | None = None
        self._error: str | None = None
        self._checks: list[dict[str, Any]] = []
        self._on: str | None = None  # the device the model is on
        self._remote = False  # the loaded model runs the checkpoint's own code

    # ------------------------------------------------------------------ loading

    def models(self) -> list[dict[str, Any]]:
        return moe_models()

    def load(self, model: str) -> dict[str, Any]:
        model = model.strip()
        if not model:
            raise RunnerError("Name a model: a Hub id or a local folder", status=422)
        with self._lock:
            if self._state == "loading":
                raise RunnerError(f"{self._model} is still loading")
            if self._chat is not None and self._chat.status()["state"] == "generating":
                raise RunnerError("A reply is being generated; stop it first")
            self._state, self._model, self._error, self._checks = "loading", model, None, []
        threading.Thread(target=self._load, args=(model,), name="innards-load", daemon=True).start()
        return self.status()

    def _release(self) -> None:
        chat, self._chat, self._remote = self._chat, None, False
        if chat is not None:
            chat.trace.close()  # removes the hooks and writes what was measured
            del chat
        _free(self._on)

    def _load(self, model_id: str) -> None:
        try:
            self._release()
            self._load_now(model_id)
            self._state = "ready"
            return
        except RunnerError as exc:  # a check said no: the reason is the whole story
            log.warning("innards: could not load %s: %s", model_id, exc)
            failed = str(exc)
        except Exception as exc:
            log.warning("innards: could not load %s", model_id, exc_info=True)
            failed = f"{type(exc).__name__}: {exc}"
        self._error, self._state = failed, "error"
        _free(self._on)  # after the except block, so the traceback no longer holds the half-loaded model

    def _load_now(self, model_id: str) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RunnerError('Loading a model needs torch and transformers: pip install "innards[hf,view]"') from exc
        from innards.experts import trace
        from innards.validate import needs_remote_code, validate

        device = self.device or _device()
        remote = self.trust_remote_code  # when the config cannot be read first, from_pretrained decides
        try:
            report = validate(
                model_id,
                device=device,
                budget_gb=self.budget_gb,
                check_access=False,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as exc:  # offline, or no config to read: the load itself will say what is wrong
            log.info("innards: before-download checks skipped for %s (%s)", model_id, exc)
        else:
            self._checks = [c.to_dict() for c in report.checks]
            if report.blocked:
                raise RunnerError(
                    " ".join(f"{c.title}: {c.message}" + (f" {c.fix}" if c.fix else "") for c in report.blocked)
                )
            if report.config is not None:
                remote = self.trust_remote_code and needs_remote_code(report.config)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=remote)
        model: Any
        if remote:  # its own code, built on real tensors (see load_with_own_code)
            model = load_with_own_code(model_id, device)
        else:  # the original checkpoint, at its published precision
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
            model = model.to(device).eval()
        self._on, self._remote = device, remote
        try:
            traced = trace(model, tokenizer, run=self.run, root=str(self.root), model_id=model_id)
        except ValueError as exc:
            raise RunnerError(f"{model_id} is not an MoE model Innards can trace: {exc}") from None
        try:
            self._chat = ChatRunner(traced, generate=self.generate, topic=self.topic)
        except ValueError:
            traced.close()
            raise
        traced.flush()  # the MoE layers tab lists the trace at once

    def unload(self) -> dict[str, Any]:
        with self._lock:
            if self._state == "loading":
                raise RunnerError(f"{self._model} is still loading")
            self._release()
            self._state, self._model, self._error, self._checks = "idle", None, None, []
        return self.status()

    # ------------------------------------------------------------------ Runner protocol

    def status(self) -> dict[str, Any]:
        base = {
            "name": self.name,
            "loadable": True,
            "device": self._on,
            "checks": self._checks,
            "trust_remote_code": self.trust_remote_code,
            "remote_code": self._remote,
        }
        chat = self._chat
        if self._state == "ready" and chat is not None:
            return {**chat.status(), **base}
        return {
            **base,
            "state": self._state,
            "error": self._error,
            "model": self._model,
            "mode": None,
            "trace": None,
            "conversation": None,
            "turns": 0,
            "topic": self.topic,
            "job": None,
        }

    def _need(self) -> ChatRunner:
        if self._chat is None or self._state != "ready":
            raise RunnerError("Load an MoE model first" if self._state != "loading" else f"{self._model} is loading")
        return self._chat

    def new_conversation(self) -> dict[str, Any]:
        self._need().new_conversation()
        return self.status()

    def chat(self, message: str, topic: str | None = None) -> TurnJob:
        return self._need().chat(message, topic)

    def stop(self) -> bool:
        return self._chat.stop() if self._chat is not None else False

    def shutdown(self) -> None:
        """Close the trace when the server stops, so everything measured is written (safe to call twice)."""
        chat = self._chat
        if chat is not None:
            chat.stop()
            chat.trace.close()
