"""The Conversation tab's runner: talk to a traced model, with its expert trace attached.

Package use (your code loads the model)::

    trace = innards.experts.trace(model, tokenizer, run="chat")
    trace.view(chat=True)       # Conversation: talk to the model; MoE layers: what each turn routed

Independent use (``innards view``) loads the model in :mod:`innards.view.loader` and uses this runner too.

This runner calls ``model.generate()`` on the traced model only when a message is sent from the page,
with the model's generation config (plus the ``generate()`` arguments given to ``chat=``). Each turn
runs inside ``trace.topic(topic)``, so its routing is recorded like any other traced forward.

* ``chat`` mode (the tokenizer has a chat template): the conversation is built with that template.
* ``text`` mode (a base model, no chat template): no template is made up. The first message is
  tokenized as the tokenizer does by default (its BOS, if it adds one); each later message is
  appended, exactly as typed, to the tokens so far (earlier messages and replies), and the model
  continues the whole text.

When the model's cache holds attention keys and values only, the next turn reuses it if the new prompt
starts with every token already in it: then each token goes through the MoE layers once and the
conversation is one sequence of the trace. Recurrent or convolution states (Mamba, DeltaNet, conv
layers) are not carried over, and neither is a cache the chat template no longer matches (some
templates drop earlier reasoning): those turns are prefilled from the start, and the page shows 0 tokens
reused.

Every turn is appended to ``<run>/experts/<name>/conversations.jsonl``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from typing import Any

from innards.store import now_iso
from innards.view.runner import RunnerError, TurnJob

__all__ = ["CONVERSATIONS", "ChatRunner"]

log = logging.getLogger("innards.view")

CONVERSATIONS = "conversations.jsonl"
"""The file in a trace folder that holds the Conversation tab's turns, one JSON object per line."""

_RESERVED = {"input_ids", "attention_mask", "past_key_values", "streamer", "return_dict_in_generate"}


def _attention_only(cache: Any) -> bool:
    """True when every layer of ``cache`` holds keys and values only (no recurrent or conv state)."""
    from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin

    layers = getattr(cache, "layers", None) or []
    return bool(layers) and all(
        isinstance(layer, DynamicLayer) and not isinstance(layer, LinearAttentionCacheLayerMixin) for layer in layers
    )


def _streamer(tokenizer: Any, job: TurnJob) -> Any:
    """A transformers streamer that hands each finished piece of text to the page."""
    from transformers import TextStreamer

    class PageStreamer(TextStreamer):
        generated = 0

        def put(self, value: Any) -> None:
            if not self.next_tokens_are_prompt:
                self.generated += int(value.numel())
            super().put(value)  # type: ignore[no-untyped-call]

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            if text:
                job.token(text, self.generated)

    return PageStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


def _cancel(event: threading.Event) -> Any:
    from transformers import StoppingCriteria

    class Cancel(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            import torch

            return torch.full((input_ids.shape[0],), event.is_set(), dtype=torch.bool, device=input_ids.device)

    return Cancel()


def _explain(exc: BaseException, model_id: str) -> str:
    """The error, and when it came through the checkpoint's own code (trust_remote_code), which version
    question to ask: such code is written against one transformers release, and later ones rename things."""
    text = f"{type(exc).__name__}: {exc}"
    if not any("transformers_modules" in frame.filename for frame in traceback.extract_tb(exc.__traceback__)):
        return text
    try:
        import transformers

        version = f"transformers {transformers.__version__}"
    except ImportError:
        version = "this transformers"
    end = "" if text.endswith((".", "?", "!")) else "."
    return (
        f"{model_id}'s own code fails with {version}: {text}{end} Its code was written for another transformers "
        "release (its model card says which); Innards runs a checkpoint's code as published and does not patch it."
    )


class ChatRunner:
    """Talks to the model of an :class:`~innards.experts.ExpertTrace`; one conversation at a time."""

    name = "chat"

    def __init__(self, trace: Any, generate: Mapping[str, Any] | None = None, topic: str = "chat") -> None:
        tokenizer = getattr(trace, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("The Conversation tab needs the tokenizer: innards.experts.trace(model, tokenizer, ...)")
        clash = sorted(_RESERVED & set(generate or {}))
        if clash:
            raise ValueError(f"The Conversation tab sets {', '.join(clash)} itself; leave them out of chat=")
        if not str(topic or "").strip():
            raise ValueError("topic must be a non-empty string")
        self.trace = trace
        self.mode = "chat" if getattr(tokenizer, "chat_template", None) else "text"
        self.generate_kwargs: dict[str, Any] = {"max_new_tokens": 512, **(generate or {})}
        self.topic = str(topic).strip()
        self._busy = threading.Lock()
        self._job: TurnJob | None = None
        self._error: str | None = None
        self._new()

    def _new(self) -> None:
        self.conversation = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        self.history: list[dict[str, str]] = []
        self.turns = 0
        self._cache: Any = None
        self._ids: Any = None  # the last turn's full token ids (prompt + reply), on the CPU

    # ------------------------------------------------------------------ Runner protocol

    def status(self) -> dict[str, Any]:
        job = self._job
        generating = job is not None and not job.done
        return {
            "name": self.name,
            "state": "generating" if generating else "error" if self._error else "ready",
            "error": self._error,
            "model": self.trace.model_id,
            "mode": self.mode,
            "trace": {"run": self.trace.run, "name": self.trace.name},
            "conversation": self.conversation,
            "turns": self.turns,
            "topic": self.topic,
            "job": job.snapshot() if generating and job is not None else None,
        }

    def new_conversation(self) -> dict[str, Any]:
        if not self._busy.acquire(blocking=False):
            raise RunnerError("A reply is being generated; stop it first")
        try:
            self._new()
            self._error = None
        finally:
            self._busy.release()
        return self.status()

    def chat(self, message: str, topic: str | None = None) -> TurnJob:
        label = str(topic if topic is not None else self.topic).strip()
        if not label:
            raise RunnerError("A topic is a non-empty string", status=422)
        if not self._busy.acquire(blocking=False):
            raise RunnerError("A reply is still being generated")
        self.topic = label
        job = self._job = TurnJob("chat", message)
        threading.Thread(target=self._run, args=(job, message, label), name="innards-chat", daemon=True).start()
        return job

    def stop(self) -> bool:
        job = self._job
        if job is None or job.done:
            return False
        job.cancel.set()
        return True

    # ------------------------------------------------------------------ one turn

    def _run(self, job: TurnJob, message: str, topic: str) -> None:
        try:
            job.put("turn", {"record": self._turn(job, message, topic)})
            self._error = None
        except Exception as exc:
            log.warning("innards: the chat turn failed", exc_info=True)
            self._error = _explain(exc, self.trace.model_id)
            job.put("error", {"message": self._error})
        finally:
            job.close()
            self._busy.release()

    def _reused(self, ids: Any) -> int:
        """Tokens of ``ids`` the kept cache already holds (0 when it does not hold exactly that prefix)."""
        import torch

        if self._cache is None or self._ids is None:
            return 0
        cached = int(self._cache.get_seq_length())
        if not 0 < cached < int(ids.shape[0]) or cached > int(self._ids.shape[0]):
            return 0
        return cached if torch.equal(ids[:cached].cpu(), self._ids[:cached]) else 0

    def _turn(self, job: TurnJob, message: str, topic: str) -> dict[str, Any]:
        import torch
        from transformers import StoppingCriteriaList

        trace, tokenizer, model = self.trace, self.trace.tokenizer, self.trace.model
        history = [*self.history, {"role": "user", "content": message}]
        job.turn_start(self.turns + 1, message, topic)
        if self.mode == "chat":
            inputs = tokenizer.apply_chat_template(
                history, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        else:  # a base model: the text so far, then this message as typed
            new = tokenizer(message, add_special_tokens=self._ids is None, return_tensors="pt")["input_ids"][0]
            ids = new if self._ids is None else torch.cat([self._ids, new.to(self._ids.dtype)])
            inputs = {"input_ids": ids[None], "attention_mask": torch.ones_like(ids)[None]}
        device = getattr(model, "device", None) or trace.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt = int(inputs["input_ids"].shape[1])
        reused = self._reused(inputs["input_ids"][0])
        kwargs: dict[str, Any] = {**self.generate_kwargs, **inputs}
        criteria = list(kwargs.pop("stopping_criteria", None) or [])
        kwargs["stopping_criteria"] = StoppingCriteriaList([*criteria, _cancel(job.cancel)])
        kwargs["streamer"] = _streamer(tokenizer, job)
        kwargs["return_dict_in_generate"] = True
        if reused:
            kwargs["past_key_values"] = self._cache
        started = now_iso()
        with trace.topic(topic), torch.no_grad():
            out = model.generate(**kwargs)
        full = out.sequences[0].detach().cpu()
        cache = getattr(out, "past_key_values", None)
        self._cache = cache if cache is not None and _attention_only(cache) else None
        self._ids = full
        reply = tokenizer.decode(full[prompt:], skip_special_tokens=True)
        self.history = [*history, {"role": "assistant", "content": reply}]
        self.turns += 1
        trace.flush()  # the MoE layers tab reads the run folder
        record = {
            "conversation": self.conversation,
            "turn": self.turns,
            "mode": self.mode,
            "topic": topic,
            "message": message,
            "reply": reply,
            "stopped": job.cancel.is_set(),
            "sequence": trace.last_sequence,
            "positions": [reused, int(full.shape[0]) - 1],
            "tokens": {
                "prompt": prompt,
                "reused": reused,
                "prefilled": prompt - reused,
                "generated": int(full.shape[0]) - prompt,
                "routed": int(full.shape[0]) - 1 - reused,
            },
            "model": trace.model_id,
            "started_at": started,
            "finished_at": now_iso(),
        }
        folder = trace.folder
        if folder is not None:
            with (folder / CONVERSATIONS).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
