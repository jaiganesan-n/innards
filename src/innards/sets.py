"""Conversation sets: the same input for every model, so a pair compares like with like.

* ``basic_qa``: short, independent questions. Shows baseline KV per token and speed.
* ``with_context``: a chat that plants facts early and asks about them later. Shows KV growth across
  turns, recall and prefix reuse.
* ``input_docs``: your own .txt, .md or .pdf, then questions about it. Shows a large prefill, jumps in
  context and KV, and window and state effects.
* ``long_document``: one long built-in document, then questions about it. Shows the capacity limit and
  how sliding, global and recurrent layers each behave.

Every set is checked against the model's context window and memory before it runs
(:func:`check_set`), and uploaded documents are counted in tokens before they are sent.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = [
    "SETS",
    "ConversationSet",
    "count_tokens",
    "estimate_tokens",
    "get_set",
    "list_sets",
    "load_document",
    "long_document",
]

DOCUMENT_TYPES = (".txt", ".md", ".markdown", ".pdf")


@dataclass(frozen=True)
class ConversationSet:
    name: str
    title: str
    contents: str
    shows: str
    steps: tuple[str, ...]
    needs_document: bool = False
    document: str | None = None
    document_name: str | None = None
    facts: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """``(question, expected answer)`` pairs the set plants, for checking recall by eye."""

    def messages(self) -> list[str]:
        """The user messages, in order (the document, when there is one, goes into the first)."""
        if self.needs_document and not self.document:
            raise ValueError(f"The {self.title!r} set needs a document: upload a .txt, .md or .pdf")
        return list(self.steps)

    def with_document(self, text: str, name: str | None = None) -> ConversationSet:
        first = (
            f"Here is a document{f' ({name})' if name else ''}. Read it; I will ask you questions about it.\n\n"
            f"{text.strip()}"
        )
        return replace(self, document=text, document_name=name, steps=(first, *self.steps[1:]))

    def to_dict(self, *, with_steps: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "contents": self.contents,
            "shows": self.shows,
            "turns": len(self.steps),
            "needs_document": self.needs_document,
            "document_name": self.document_name,
            "estimated_tokens": estimate_tokens("\n".join(self.steps)),
        }
        if with_steps:
            data["steps"] = [s if len(s) < 2000 else s[:2000] + f"… ({len(s):,} characters)" for s in self.steps]
        return data


# --------------------------------------------------------------------------- the built-in long document

_WEATHER = ("clear", "overcast", "foggy", "windy", "calm", "drizzly", "stormy", "bright")
_TASKS = (
    "polished the brass fittings of the lantern room",
    "trimmed the wicks and cleaned the soot from the glass",
    "painted the iron railing on the gallery",
    "logged the passing ships and their flags",
    "mended the rain gutter above the store room",
    "checked the fog bell and oiled its hinge",
    "counted the oil casks in the cellar",
    "walked the path to the landing and cleared the stones",
)
_SEEN = (
    "a pair of gannets diving off the north rocks",
    "two fishing boats working the bank at dawn",
    "a steamer heading east with a red funnel",
    "seals resting on the low ledge",
    "a line of cormorants drying their wings",
    "no ships at all for the whole day",
    "a schooner beating against the wind",
    "the ferry running late on its evening crossing",
)
_PLANTED = {
    2: "The new lamp was lit for the first time on the fourteenth of March, and the keeper named it Aster.",
    26: (
        "The relief boat Marigold arrived at midsummer and brought 312 litres of lamp oil, three sacks of flour "
        "and a letter from the harbour master."
    ),
    49: "The last visitor of the year was a surveyor named Ilse Varga, who measured the tower and left on the ferry.",
}
_FACTS = (
    ("What name did the keeper give the new lamp?", "Aster (first lit on the fourteenth of March)"),
    ("What was the relief boat called, and how much lamp oil did it bring?", "Marigold, 312 litres"),
    ("Who was the last visitor of the year?", "Ilse Varga, a surveyor"),
)


def long_document(weeks: int = 52) -> str:
    """A long log with three facts planted at the start, the middle and the end (about 9,000 tokens)."""
    lines = ["The Keeper's Log of Gull Point Light, One Year", ""]
    for week in range(1, weeks + 1):
        weather = _WEATHER[(week * 3) % len(_WEATHER)]
        task = _TASKS[(week * 5) % len(_TASKS)]
        other = _TASKS[(week * 7 + 3) % len(_TASKS)]
        seen = _SEEN[(week * 11) % len(_SEEN)]
        lines.append(f"Week {week}.")
        lines.append(
            f"The weather was mostly {weather}. On Monday the keeper {task}, and on Wednesday {other}. "
            f"From the gallery the keeper saw {seen}. The lamp burned from dusk to dawn every night, and the "
            f"logbook records {12 + (week * 7) % 9} hours of light on the longest night of the week. The wind came "
            f"from the {('north', 'south', 'east', 'west')[week % 4]} for most of the week, and the tide tables "
            "matched the almanac to within ten minutes. Supper was fish, bread and tea, as it is most weeks."
        )
        if week in _PLANTED:
            lines.append(_PLANTED[week])
        lines.append("")
    return "\n".join(lines)


SETS: dict[str, ConversationSet] = {
    "basic_qa": ConversationSet(
        name="basic_qa",
        title="Basic Q&A",
        contents="Short, independent questions",
        shows="Baseline KV per token and speed",
        steps=(
            "What is the capital of Japan?",
            "Name three primary colours.",
            "How many days are there in a leap year?",
            "What does a thermometer measure?",
            "Give one word that rhymes with light.",
            "What is 12 times 12?",
        ),
    ),
    "with_context": ConversationSet(
        name="with_context",
        title="With context",
        contents="Multi-turn chat that plants facts early and asks about them later",
        shows="KV growth across turns, recall, prefix reuse",
        steps=(
            "Hi! My name is Priya. I am planning a trip to Madurai in March, and my budget is 40,000 rupees.",
            "What are two temples I should visit there?",
            "Suggest a simple vegetarian dish I should try.",
            "How should I travel from Chennai to Madurai?",
            "What is my name?",
            "Where am I going, and in which month?",
            "What was my budget for the trip?",
            "Summarise my plans in two sentences.",
        ),
        facts=(
            ("What is my name?", "Priya"),
            ("Where am I going, and in which month?", "Madurai, in March"),
            ("What was my budget for the trip?", "40,000 rupees"),
        ),
    ),
    "input_docs": ConversationSet(
        name="input_docs",
        title="Input docs",
        contents="Your own .txt, .md or .pdf, then questions about it",
        shows="Large prefill, jumps in context and KV, window and state effects",
        steps=(
            "(your document)",
            "Summarise the document in three sentences.",
            "What are the most important facts or numbers it mentions?",
            "What does the document say near its very beginning?",
            "What does it say near its end?",
        ),
        needs_document=True,
    ),
    "long_document": ConversationSet(
        name="long_document",
        title="Long document",
        contents="One long built-in document (a lighthouse keeper's log), then questions about it",
        shows="Capacity limit; how sliding, global and recurrent layers each behave",
        steps=("(the document)", *(q for q, _ in _FACTS), "Summarise the year in two sentences."),
        facts=_FACTS,
    ).with_document(long_document(), "The Keeper's Log of Gull Point Light"),
}


def list_sets() -> list[dict[str, Any]]:
    return [s.to_dict() for s in SETS.values()]


def get_set(name: str, document: str | None = None, document_name: str | None = None) -> ConversationSet:
    """A set by name; ``input_docs`` needs ``document`` (text you loaded with :func:`load_document`)."""
    try:
        chosen = SETS[name]
    except KeyError:
        raise KeyError(f"Unknown conversation set {name!r}; known: {', '.join(SETS)}") from None
    if document is not None:
        chosen = chosen.with_document(document, document_name)
    return chosen


# --------------------------------------------------------------------------- documents and tokens


def load_document(source: str | Path | bytes, name: str | None = None) -> str:
    """Text of a .txt, .md or .pdf file (a path, or the file's bytes with its ``name``).

    PDFs need ``pypdf`` (``pip install "innards[docs]"``); scanned PDFs without a text layer are refused.
    """
    if isinstance(source, bytes):
        data = source
    else:
        path = Path(source)
        name = name or path.name
        data = path.read_bytes()
    suffix = Path(name or "").suffix.lower()
    if suffix not in DOCUMENT_TYPES:
        raise ValueError(f"Documents must be .txt, .md or .pdf, not {suffix or 'a file without an extension'}")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError('Reading PDFs needs pypdf: pip install "innards[docs]"') from exc
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
        if not text:
            raise ValueError("This PDF has no text layer (scanned pages?); export it as text first")
        return text
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the document as text")  # pragma: no cover - latin-1 always decodes


def estimate_tokens(text: str) -> int:
    """About 4 characters per token: a rough count when no tokenizer is at hand."""
    return max(1, len(text) // 4) if text else 0


def count_tokens(
    conversation: ConversationSet,
    tokenizer: Any = None,
    *,
    system: str | None = None,
    reply_tokens: int = 256,
    render: Callable[[list[dict[str, str]]], Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Tokens the set puts in the context, turn by turn, before anything is sent.

    With a tokenizer, each turn's prompt is rendered with its chat template (replies are assumed to
    be ``reply_tokens`` long); without one, characters / 4. Returns per-turn prompt tokens and the
    largest context the set reaches.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system}] if system else []
    per_turn = []
    placeholder = " ".join(["word"] * max(reply_tokens // 1, 1))
    exact = render is not None or (tokenizer is not None and getattr(tokenizer, "chat_template", None))
    for step in conversation.messages():
        messages.append({"role": "user", "content": step})
        if render is not None:
            prompt = len(render(messages))
        elif exact:
            rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if isinstance(rendered, Mapping):  # transformers 5 returns a BatchEncoding
                rendered = rendered["input_ids"]
            prompt = len(rendered)
        elif tokenizer is not None:
            prompt = len(tokenizer("\n".join(m["content"] for m in messages))["input_ids"])
        else:
            prompt = sum(estimate_tokens(m["content"]) + 4 for m in messages)
        per_turn.append(prompt)
        messages.append({"role": "assistant", "content": placeholder})
    peak = (per_turn[-1] + reply_tokens) if per_turn else 0
    return {"per_turn": per_turn, "max_context": peak, "exact": bool(exact), "reply_tokens": reply_tokens}
