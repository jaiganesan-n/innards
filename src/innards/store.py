"""Local run store: results live next to the run, on the machine that runs the model.

Layout (``./innards_runs`` by default, or ``$INNARDS_RUNS``)::

    innards_runs/<run>/run.json                          name, created/updated, host, environment
    innards_runs/<run>/<conversation>/conversation.json  model, features, set, status, turn count
    innards_runs/<run>/<conversation>/turns.jsonl        one TurnRecord (schema 1.0) per turn

A run folder is self-contained: copy it (or sync it with :mod:`innards.sync`) to another machine and
the view opens it there. Turns are appended as they finish, so a view pointed at a running
conversation fills in live. Readers skip a torn last line while a writer is mid-append.
"""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from innards.schema import SCHEMA_VERSION, ConversationRecord, TurnRecord

__all__ = ["DEFAULT_ROOT", "RunStore", "now_iso", "safe_name", "valid_name"]

DEFAULT_ROOT = "innards_runs"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def valid_name(name: str) -> bool:
    """Run and conversation names are path-safe: letters, digits, ``.``, ``_`` and ``-``; no leading dot."""
    return bool(_NAME.match(name)) and ".." not in name


def safe_name(name: str, fallback: str = "run") -> str:
    """Turn any label into a valid name (``"DeltaNet study!"`` → ``"DeltaNet-study"``)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")[:120]
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    return cleaned if cleaned and valid_name(cleaned) else fallback


def _check(*names: str) -> None:
    for name in names:
        if not valid_name(name):
            raise ValueError(f"Invalid run or conversation name {name!r}")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)  # atomic: readers never see a half-written file


def _parse_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:  # a torn last line while a writer is mid-append
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


@dataclass(frozen=True)
class _Location:
    root: Path
    single_run: str | None
    """Set when the store was opened on one run folder rather than a folder of runs."""


def _locate(path: Path) -> _Location:
    """A folder of runs, or one run folder (it holds ``run.json``)."""
    if (path / "run.json").is_file():
        return _Location(path.parent, path.name)
    return _Location(path, None)


class RunStore:
    """Runs under a local folder. Thread-safe within a process; appends are line-atomic.

    ``RunStore(path)`` accepts a folder of runs or a single run folder (one with ``run.json``), so
    ``innards view ./innards_runs/deltanet-study`` opens just that run.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        target = Path(root or os.environ.get("INNARDS_RUNS") or DEFAULT_ROOT).expanduser().resolve()
        location = _locate(target)
        self.root = location.root
        self.only = location.single_run
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"file://{self.root / self.only if self.only else self.root}"

    def run_dir(self, run: str) -> Path:
        _check(run)
        return self.root / run

    # ------------------------------------------------------------------ writing

    def ensure_run(self, run: str, environment: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create ``run.json`` if needed; refresh ``updated_at``. Returns the manifest."""
        path = self.run_dir(run) / "run.json"
        with self._lock:
            manifest = _read_json(path) or {
                "schema_version": SCHEMA_VERSION,
                "name": run,
                "created_at": now_iso(),
                "host": socket.gethostname(),
                "environment": environment or {},
            }
            manifest["updated_at"] = now_iso()
            if environment and not manifest.get("environment"):
                manifest["environment"] = environment
            _write_json(path, manifest)
        return manifest

    def write_conversation(self, run: str, record: ConversationRecord) -> None:
        _check(run, record.id)
        record.updated_at = now_iso()
        with self._lock:
            _write_json(self.root / run / record.id / "conversation.json", record.to_dict())

    def append_turn(self, run: str, conversation: str, record: TurnRecord | dict[str, Any]) -> None:
        _check(run, conversation)
        row = record.to_dict() if isinstance(record, TurnRecord) else record
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        directory = self.root / run / conversation
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            with open(directory / "turns.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    # ------------------------------------------------------------------ reading

    def list_runs(self) -> list[dict[str, Any]]:
        """Run manifests with a summary of their conversations, newest first."""
        if not self.root.is_dir():
            return []
        names = [self.only] if self.only else sorted(p.name for p in self.root.iterdir() if p.is_dir())
        runs = []
        for name in names:
            if not valid_name(name):
                continue
            manifest = _read_json(self.root / name / "run.json")
            if manifest is None:
                continue
            conversations = self.conversations(name)
            manifest["conversations"] = [
                {
                    "id": c.get("id"),
                    "model": (c.get("model") or {}).get("id"),
                    "architecture": (c.get("model") or {}).get("architecture"),
                    "label": c.get("label"),
                    "set": c.get("set"),
                    "status": c.get("status"),
                    "turns": c.get("turns"),
                    "started_at": c.get("started_at"),
                    "updated_at": c.get("updated_at"),
                    "device": (c.get("runtime") or {}).get("device"),
                }
                for c in conversations
            ]
            runs.append(manifest)
        return sorted(runs, key=lambda m: str(m.get("updated_at") or m.get("created_at") or ""), reverse=True)

    def read_run(self, run: str) -> dict[str, Any] | None:
        if not valid_name(run) or (self.only and run != self.only):
            return None
        return _read_json(self.root / run / "run.json")

    def conversations(self, run: str) -> list[dict[str, Any]]:
        """Conversation records of a run, oldest first."""
        if not valid_name(run):
            return []
        directory = self.root / run
        if not directory.is_dir():
            return []
        out = []
        for path in directory.iterdir():
            if path.is_dir() and valid_name(path.name):
                data = _read_json(path / "conversation.json")
                if data is not None:
                    out.append(data)
        return sorted(out, key=lambda c: str(c.get("started_at") or ""))

    def read_conversation(self, run: str, conversation: str) -> dict[str, Any] | None:
        if not (valid_name(run) and valid_name(conversation)):
            return None
        return _read_json(self.root / run / conversation / "conversation.json")

    def turns(self, run: str, conversation: str) -> list[dict[str, Any]]:
        if not (valid_name(run) and valid_name(conversation)):
            return []
        try:
            text = (self.root / run / conversation / "turns.jsonl").read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            return []
        return sorted(_parse_lines(text), key=lambda t: int(t.get("turn") or 0))

    def turn_records(self, run: str, conversation: str) -> list[TurnRecord]:
        return [TurnRecord.from_dict(row) for row in self.turns(run, conversation)]

    def find_conversation(self, conversation: str) -> str | None:
        """The run that holds ``conversation``, if any."""
        for manifest in self.list_runs():
            if any(c.get("id") == conversation for c in manifest.get("conversations") or []):
                return str(manifest["name"])
        return None

    def files(self, run: str) -> Iterable[Path]:
        """Every file of a run folder (for syncing)."""
        directory = self.run_dir(run)
        return (p for p in sorted(directory.rglob("*")) if p.is_file() and not p.name.startswith("."))
