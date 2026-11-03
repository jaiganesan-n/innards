"""Every code example in README.md runs.

Each fenced block is preceded by an optional ``<!-- test: ... -->`` comment (invisible on
PyPI) that declares what it needs:

* ``requires=hf,network``: skipped without the [hf] extra or the Hugging Face Hub
  (unless INNARDS_REQUIRE_NETWORK=1, in which case a missing Hub fails the test).
* ``check-output``: every ``print(...)  # value`` comment must appear in the output.
* ``cli``: each ``innards ...`` line runs as a command.
* ``install``: pip/uv install lines; the CI wheel-install job runs these in a fresh environment.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import hf_available, hub_reachable

README = Path(__file__).parents[1] / "README.md"
_BLOCK = re.compile(r"(?:<!-- test: (?P<meta>[^>]*?) -->\n)?```(?P<lang>\w+)\n(?P<code>.*?)```", re.S)


@dataclass
class Example:
    index: int
    line: int
    lang: str
    code: str
    meta: str

    @property
    def requires(self) -> set[str]:
        match = re.search(r"requires=([\w,]+)", self.meta)
        return set(match.group(1).split(",")) if match else set()

    @property
    def id(self) -> str:
        return f"L{self.line}-{self.lang}"


def examples() -> list[Example]:
    text = README.read_text(encoding="utf-8")
    out = []
    for i, m in enumerate(_BLOCK.finditer(text)):
        line = text[: m.start("code")].count("\n")
        out.append(Example(i, line, m.group("lang"), m.group("code"), m.group("meta") or ""))
    return out


ALL = examples()
PYTHON = [e for e in ALL if e.lang == "python"]
CLI = [e for e in ALL if e.lang == "bash" and "cli" in e.meta.split()]


def _skip_unless_available(example: Example) -> None:
    if "hf" in example.requires and not hf_available():
        pytest.skip("needs the [hf] extra")
    if "network" in example.requires and not hub_reachable() and os.environ.get("INNARDS_REQUIRE_NETWORK") != "1":
        pytest.skip("Hugging Face Hub is unreachable")


def test_every_block_is_accounted_for() -> None:
    langs = {e.lang for e in ALL}
    assert langs <= {"python", "bash", "text"}, langs
    for e in ALL:
        if e.lang == "bash":
            assert {"cli", "install"} & set(e.meta.split()), f"README line {e.line}: bash block without a test tag"


@pytest.mark.parametrize("example", PYTHON, ids=[e.id for e in PYTHON])
def test_python_example(example: Example, tmp_path: Path) -> None:
    _skip_unless_available(example)
    result = subprocess.run(
        [sys.executable, "-c", example.code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, f"README line {example.line} failed:\n{result.stderr[-4000:]}"
    if "check-output" in example.meta.split():
        for comment in re.findall(r"^\s*print\(.*\)\s+# (.+)$", example.code, re.M):
            assert comment.strip() in result.stdout, f"README line {example.line}: {comment!r} not in output"


def _cli_commands() -> list[tuple[int, str]]:
    commands = []
    for e in CLI:
        for offset, raw in enumerate(e.code.splitlines()):
            command = raw.split("  #", 1)[0].strip()
            if command.startswith("innards "):
                commands.append((e.line + offset + 1, command))
    return commands


@pytest.mark.parametrize(("line", "command"), _cli_commands(), ids=[f"L{n}" for n, _ in _cli_commands()])
def test_cli_example(line: int, command: str, tmp_path: Path) -> None:
    args = shlex.split(command)[1:]
    uses_hub = any("/" in a or a == "gpt2" for a in args)
    needs = {"network"} if uses_hub else set()
    if args[0] == "validate":  # validate blocks (exit 1) when transformers is missing
        needs.add("hf")
    _skip_unless_available(Example(0, line, "bash", command, "requires=" + ",".join(sorted(needs))))
    result = subprocess.run(
        [sys.executable, "-m", "innards", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, f"README line {line} `{command}` failed:\n{result.stderr[-4000:]}"
