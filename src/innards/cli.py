"""Command line.

innards view [folder] [--model M]             chat with an MoE model and see its experts per layer (web view)
innards validate <model> [--device --set]     the before-download checks for this machine
innards predict --model <id> --context <n>    KV size from the config, no weights
innards explain <family | model>              what an architecture changed from its predecessor
innards lineup                                the pairs and what each newer model changed
innards sync up <run> s3://bucket/prefix      copy a run folder to S3 (and ``sync down`` back)
innards experts [folder] [--topic T]          MoE routing traces: top experts and the set a topic needs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from typing import Any

from innards import __version__

__all__ = ["main"]

_UNITS = {"": 1, "b": 1, "k": 1000, "kb": 1000, "m": 10**6, "mb": 10**6, "g": 10**9, "gb": 10**9, "t": 10**12}
_UNITS |= {"tb": 10**12, "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}


def parse_bytes(text: str) -> int:
    """``"16GiB"``, ``"8 GB"``, ``"512m"`` or a plain integer."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([a-zA-Z]*)\s*", text)
    if not match or match.group(2).lower() not in _UNITS:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (try 16GiB, 8GB or 1073741824)")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="innards", description="See inside open-weight models: KV cache, context growth and memory per message."
    )
    parser.add_argument("--version", action="version", version=f"innards {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    view = sub.add_parser(
        "view", help="Web view: chat with an MoE model (Conversation) and see its experts per layer (MoE layers)"
    )
    view.add_argument(
        "root", nargs="?", default=None, help="Folder of runs, or one run folder (default ./innards_runs)"
    )
    view.add_argument("--model", default=None, help="MoE model to load at start (Hub id or local folder)")
    view.add_argument("--run", default="chat", help="Run the chat's expert trace goes to (default chat)")
    view.add_argument("--topic", default="chat", help="Topic of the first turns; change it on the page")
    view.add_argument("--max-new-tokens", type=int, default=512, help="Longest reply, in tokens (default 512)")
    view.add_argument("--device", default=None, help="cuda, mps or cpu (default: the best available)")
    view.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Also load checkpoints that ship their own code for an architecture transformers lacks "
        "(e.g. facebook/MobileMoE); native architectures always use transformers' code",
    )
    view.add_argument("--host", default="127.0.0.1", help="Address to bind (default 127.0.0.1; use an SSH tunnel)")
    view.add_argument("--port", type=int, default=8000)
    view.add_argument("--open", action="store_true", help="Open the page in a browser")

    validate = sub.add_parser("validate", help="Check a model against this machine before downloading it")
    validate.add_argument("model", help="Hub repo id or local folder")
    validate.add_argument("--device", default=None, help="cuda, mps or cpu (default: the best available)")
    validate.add_argument("--precision", default=None, help="auto (published), bf16, fp16, fp32 ...")
    validate.add_argument("--set", default=None, help="Conversation set that sizes the KV check")
    validate.add_argument(
        "--budget-gb", type=float, default=None, help="Memory budget (default 16 or $INNARDS_GPU_BUDGET_GB)"
    )
    validate.add_argument("--json", action="store_true")

    lineup = sub.add_parser("lineup", help="The lineup: pairs and what each newer model changed")
    lineup.add_argument("--json", action="store_true")

    sync = sub.add_parser("sync", help="Copy run folders to or from S3 (needs innards[aws])")
    sync_sub = sync.add_subparsers(dest="direction", required=True)
    up = sync_sub.add_parser("up", help="Upload a run folder: innards sync up <run> s3://bucket/prefix")
    up.add_argument("run")
    up.add_argument("url")
    up.add_argument("--root", default=None, help="Folder of runs (default ./innards_runs)")
    down = sync_sub.add_parser("down", help="Download runs: innards sync down s3://bucket/prefix[/run]")
    down.add_argument("url")
    down.add_argument("--root", default=None, help="Folder of runs (default ./innards_runs)")

    predict = sub.add_parser("predict", help="Predict KV-cache size from a model config (no weights)")
    predict.add_argument("--model", required=True, help="Hub repo id, local path or config.json")
    predict.add_argument("--context", type=int, required=True, help="Context length in tokens")
    predict.add_argument(
        "--kv-dtype", default="float16", help="float32, float16, bfloat16 or float8 (default: float16)"
    )
    predict.add_argument("--memory", type=parse_bytes, default=None, help="KV memory budget for capacity, e.g. 16GiB")
    predict.add_argument("--batch-size", type=int, default=1)
    predict.add_argument("--revision", default=None, help="Hub revision")
    predict.add_argument("--json", action="store_true", help="Print the prediction as JSON")

    experts = sub.add_parser("experts", help="MoE routing traces: which experts a topic or token activates")
    experts.add_argument(
        "path", nargs="?", default=None, help="Folder of runs, a run folder or a trace folder (default ./innards_runs)"
    )
    experts.add_argument("--topic", default=None, help="Topic to show (default: list the traces and topics)")
    experts.add_argument("--by", default="selection", choices=["selection", "gate", "lift"], help="Ranking measure")
    experts.add_argument("--layer", type=int, action="append", default=None, help="Only these MoE layers")
    experts.add_argument("--top", type=int, default=8, help="Experts to show per layer (default 8)")
    experts.add_argument(
        "--select", type=float, default=None, metavar="COVERAGE", help="The experts that cover this share, e.g. 0.9"
    )
    experts.add_argument("--out", default=None, help="Write the --select expert set to this JSON file")
    experts.add_argument("--token", default=None, help="Where one token goes, per layer (reads the token shards)")
    experts.add_argument("--json", action="store_true")

    explain = sub.add_parser("explain", help="What an architecture changed from its predecessor")
    explain.add_argument("target", help="Family (mha, gqa, mla, sliding_window, hybrid ...) or a model id/path")
    explain.add_argument("--json", action="store_true")
    return parser


# --------------------------------------------------------------------------- predict / explain


def _cmd_predict(args: argparse.Namespace) -> int:
    from innards.kv import predict
    from innards.lineage import explain
    from innards.sinks.console import format_bytes

    p = predict(
        args.model,
        args.context,
        args.kv_dtype,
        memory_bytes=args.memory,
        batch_size=args.batch_size,
        revision=args.revision,
    )
    if args.json:
        print(json.dumps(p.to_dict(), indent=2))
        return 0
    groups = p.by_layer_type
    attention = f" ({p.attention.upper()} attention)" if p.attention and p.attention != p.architecture else ""
    print(f"{p.model_id or args.model} · {p.architecture}{attention} · KV dtype {p.kv_dtype}")
    print(f"  per token: {p.formula}")
    print(
        f"  at {p.context_tokens:,} tokens: KV {format_bytes(p.kv_bytes)} ({p.kv_bytes:,} B) · "
        f"global {format_bytes(groups['global'])} · sliding {format_bytes(groups['sliding'])} · "
        f"recurrent {format_bytes(p.recurrent_state_bytes)}"
    )
    if p.max_context_tokens:
        line = f"  max context: {p.max_context_tokens:,} tokens"
        if p.memory_bytes is not None:
            line += f" · fits in {format_bytes(p.memory_bytes)}: {p.capacity_tokens:,} tokens"
        print(line)
    entry = explain(p.architecture)
    print(f"  lineage: {entry.family} ← {entry.predecessor or '-'}: {entry.change}")
    for note in p.notes:
        print(f"  note: {note}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from innards.lineage import explain

    entry = explain(args.target)
    print(json.dumps(entry.to_dict(), indent=2) if args.json else str(entry))
    return 0


# --------------------------------------------------------------------------- view / validate / lineup / sync


def _cmd_view(args: argparse.Namespace) -> int:
    from innards.store import RunStore
    from innards.view import serve
    from innards.view.loader import ModelRunner

    store = RunStore(args.root)
    runner = ModelRunner(
        store.root,
        store.only or args.run,
        topic=args.topic,
        generate={"max_new_tokens": args.max_new_tokens},
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    if args.model:
        runner.load(args.model)  # in the background; the page shows it loading
    try:
        serve(root=args.root, runner=runner, host=args.host, port=args.port, open_browser=args.open, block=True)
    except KeyboardInterrupt:
        pass  # Ctrl-C is how the view stops
    finally:
        runner.shutdown()  # writes the loaded model's trace
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from innards.validate import validate

    report = validate(
        args.model, precision=args.precision, device=args.device, conversation_set=args.set, budget_gb=args.budget_gb
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else str(report))
    return 0 if report.ok else 1


def _cmd_lineup(args: argparse.Namespace) -> int:
    from innards.lineage.lineup import lineup, what_changed

    data = lineup()
    if args.json:
        print(json.dumps(data.to_dict(), indent=2))
        return 0
    for pair in data.pairs:
        changed = what_changed(pair.newer, pair.previous)
        added = ", ".join(a["name"] for a in changed["added"]) or "-"
        print(f"{pair.feature}  [{pair.where}]")
        print(f"  {pair.previous}  →  {pair.newer}")
        print(f"  adds: {added}")
        for newer, previous in pair.alternates:
            print(f"  or:   {previous}  →  {newer}")
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    from innards.store import RunStore
    from innards.sync import sync_down, sync_up

    if args.direction == "up":
        result = sync_up(RunStore(args.root), args.run, args.url)
    else:
        result = sync_down(args.url, RunStore(args.root).root)
    print(result)
    return 0


def _pct(value: float | None) -> str:
    return f"{value * 100:5.1f}%" if value is not None else "  n/a "


def _cmd_experts(args: argparse.Namespace) -> int:
    from pathlib import Path

    from innards.experts import TokenRouting, load, traces
    from innards.store import RunStore

    path = Path(args.path) if args.path else RunStore().root
    if args.topic is None and args.token is None:
        found: list[dict[str, Any]]
        if (path / "summary.json").is_file():
            found = [{"run": None, "name": str(path), **_brief(load(path))}]
        elif (path / "experts").is_dir():
            found = [t for t in traces(path.parent) if t["run"] == path.name]
        else:
            found = traces(path)
        if args.json:
            print(json.dumps(found, indent=2))
            return 0
        if not found:
            print(f"No expert traces under {path}")
        for t in found:
            where = f"{t['run']}/experts/{t['name']}" if t.get("run") else t["name"]
            print(f"{where}  ·  {t['model']}  ·  {t['layers']} MoE layers")
            for topic, count in t["tokens"].items():
                print(f"  {topic}: {count:,} tokens")
        return 0
    experts = load(path)
    layers = args.layer or experts.layers
    if args.token is not None:
        folder = experts.folder
        if folder is None:
            raise ValueError("--token reads one trace folder's token shards; pass that folder")
        where = TokenRouting(folder).token(args.token, topic=args.topic)
        if args.json:
            print(json.dumps(where, indent=2))
            return 0
        print(f"{args.token!r}: {where['occurrences']:,} occurrences (measured; ids {where['ids']})")
        for layer, row in zip(where["layers"], where["counts"], strict=True):
            if layer in layers:
                total = where["occurrences"] or 1
                ranked = sorted(range(len(row)), key=lambda e: -row[e])[: args.top]
                print(f"  layer {layer:>3}: " + "  ".join(f"e{e} {row[e] / total:.0%}" for e in ranked if row[e]))
        return 0
    topic = args.topic
    if args.select is not None:
        chosen = experts.select(topic, coverage=args.select, by="gate" if args.by == "gate" else "selection")
        if args.out:
            chosen.save(args.out)
        print(json.dumps(chosen.to_dict(), indent=2) if args.json else chosen)
        if args.out and not args.json:
            print(f"(written to {args.out})")
        return 0
    rows = {layer: experts.top(topic, layer, n=args.top, by=args.by) for layer in layers}
    if args.json:
        print(json.dumps({str(k): v for k, v in rows.items()}, indent=2))
        return 0
    print(f"{experts.model} · {topic}: {experts.tokens(topic):,} tokens measured · ranked by {args.by} (derived)")
    print("  selection = share of the topic's tokens whose top-k included the expert; lift = vs the other topics")
    for layer, top in rows.items():
        cells = "  ".join(
            f"e{r['expert']} {_pct(r['selection']).strip()}"
            + (f" ×{r['lift']:.1f}" if r["lift"] is not None and args.by == "lift" else "")
            for r in top
        )
        print(f"  layer {layer:>3}: {cells}")
    return 0


def _brief(experts: object) -> dict[str, Any]:
    from innards.experts import ExpertMap

    assert isinstance(experts, ExpertMap)
    return {
        "model": experts.model,
        "layers": len(experts.layers),
        "tokens": {t: experts.tokens(t) for t in experts.topics},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    commands = {
        "predict": _cmd_predict,
        "explain": _cmd_explain,
        "view": _cmd_view,
        "validate": _cmd_validate,
        "lineup": _cmd_lineup,
        "sync": _cmd_sync,
        "experts": _cmd_experts,
    }
    try:
        return commands[args.command](args)
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"innards: error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
