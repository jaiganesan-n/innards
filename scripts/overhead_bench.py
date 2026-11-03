"""Innards overhead benchmark: the same model and prompt, plain ``generate`` vs observed.

    uv run python scripts/overhead_bench.py --model Qwen/Qwen3.5-0.8B --runs 20
    uv run python scripts/overhead_bench.py --model Qwen/Qwen3-0.6B --modes off,light --device mps
    uv run python scripts/overhead_bench.py --replica qwen3_5 --runs 10     # offline, tiny random weights

Modes (one model object, loaded once):

* ``off``: plain ``model.generate``: what your code does without Innards.
* ``light``: ``innards.observe(model, features="light")``: the light tier only (KV, context, memory,
  timing). This is the tier held to the 1% budget.
* ``probes``: ``innards.observe(model)``: the light tier plus every feature probe the model has.

Every run prefills the same prompt from an empty cache and generates exactly ``--max-new-tokens``
tokens (greedy). The same timestamping streamer measures every mode, so the numbers are taken at the
same points. Runs are interleaved (off, light, probes, then reversed) after warm-up; medians are compared.

``--fail-over 1.0`` exits non-zero if the light tier's median decode throughput drops by more than 1%.
Each observed run also reports Innards' own measured overhead (``observer.overhead_pct``).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROMPT = (
    "The key-value cache stores the attention keys and values of every previous token so that "
    "decoding does not recompute them. Explain, step by step, how grouped-query attention, "
    "multi-head latent attention and sliding-window attention each change the size of that cache, "
    "and what each design gives up in return."
)


@dataclass
class Run:
    mode: str
    ttft_ms: float
    decode_tps: float
    e2e_tps: float
    generated: int
    innards_overhead_pct: float | None = None


class _TimingStreamer:
    """Timestamps only (what any streaming app already does)."""

    def __init__(self) -> None:
        self.t_first: int | None = None
        self.t_last: int | None = None
        self.count = 0
        self._prompt = False

    def put(self, value: Any) -> None:
        now = time.perf_counter_ns()
        if not self._prompt:
            self._prompt = True
            return
        if self.t_first is None:
            self.t_first = now
        self.t_last = now
        self.count += value.numel()

    def end(self) -> None:
        pass


def load(args: argparse.Namespace) -> tuple[Any, Any, str]:
    import torch
    import transformers

    if args.replica:
        sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
        from replicas import replica, tokenizer

        return replica(args.replica).to(args.device), tokenizer(), f"replica/{args.replica}"
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    return model.to(args.device).eval(), tok, args.model


def prompt_ids(tok: Any, device: str) -> dict[str, Any]:
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
        )
        return dict(tok(text, return_tensors="pt", add_special_tokens=False).to(device))
    return dict(tok(PROMPT, return_tensors="pt").to(device))


def run(mode: str, model: Any, tok: Any, model_id: str, inputs: dict[str, Any], max_new: int) -> Run:
    import torch

    import innards

    watch = None
    if mode != "off":
        watch = innards.observe(
            model, tok, model_id=model_id, features="light" if mode == "light" else "auto", store=False
        )
    streamer = _TimingStreamer()
    try:
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new, min_new_tokens=max_new, do_sample=False, streamer=streamer)
        t_end = time.perf_counter_ns()
        overhead = None
        if watch is not None:
            watch.wait()
            overhead = watch.records[-1].observer.overhead_pct
    finally:
        if watch is not None:
            watch.close()
    assert streamer.t_first is not None
    assert streamer.t_last is not None
    n = streamer.count
    return Run(
        mode,
        (streamer.t_first - t0) / 1e6,
        (n - 1) / ((streamer.t_last - streamer.t_first) / 1e9),
        n / ((t_end - t0) / 1e9),
        n,
        overhead,
    )


def bootstrap_ci(values: list[float], reps: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(reps))
    return medians[int(0.025 * reps)], medians[int(0.975 * reps)]


def summarize(runs: list[Run], modes: list[str]) -> dict[str, Any]:
    by = {m: [r for r in runs if r.mode == m] for m in modes}
    med = {
        m: {
            "ttft_ms": statistics.median(r.ttft_ms for r in rs),
            "decode_tps": statistics.median(r.decode_tps for r in rs),
            "e2e_tps": statistics.median(r.e2e_tps for r in rs),
            "innards_overhead_pct": statistics.median(
                [r.innards_overhead_pct for r in rs if r.innards_overhead_pct is not None] or [0.0]
            ),
        }
        for m, rs in by.items()
    }
    out: dict[str, Any] = {"median": med}
    for m in modes:
        if m == "off":
            continue
        pairs = [(b.decode_tps - a.decode_tps) / a.decode_tps * 100 for a, b in zip(by["off"], by[m], strict=True)]
        ttft = [b.ttft_ms - a.ttft_ms for a, b in zip(by["off"], by[m], strict=True)]
        out[m] = {
            "decode_tps_change_pct": (med[m]["decode_tps"] / med["off"]["decode_tps"] - 1) * 100,
            "e2e_tps_change_pct": (med[m]["e2e_tps"] / med["off"]["e2e_tps"] - 1) * 100,
            "paired_decode_tps_diff_pct": {"median": statistics.median(pairs), "ci95": list(bootstrap_ci(pairs))},
            "paired_ttft_diff_ms": {"median": statistics.median(ttft), "ci95": list(bootstrap_ci(ttft))},
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--replica", default=None, help="Use a tiny random-weight replica (tests/replicas.py) offline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", help="float32, bfloat16 or auto (the checkpoint's)")
    parser.add_argument("--modes", default="off,light,probes")
    parser.add_argument("--runs", type=int, default=20, help="measured runs per mode")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--json", type=Path, default=None, help="write raw runs and summary here")
    parser.add_argument("--fail-over", type=float, default=None, help="exit 1 if light-tier throughput drops more")
    args = parser.parse_args()

    import torch
    import transformers

    import innards

    if args.threads:
        torch.set_num_threads(args.threads)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if modes[0] != "off":
        modes.insert(0, "off")
    model, tok, model_id = load(args)
    inputs = prompt_ids(tok, args.device)
    for _ in range(args.warmup):
        for mode in modes:
            run(mode, model, tok, model_id, inputs, args.max_new_tokens)
    runs: list[Run] = []
    for i in range(args.runs):
        for mode in modes if i % 2 == 0 else modes[::-1]:
            runs.append(run(mode, model, tok, model_id, inputs, args.max_new_tokens))
        print(f"  run {i + 1}/{args.runs}", file=sys.stderr, end="\r")

    summary = summarize(runs, modes)
    env = {
        "model": model_id,
        "device": args.device,
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "innards": innards.__version__,
        "runs_per_mode": args.runs,
        "max_new_tokens": args.max_new_tokens,
    }
    m = summary["median"]
    print(f"\nInnards overhead · {model_id} · {args.device} · {args.runs} runs/mode · {args.max_new_tokens} tokens")
    print(f"{'':20}" + "".join(f"{mode:>12}" for mode in modes))
    for key, label in (
        ("decode_tps", "decode tok/s"),
        ("e2e_tps", "end-to-end tok/s"),
        ("ttft_ms", "TTFT ms"),
        ("innards_overhead_pct", "measured overhead %"),
    ):
        print(f"{label:20}" + "".join(f"{m[mode][key]:>12.2f}" for mode in modes))
    for mode in modes[1:]:
        s = summary[mode]
        ci = s["paired_decode_tps_diff_pct"]["ci95"]
        print(
            f"{mode}: decode {s['decode_tps_change_pct']:+.2f}% (paired median "
            f"{s['paired_decode_tps_diff_pct']['median']:+.2f}%, 95% CI {ci[0]:+.2f} to {ci[1]:+.2f})"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"env": env, "summary": summary, "runs": [asdict(r) for r in runs]}, indent=2))
    if (
        args.fail_over is not None
        and "light" in summary
        and -summary["light"]["decode_tps_change_pct"] > args.fail_over
    ):
        print("overhead budget exceeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
