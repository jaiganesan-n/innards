"""README banner, drawn from a measured expert trace of Granite 3.1 1B-A400M.

    uv run --with playwright python docs/figures/src/make_banner.py --measure   # trace the model, then draw
    uv run --with playwright python docs/figures/src/make_banner.py             # redraw from banner_data.json

``--measure`` loads ``ibm-granite/granite-3.1-1b-a400m-instruct`` in bf16, generates greedily for three
topics under ``innards.experts.trace`` and writes what the banner shows to ``docs/figures/banner_data.json``.
The banner is an SVG rendered by Chrome (Playwright), in the style of the expert-transplant banner.

Left: the model, unchanged; each MoE block shows the experts one real token chose. Centre: one MoE layer
and where the trace reads it. Right: what the trace measured for three topics in that layer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tempfile
from pathlib import Path
from typing import Any

FIGURES = Path(__file__).resolve().parents[1]
DATA = FIGURES / "banner_data.json"
MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
LAYER = 12  # the layer the centre and right panels show
COVERAGE = 0.9
NEW_TOKENS = 48
ROWS = [0, 4, 8, 12, 16, 20, 23]  # the layers the left panel shows
W, H = 1600, 560

TOPICS = {
    "math": [
        "The derivative of x squared is",
        "Solve for x: 3x + 5 = 20. First,",
        "The area of a circle with radius r is",
        "A prime number is a number that",
        "The sum of the angles of a triangle is",
        "To multiply two fractions, you",
    ],
    "code": [
        "def fibonacci(n):\n    if n <",
        "import numpy as np\narr = np.zeros(",
        "SELECT name, age FROM users WHERE",
        "for (int i = 0; i < n; i++) {",
        "class Stack:\n    def __init__(self):",
        "git commit -m",
    ],
    "history": [
        "The Roman Empire fell in",
        "The French Revolution began in 1789 when",
        "The first emperor of China was",
        "World War I started after",
        "The Magna Carta was signed in",
        "The Mughal Empire was founded by",
    ],
}


def measure() -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import innards

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    trace = innards.experts.trace(model, tokenizer, run="banner", root=tempfile.mkdtemp(prefix="innards-banner-"))
    for topic, texts in TOPICS.items():
        for text in texts:
            with trace.topic(topic):
                model.generate(
                    **tokenizer(text, return_tensors="pt").to(device), max_new_tokens=NEW_TOKENS, do_sample=False
                )
    trace.close()

    experts = innards.experts.load(trace.folder)
    routing = innards.experts.TokenRouting(trace.folder)
    li = experts.layers.index(LAYER)
    seq = routing.sequence(routing.sequences("math")[0]["sequence"])
    words = [i for i, t in enumerate(seq["text"]) if t.strip().isalpha()]
    pick = next((i for i in words if seq["text"][i].strip() == "derivative"), words[0])
    selections = {t: experts.select(t, COVERAGE) for t in experts.topics}
    return {
        "measured": dt.date.today().isoformat(),
        "model": MODEL,
        "revision": getattr(model.config, "_commit_hash", None),
        "device": device,
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "innards": innards.__version__,
        "prompts": TOPICS,
        "new_tokens": NEW_TOKENS,
        "moe_layers": len(experts.layers),
        "experts": experts.experts(LAYER),
        "top_k": trace.facts.get("top_k"),
        "tokens": {t: experts.tokens(t) for t in experts.topics},
        "layer": LAYER,
        "coverage": COVERAGE,
        "selection": {t: [round(v or 0.0, 4) for v in experts.selection(t)[li]] for t in experts.topics},
        "keep": {t: s.keep(LAYER) for t, s in selections.items()},
        "kept": {t: [s.kept, s.total] for t, s in selections.items()},
        "token": {"text": seq["text"][pick], "layers": seq["layers"], "experts": seq["experts"][pick]},
    }


# drawing -------------------------------------------------------------------------------------------

THEME: dict[str, Any] = {
    "bg": "#F8F8F8", "edge": "#E1E0D9", "panel": "#FFFFFF", "panel_edge": "#D5D8DC", "ink": "#222222",
    "muted": "#6B7280", "faint": "#9AA0A8", "attn": ("#FFF3B0", "#D9B92B"), "moe": ("#FDE3C8", "#E8892B"),
    "lit": "#E8892B", "hook": ("#D5F0DD", "#2E9E4F"), "arrow": "#3A3F47",
    "heat": ["#FFFFFF", "#FDE3C8", "#F5B26B", "#E8892B", "#A85410"],
}  # fmt: skip


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str, rx: float = 8, sw: float = 1.6,
         dash: str | None = None) -> str:  # fmt: skip
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw}"{d}/>'
    )


def line(d: str, stroke: str, sw: float = 1.8, dash: str | None = None, marker: str | None = None) -> str:
    extra = (f' stroke-dasharray="{dash}"' if dash else "") + (f' marker-end="url(#{marker})"' if marker else "")
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="none"{extra}/>'


def text(x: float, y: float, s: str, size: float, fill: str, weight: int = 400, anchor: str = "start",
         style: str = "") -> str:  # fmt: skip
    return (
        f'<text x="{x}" y="{y}" font-family="Inter, DejaVu Sans, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {style}>{s}</text>'
    )


def grid(x: float, y: float, cols: int, n: int, size: float, gap: float, t: dict[str, Any], lit: set[int]) -> str:
    return "".join(
        rect(x + (e % cols) * (size + gap), y + (e // cols) * (size + gap), size, size,
             t["lit"] if e in lit else t["panel"], t["lit"] if e in lit else t["moe"][1], rx=2.5, sw=1.1)
        for e in range(n)
    )  # fmt: skip


def heat(stops: list[str], v: float) -> str:
    v = min(max(v, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(v), len(stops) - 2)
    a, b = stops[i], stops[i + 1]
    mix = [round(int(a[k : k + 2], 16) + (int(b[k : k + 2], 16) - int(a[k : k + 2], 16)) * (v - i)) for k in (1, 3, 5)]
    return "#" + "".join(f"{c:02X}" for c in mix)


def svg(t: dict[str, Any], data: dict[str, Any]) -> str:
    n, k, layer = int(data["experts"]), int(data["top_k"]), int(data["layer"])
    token = data["token"]

    def chosen(at: int) -> set[int]:
        return {e for e in token["experts"][token["layers"].index(at)] if 0 <= e < n}

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    p.append(
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
        '<marker id="ahg" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["hook"][1]}"/></marker></defs>'
    )
    p.append(rect(1, 1, W - 2, H - 2, t["bg"], t["edge"], rx=26, sw=1.5))

    # ---- left: the model, unchanged; lit experts are the ones one real token chose
    gx, gw, top, rh, gap = 70, 360, 118, 40, 11
    p.append(text(gx, 80, "Granite 3.1 1B", 32, t["ink"], 700))
    p.append(text(gx + 244, 80, "your model · unchanged", 20, t["muted"]))
    for i, at in enumerate(ROWS):
        y = top + i * (rh + gap)
        p.append(text(gx - 10, y + 25, str(at), 12, t["faint"], 500, "end"))
        p.append(rect(gx, y, gw, rh, t["panel"], t["panel_edge"], rx=9, sw=1.2))
        p.append(rect(gx + 10, y + 8, 74, rh - 16, *t["attn"], rx=6, sw=1.2))
        p.append(text(gx + 47, y + 26, "Attn", 13, t["ink"], 500, "middle"))
        p.append(rect(gx + 98, y + 6, 222, rh - 12, *t["moe"], rx=6, sw=1.3))
        p.append(grid(gx + 108, y + 12, 16, n, 6.5, 6.6, t, chosen(at)))
        p.append(rect(gx + 331, y + rh / 2 - 7, 14, 14, *t["hook"], rx=7, sw=1.6))  # the read-only hook
    bottom = top + len(ROWS) * (rh + gap) - gap
    word = token["text"].strip()
    p.append(text(gx, bottom + 36, f"lit: the {k} of {n} experts “{word}” chose, per layer", 18, t["muted"]))

    # ---- centre: one MoE layer, and where the trace reads it
    cx, cy, cw, ch = 478, 150, 646, 268
    zy = top + ROWS.index(layer) * (rh + gap)
    for a, b in ((zy + 2, cy), (zy + rh - 2, cy + ch)):
        p.append(line(f"M {gx + gw - 8} {a} L {cx} {b}", t["faint"], 1.2, "3 4"))
    p.append(rect(cx, cy, cw, ch, t["panel"], t["panel_edge"], rx=16, sw=1.5))
    p.append(text(cx + 22, cy + 36, f"one MoE layer · layer {layer}", 18, t["muted"], 600))
    my = cy + 120

    def arrow(x1: float, x2: float, y: float = my) -> str:
        return line(f"M {x1} {y} L {x2} {y}", t["arrow"], marker="ah")

    p.append(text(cx + 30, my + 7, "x", 24, t["ink"], 500, "middle", 'font-style="italic"'))
    p.append(arrow(cx + 44, cx + 72))
    p.append(rect(cx + 74, my - 36, 112, 72, *t["moe"], rx=10, sw=1.8))
    p.append(text(cx + 130, my - 4, "router", 18, t["ink"], 700, "middle"))
    p.append(text(cx + 130, my + 20, f"top {k} of {n}", 14, t["muted"], 500, "middle"))
    p.append(arrow(cx + 186, cx + 250))
    mx, mw, mh = cx + 252, 240, 150
    p.append(rect(mx, my - mh / 2, mw, mh, *t["moe"], rx=12, sw=1.8))
    p.append(text(mx + mw / 2, my - mh / 2 + 24, "experts", 15, t["ink"], 600, "middle"))
    p.append(grid(mx + 40, my - mh / 2 + 38, 8, n, 14, 6, t, chosen(layer)))
    p.append(text(mx + mw / 2, my + mh / 2 - 12, "computed by the model, unchanged", 13, t["muted"], 500, "middle"))
    p.append(arrow(mx + mw, mx + mw + 30))
    p.append(text(mx + mw + 50, my + 7, "y", 24, t["ink"], 500, "middle", 'font-style="italic"'))
    hx, hy, hw, hh = cx + 112, cy + ch - 62, 330, 46
    p.append(rect(hx, hy, hw, hh, *t["hook"], rx=10, sw=1.8, dash="6 4"))
    p.append(text(hx + hw / 2, hy + 20, "Innards reads, read-only", 15, t["hook"][1], 700, "middle"))
    p.append(text(hx + hw / 2, hy + 37, "top_k_index · top_k_weights", 13, t["muted"], 500, "middle"))
    p.append(line(f"M {cx + 218} {my + 2} L {cx + 218} {hy - 2}", t["hook"][1], dash="5 4", marker="ahg"))

    # ---- right: what the trace measured for three topics in this layer
    dx, dw = 1170, 360
    p.append(text(dx, 80, "Innards", 32, t["ink"], 700))
    p.append(text(dx + 132, 80, "measures only", 20, t["muted"]))
    cell = (dw - 24) / n
    ry = 118
    for i, (topic, sel) in enumerate(data["selection"].items()):
        y = ry + i * 82
        keep = set(data["keep"][topic])
        share = f"{len(keep)} of {n} take {data['coverage']:.0%}"
        p.append(rect(dx, y, dw, 70, t["panel"], t["panel_edge"], rx=9, sw=1.2))
        p.append(text(dx + 12, y + 24, topic, 15, t["ink"], 700))
        p.append(text(dx + dw - 12, y + 24, share, 13, t["muted"], 500, "end"))
        for e, v in enumerate(sel):
            on = e in keep
            p.append(rect(dx + 12 + e * cell, y + 36, cell - 2.4, 22, heat(t["heat"], v / 0.75),
                          t["moe"][1] if on else t["panel_edge"], rx=2, sw=1.5 if on else 0.6))  # fmt: skip
    y = ry + 3 * 82 + 10
    p.append(text(dx, y + 8, f"layer {layer}: share of each topic's tokens that chose", 15, t["muted"]))
    p.append(text(dx, y + 28, f"each expert; outlined, the fewest that take {data['coverage']:.0%}", 15, t["muted"]))
    kept, of = data["kept"]["math"]
    p.append(rect(dx, y + 44, dw, 56, *t["hook"], rx=9, sw=1.6))
    p.append(
        text(dx + 14, y + 67, f'select("math", {data["coverage"]}): keep {kept} of {of} experts', 15, t["ink"], 600)
    )
    p.append(text(dx + 14, y + 89, "→ math-experts.json, for your extraction code", 13, t["muted"], 500))
    curve = f"M {hx + hw + 4} {hy + hh / 2} C {cx + cw + 30} {hy + hh / 2}, {dx - 60} {ry + 120}, {dx - 6} {ry + 120}"
    p.append(line(curve, t["hook"][1], 2, "7 5", "ahg"))
    p.append(text(hx + hw + 72, hy + hh / 2 + 24, "counted per topic", 14, t["hook"][1], 600, "middle"))

    # ---- legend
    ly = H - 28
    items = [
        (t["hook"], "Innards: read-only hooks"),
        (t["moe"], "MoE block (the model's own)"),
        (t["attn"], "attention"),
    ]
    x = 70
    for (fill, stroke), label in items:
        p.append(rect(x, ly - 15, 20, 18, fill, stroke, rx=4, sw=1.4))
        p.append(text(x + 28, ly, label, 18, t["muted"]))
        x += 28 + len(label) * 8.7 + 42
    where = f"measured: {data['model'].split('/')[-1]} · bf16 · {data['device'].upper()}"
    p.append(text(W - 70, ly, where, 16, t["faint"], 500, "end"))
    p.append("</svg>")
    return "".join(p)


def draw(data: dict[str, Any]) -> None:
    from playwright.sync_api import sync_playwright

    inter = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=block"
    head = f'<head><link rel="stylesheet" href="{inter}"></head>'
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception:  # no bundled Chromium: use the installed Chrome
            browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        body = f'<body style="margin:0;background:transparent">{svg(THEME, data)}</body>'
        page.set_content(f"<html>{head}{body}</html>")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(300)
        out = FIGURES / "banner_light.png"
        page.locator("svg").screenshot(path=str(out), omit_background=True)
        print("wrote", out)
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="README banner from a measured expert trace")
    parser.add_argument("--measure", action="store_true", help="trace the model and rewrite banner_data.json first")
    args = parser.parse_args()
    if args.measure:
        DATA.write_text(json.dumps(measure(), indent=1) + "\n", encoding="utf-8")
        print("wrote", DATA)
    draw(json.loads(DATA.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
