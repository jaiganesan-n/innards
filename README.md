# Innards

Innards shows what happens inside an open-weight model while it runs on your machine. Your code loads
the original checkpoint and runs it; Innards attaches read-only hooks to the model you loaded and
analyzes what they see beside it, never changing what the model computes.

- **MoE expert routing:** which experts every token and every topic activates, during inference
  (`generate()`) and training (any forward pass), on any MoE architecture, including remote code such
  as `facebook/MobileMoE`. From that: the experts a topic needs per layer, ready to export.
- **Every model:** tokens, context, KV cache per layer, memory and timing, with the KV predicted from
  the config checked against the cache the engine really holds.
- **Each architectural feature at work:** one analyzer per feature (Gated DeltaNet, gated attention,
  MoE and LatentMoE, mHC, CSA/HCA, DSA, MLA), picked automatically from the model's config.
- **A live web view** served from the machine that runs the model. There is no central site, no
  exporter and no API key.

Innards targets models that run on a local machine with 16 GB at 16-bit precision or higher (bf16,
fp16, fp32), through Hugging Face Transformers 5. It never loads or runs a model itself, and it never
removes, merges or edits experts: it measures, and labels everything it derives from a measurement.

## Install

<!-- test: install (checked by the fresh-environment wheel install in CI) -->
```bash
pip install innards                  # core: analysis of stored traces, KV calculator, lineage, run store
pip install "innards[hf]"            # + torch, transformers, psutil: observe and trace a model
pip install "innards[view]"          # + FastAPI and uvicorn for the web view
pip install "innards[aws]"           # + boto3, to sync run folders to S3
pip install "innards[docs]"          # + pypdf, for PDF documents in the Input docs set
pip install "innards[all]"

uv add "innards[hf,view]"            # inside a uv project
```

The core install has two small dependencies (`httpx`, `huggingface-hub`) and works on Python 3.10+.

## MoE expert routing

`innards.experts.trace` hooks every MoE layer of the model you loaded. Inside `trace.topic(...)`, every
forward pass is recorded: the prefill and each decoded token of `generate()`, or a training batch.
Outside a topic nothing is recorded.

<!-- test: requires=hf,network -->
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import innards

model_id = "ibm-granite/granite-3.1-1b-a400m-instruct"  # 32 experts, 8 per token; 2.7 GB in bf16
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)

trace = innards.experts.trace(model, tokenizer, run="topic-experts")  # read-only hooks on every MoE layer
prompts = {
    "math": ["The derivative of x squared is", "Solve for x: 3x + 5 = 20. First,"],
    "code": ["def fibonacci(n):\n    if n <", "import numpy as np\narr = np.zeros("],
}
for topic, texts in prompts.items():
    for text in texts:
        with trace.topic(topic):
            model.generate(**tokenizer(text, return_tensors="pt"), max_new_tokens=16, do_sample=False)

experts = trace.map()  # topic x layer x expert, from the routing the model chose
print(experts.top("math", layer=12, n=4))  # the experts math tokens chose most in layer 12
print(experts.specialists("code", n=3))  # chosen far more for code than for math
keep = experts.select("math", coverage=0.9)  # per layer, the fewest experts that take 90% of math's routing
keep.save("math-experts.json")
print(keep)

routing = trace.tokens()  # every recorded token's experts, per layer
print(routing.token(" x")["counts"][12])  # how often each expert took the token " x" in layer 12
print(routing.expert(layer=12, expert=keep.keep(12)[0])[:3])  # the tokens one expert takes most
trace.close()  # removes the hooks and writes innards_runs/topic-experts/experts/<model>/
```

During training, give one topic per sequence of the batch. Padding is never counted, and the forward
that gradient checkpointing recomputes inside `backward()` is not a new forward:

<!-- test: requires=hf,network -->
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import innards

model_id = "ibm-granite/granite-3.1-1b-a400m-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

trace = innards.experts.trace(model, tokenizer, run="finetune", tokens=False)  # counts only: light for long runs
batch = tokenizer(
    ["The integral of 2x is x squared.", "for i in range(3):\n    print(i)"], return_tensors="pt", padding=True
)
labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
with trace.topic(["math", "code"]):  # one label per sequence
    loss = model(**batch, labels=labels).loss
loss.backward()
optimizer.step()
print(trace.map())  # <ExpertMap ...: 24 MoE layers; tokens math: ..., code: ...>
trace.close()
```

### What the trace measures, and what it derives

| | Value | Meaning |
| --- | --- | --- |
| Measured | `tokens` | Tokens of a topic that went through each MoE layer (padding excluded) |
| Measured | `counts` | How often each expert was in a token's top-k |
| Measured | per-token routing | Each token's chosen experts and their weights, per layer (`tokens=True`) |
| Derived | `selection` | Share of the topic's tokens whose top-k included the expert (`counts / tokens`) |
| Derived | `gate` | The expert's average share of a token's routing weight |
| Derived | `lift` | Selection for this topic ÷ selection for every other topic together |
| Derived | `select(topic, coverage)` | Per layer, the fewest experts that together take `coverage` of the topic's routing |

`select` returns the expert ids to keep per layer, with the module paths of each layer's MoE block,
router, experts and shared expert, the routing settings from the config (`top_k`, `norm_topk_prob`,
`routed_scaling_factor` ...) and how many tokens it was derived from. That file is the input for your
own code that builds a smaller model for the topic; shared experts run for every token and always stay.

### Any MoE architecture

Each MoE layer is read from the first place that carries the model's own routing decision, and the
source is recorded with the data:

| Source | Used when | Exact |
| --- | --- | --- |
| `experts_input` | The block calls `experts(hidden, top_k_index, top_k_weights)` (transformers 5: Qwen3-MoE, Qwen3.5-MoE, Mixtral, OLMoE, Granite MoE, GPT-OSS, DeepSeek-V2/V3, GLM-4 MoE, ERNIE 4.5, Phi-MoE, LFM2-MoE, Trinity/AFMoE, Nemotron-H ...) | Experts and applied weights |
| `router_output` | The router module returns the chosen experts | Experts and weights |
| `topk` | Code that routes inline, such as `facebook/MobileMoE` (its block never calls its router or experts modules): the block's own `torch.topk`, and a `gather` that reuses its indices | Experts exactly; weights are the unbiased scores before normalization |

Every model in that list keeps its outputs bit-identical with and without the trace (tested on replicas
of each family). Slots that are not real experts (LongCat's zero-compute experts, Zaya's skipped slots
with weight 0) are not counted.

MoE checkpoints that fit 16 GB at bf16 include `ibm-granite/granite-3.1-1b-a400m-instruct` (2.7 GB),
`ibm-granite/granite-swash-3b-a600m` (6 GB), `microsoft/Phi-tiny-MoE-instruct` (7.5 GB),
`facebook/MobileMoE-S-Base` (2.4 GB, remote code), `arcee-ai/Trinity-Nano-Preview` (12.2 GB) and
`allenai/OLMoE-1B-7B-0125-Instruct` (13.8 GB, tight).

### From the command line and in the view

<!-- test: cli -->
```bash
innards experts                                   # every trace under ./innards_runs: model, topics, tokens
```

<!-- test: text -->
```text
innards experts innards_runs/topic-experts --topic math                 # top experts per layer
innards experts innards_runs/topic-experts --topic code --by lift       # specialists against the other topics
innards experts innards_runs/topic-experts --topic math --select 0.9 --out math-experts.json
innards experts innards_runs/topic-experts/experts/granite-3.1-1b-a400m-instruct --token " x"
```

The **Experts** tab of the web view shows a trace live (it is written every 30 seconds and on
`trace.flush()`): experts per layer for a topic by selection, gate share or lift; topics side by side in
one layer; the experts each topic needs per layer and the overlap between topics' sets; specialists;
each recorded sequence as tokens x experts; and, on a click, the tokens one expert takes. It downloads
the expert set for the coverage you pick.

## Observe a model: one turn per `generate()`

<!-- test: requires=hf,network -->
```python
import innards
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")  # your code loads the original checkpoint

watch = innards.observe(model, tokenizer, model_id="Qwen/Qwen3.5-0.8B", run="deltanet-study")

messages = [{"role": "user", "content": "Remember this: my name is Priya."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=32)  # your code, unchanged: one turn

turn = watch.last_turn().record
print(turn.kv.measured_bytes == turn.kv.predicted_bytes)  # True
print(sorted(turn.features))  # ['gated_attention', 'gated_deltanet', 'kv']
watch.close()
```

`watch.view()` serves the web view from the same process (in a background thread, so a notebook keeps
working) and returns its address, for example `http://127.0.0.1:8000/`. `observe` and
`innards.experts.trace` can watch the same model at once.

### What each turn records

The **light tier** is always on and costs metadata only: tokens and prefix reuse, the context, KV bytes
per layer and per layer type (measured from tensor shapes and dtypes, never values), memory against the
budget (16 GB by default, `INNARDS_GPU_BUDGET_GB`), TTFT, prefill and decode speed, and Innards' own overhead.

**Feature probes** attach only to the features the config says the model has. Each probe reports its
own measured overhead, supports sampling (`sample_every=N` tokens, or `layers=[...]`), and copies what
it reads to pinned CPU memory without blocking; a worker thread analyzes it.

| Analyzer | Reads | Shows |
| --- | --- | --- |
| `kv` (every model) | Cache object after each turn | KV per layer and layer type, growth, context, predicted vs measured, with gaps explained |
| `gated_deltanet` | β and decay per token and layer | Write strength, retention, half-life, fixed state next to growing KV |
| `gated_attention` | Output gate | Gate openness per head and token |
| `moe` (MoE and LatentMoE) | Chosen experts of every MoE block | Expert load per turn, routing entropy, latent size vs hidden size |
| `mhc` | H_pre, H_post, H_res per layer | Stream mixing, doubly-stochastic check (row/column sums), signal gain across depth |
| `csa_hca` | Cache layout | Compressed vs window entries, indexer cache, tokens waiting to be compressed |
| `dsa` | Indexer keys | Indexer cache, share of the context each query attends to |
| `mla` | Cache layout | Latent vs expanded layout |

Features Innards recognizes but does not analyze yet (Mamba-2 layers, short convolutions) are monitored
at the light tier and marked "not analyzed yet"; multi-token prediction heads are marked "not run by
transformers". When a new architecture comes out, it gets a new analyzer (`innards.features.register`).

An analyzer can also be called on a captured turn; `args` is what `observe` captured, not the output
of `generate()`, which holds no hook captures:

<!-- test: requires=hf,network -->
```python
import innards
from innards import GatedDeltaNet
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
watch = innards.observe(model, tokenizer, store=False)
model.generate(**tokenizer("The keeper lit the lamp", return_tensors="pt"), max_new_tokens=8)

payload = GatedDeltaNet(model="Qwen/Qwen3.5-0.8B", args=watch.last_turn())
print(payload["findings"][0])  # Write strength β averaged ...
watch.close()
```

## The web view

A light monitoring dashboard with three tabs and a sidebar.

- **Header:** model picker (the lineup, any Hub id or local folder), precision, Check and Load, a live
  indicator ("● Live · turn 7") and a memory gauge against the budget.
- **Sidebar:** what is monitored now (KV cache, context, memory, timing), the model's features
  (analyzed, not in this model, or not analyzed yet), and health: each probe's overhead, dropped events
  and errors.
- **Conversation:** run a conversation set turn by turn, or type your own messages; every reply shows
  tokens reused and prefilled, the KV change and speed.
- **Analysis:** stat tiles with sparklines, what changed from the version before, each feature at work,
  KV/context/memory per turn predicted vs measured with a per-layer heatmap, and the same set side by side
  with the version before. Hovering turn 5 highlights turn 5 in every chart.
- **Experts:** the expert routing traces of the run folder (see MoE expert routing).

Load, Check and Send need a runner (`innards-exp serve`), because Innards never loads a model. Without one,
the view monitors whatever your code generates, and opens any run folder:

<!-- test: text -->
```text
innards view                                   # ./innards_runs on http://127.0.0.1:8000
innards view ./innards_runs/deltanet-study     # one run folder, e.g. copied from another machine
ssh -L 8000:127.0.0.1:8000 user@<gpu-host>     # a remote machine's view through a tunnel
innards sync up deltanet-study s3://bucket/innards      # copy a run folder to S3
innards sync down s3://bucket/innards/deltanet-study    # and back, then `innards view`
```

## Where results go

Results stay on the machine that runs the model, next to the run:

<!-- test: text -->
```text
innards_runs/<run>/run.json                                 name, host, environment
innards_runs/<run>/<conversation>/conversation.json         model, config, features, conversation set, status
innards_runs/<run>/<conversation>/turns.jsonl               one TurnRecord (schema 1.0) per turn
innards_runs/<run>/experts/<name>/summary.json              expert trace: topics, tokens and counts per layer
innards_runs/<run>/experts/<name>/vocab.json                the text of every recorded token id
innards_runs/<run>/experts/<name>/tokens/shard-*.safetensors   per-token experts, weights, positions, topics
```

Schema 1.0 records a run, a conversation and a turn, with one payload per analyzed feature under
`features`. Missing measurements are `null`, never zero. Token shards are plain safetensors files
(`experts` int16 `[tokens, layers, k]` with -1 for no expert, `weights` float32, `token`, `sequence`,
`position`, `topic`), readable from torch, numpy or JAX.

<!-- test: check-output -->
```python
from innards.schema import TurnRecord

record = TurnRecord.from_dict(
    {
        "schema_version": "1.0",
        "run": "deltanet-study",
        "conversation": "20261103-081500-Qwen3.5-0.8B-a1b2",
        "turn": 7,
        "timestamp": "2026-11-03T08:15:42Z",
        "model": {"id": "Qwen/Qwen3.5-0.8B", "architecture": "hybrid", "features": ["gated_deltanet"]},
        "runtime": {"backend": "observe"},
        "kv": {"predicted_bytes": 2154496, "measured_bytes": 2154496, "measured_source": "hf_cache"},
    }
)
print(record.kv.measured_bytes, record.memory.peak_bytes)  # 2154496 None
```

## Checks before loading

Every check runs before anything is downloaded, and says what is wrong, the numbers and the fix:
architecture support in the installed transformers (never `trust_remote_code`), whether this device can
run the checkpoint's published precision (FP8 and FP4 need CUDA), whether weights (from safetensors
metadata) plus the KV for the chosen conversation set plus headroom fit the budget, gated-repo access,
and which features have analyzers. The runner checks memory again after loading and unloads cleanly on
out-of-memory.

<!-- test: check-output requires=hf -->
```python
from innards.validate import validate

report = validate("ibm-granite/granite-swash-3b-a600m", device="mps", check_access=False)
print(report.status)  # Ready
small = validate("ibm-granite/granite-swash-3b-a600m", device="mps", budget_gb=4, check_access=False)
print([check.id for check in small.blocked])  # ['memory']
```

## Conversation sets

| Set | Contents | What it shows |
| --- | --- | --- |
| `basic_qa` | Short, independent questions | Baseline KV per token and speed |
| `with_context` | A chat that plants facts early and asks about them later | KV growth across turns, recall, prefix reuse |
| `input_docs` | Your own .txt, .md or .pdf, then questions about it | Large prefill, jumps in context and KV |
| `long_document` | One long built-in document, then questions about it | Capacity; sliding, global and recurrent layers |

Running the same set on both models of a pair makes the Analysis tab compare like with like.

## The lineup

Pairs of a newer model and the version before it, original checkpoints only, each running on a 16 GB
machine at 16-bit precision:

| Feature | Newer model | Version before |
| --- | --- | --- |
| Gated DeltaNet + gated attention | Qwen3.5-0.8B (or 2B) | Qwen3-0.6B (or 1.7B) |
| MoE with a shared expert + sliding-window attention | Granite Swash 3B-A600M | Granite 3.1 3B-A800M |

Pinned configs ship with Innards, so each pair's architecture, what changed and its KV predicted from the
config are available before either model runs (labeled as predicted):

<!-- test: check-output -->
```python
from innards.kv import predict
from innards.lineage.lineup import model_config, what_changed

p = predict(model_config("ibm-granite/granite-swash-3b-a600m"), context_tokens=32768, kv_dtype="bf16")
print(p.architecture, round(p.total_bytes / 2**20))  # sliding_window 258
changed = what_changed("ibm-granite/granite-swash-3b-a600m", "ibm-granite/granite-3.1-3b-a800m-instruct")
print([a["name"] for a in changed["added"]])  # ['Shared expert (runs for every token)']
```

## KV calculator and lineage

`predict` gives architectural KV bytes, capacity and a per-layer breakdown from a config alone:

<!-- test: check-output -->
```python
from innards.kv import predict

gemma_like = {
    "model_type": "gemma3_text",
    "num_hidden_layers": 26,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "sliding_window": 512,
    "sliding_window_pattern": 6,
    "max_position_embeddings": 32768,
}
p = predict(gemma_like, context_tokens=8192, kv_dtype="bf16")
print(p.by_layer_type)  # {'global': 33554432, 'sliding': 11511808, 'recurrent': 0}
```

`explain` says what an architecture changed from its predecessor, the problem it solved and the trade-off:

<!-- test: check-output -->
```python
from innards.lineage import chain, explain

print(explain("gqa").predecessor, "->", explain("gqa").family)  # mha -> gqa
print([entry.family for entry in chain("compressed")])  # ['compressed', 'dsa', 'mla', 'gqa', 'mha']
```

## CLI

<!-- test: cli -->
```bash
innards lineup                                               # the pairs and what each newer model adds
innards predict --model Qwen/Qwen3-0.6B --context 32768 --kv-dtype fp8
innards explain sliding_window
innards validate Qwen/Qwen3.5-0.8B --device cpu --set with_context
```

`innards experts` reads expert traces (see MoE expert routing), `innards view [folder]` serves the web
view and `innards sync up|down` copies run folders to and from S3.

## Zero slowdown

- **Hot path:** a `generate` wrapper and one forward pre-hook take timestamps, count steps and offer a
  small event to a bounded queue that never blocks (full means dropped and counted). No network calls,
  disk writes or device synchronization happen during generation.
- **Feature probes and expert routing** reduce small tensors on the device (expert counts are added with
  `index_add_`, no host synchronization); only a worker, or `trace.flush()` between forwards, waits for
  the device. Every hook is timed, so its overhead is measured and shown, and every hook is fail-open: an
  error is counted, never raised into the model.
- **Worker threads** do the rest: KV math and measurement, memory, analysis, writing the run folder.
- **Same output:** greedy replies are identical with and without the hooks (tested on the Qwen pair, on
  `granite-3.1-1b-a400m`, and on replicas of every analyzer's architecture and every MoE family above).

**Budget:** throughput within 1% with the light tier, measured by `scripts/overhead_bench.py`
(same model and prompts, observed vs plain `generate`, interleaved runs, medians).

## Supported

**Architectures** (KV calculator and measurements): MHA, MQA, GQA, MLA (DeepSeek-V2/V3), DSA
(DeepSeek-V3.2, with its indexer keys), compressed attention (DeepSeek-V4 CSA/HCA), sliding window and
chunked layers (Gemma 2/3, GPT-OSS, OLMo 3, Granite Swash, Llama 4 ...), hybrids (Qwen3.5/3.6 and
Qwen3-Next Gated DeltaNet, Nemotron-H and Nemotron 3, Granite 4.0-H, LFM2, Jamba, Bamba, Falcon-H1) and
SSMs. Recurrent state is sized in float32 where the engine keeps it so. **MoE routing:** every MoE block
with an `experts` submodule (see Any MoE architecture).

**Models:** open-weight, as the vendor published them, at 16-bit precision or higher on a 16 GB machine.
Community quantizations, distillations and conversions are out of scope; tiny random-weight replicas
appear only in the test suite.

**Engine:** Hugging Face Transformers 5.x in-process (probes need forward hooks), on CUDA, Apple MPS or
CPU. **Python:** 3.10 to 3.15.

## License

Apache-2.0
