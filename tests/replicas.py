"""Tiny random-weight replicas of real architectures, for testing analyzer code offline.

Replicas appear only in tests: they exercise the probes, hooks and analysis on the exact module
classes transformers uses for each architecture, but their numbers mean nothing. Results always come
from original checkpoints.
"""

from __future__ import annotations

import functools
from typing import Any

WORDS = [
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "of",
    "to",
    "in",
    "and",
    "or",
    "on",
    "for",
    "with",
    "my",
    "your",
    "name",
    "what",
    "why",
    "how",
    "when",
    "who",
    "which",
    "that",
    "this",
    "it",
    "remember",
    "favourite",
    "colour",
    "green",
    "blue",
    "red",
    "priya",
    "arun",
    "cache",
    "memory",
    "model",
    "token",
    "tokens",
    "layer",
    "layers",
    "kv",
    "key",
    "value",
    "attention",
    "grows",
    "context",
    "long",
    "short",
    "tell",
    "me",
    "about",
    "story",
    "tea",
    "bus",
    "city",
    "school",
    "day",
    "night",
    "hello",
    "hi",
    "thanks",
    "please",
    "yes",
    "no",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "again",
    "first",
    "last",
    "question",
    "answer",
    "write",
    "read",
    "keep",
    "forget",
    "more",
    "less",
    "fact",
    "facts",
    "document",
    "page",
    "chapter",
    "line",
    ",",
    ".",
    "?",
    "!",
    ":",
    ";",
]


@functools.cache
def tokenizer() -> Any:
    """A word-level tokenizer with a ChatML chat template (special tokens like Qwen's)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {tok: i for i, tok in enumerate([*specials, "\n", "user", "assistant", "system", *WORDS])}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split("\n", behavior="isolated"), pre_tokenizers.Whitespace()]
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    return tok


def vocab_size() -> int:
    return len(tokenizer())


def _config(kind: str) -> Any:
    v = vocab_size()
    if kind == "qwen3":
        from transformers import Qwen3Config

        return Qwen3Config(
            vocab_size=v,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=2048,
        )
    if kind == "qwen3_5":
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        return Qwen3_5TextConfig(
            vocab_size=v,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_num_value_heads=4,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            full_attention_interval=2,
            max_position_embeddings=2048,
            attn_output_gate=True,
        )
    if kind == "nemotron_h":
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

        return NemotronHConfig(
            vocab_size=v,
            hidden_size=64,
            intermediate_size=64,
            hybrid_override_pattern="ME*EM",
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            mamba_num_heads=4,
            mamba_head_dim=16,
            ssm_state_size=16,
            n_groups=2,
            conv_kernel=4,
            expand=1,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=32,
            moe_latent_size=32,
            n_group=1,
            topk_group=1,
            max_position_embeddings=2048,
            mamba_ssm_cache_dtype="float32",
            chunk_size=8,
        )
    if kind == "deepseek_v4":
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

        return DeepseekV4Config(
            vocab_size=v,
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=32,
            qk_rope_head_dim=8,
            q_lora_rank=32,
            o_lora_rank=16,
            o_groups=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=4,
            sliding_window=8,
            hc_mult=2,
            num_hash_layers=1,
            compress_ratios=[0, 4, 128, 4],
            max_position_embeddings=4096,
        )
    if kind == "deepseek_v32":
        from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config

        return DeepseekV32Config(
            vocab_size=v,
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=16,
            v_head_dim=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=8,
            max_position_embeddings=2048,
        )
    if kind == "deepseek_v3":
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        return DeepseekV3Config(
            vocab_size=v,
            hidden_size=64,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=16,
            v_head_dim=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            max_position_embeddings=2048,
        )
    raise ValueError(kind)


KINDS = ("qwen3", "qwen3_5", "nemotron_h", "deepseek_v4", "deepseek_v32", "deepseek_v3")


def replica(kind: str, dtype: str = "float32", seed: int = 0) -> Any:
    """A tiny random-weight model of ``kind`` (see :data:`KINDS`), in eval mode on the CPU."""
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(_config(kind), dtype=getattr(torch, dtype))
    model.config._name_or_path = f"replica/{kind}"
    model.generation_config.pad_token_id = tokenizer().pad_token_id
    model.generation_config.eos_token_id = tokenizer().eos_token_id
    return model.eval()


def chat_ids(messages: list[dict[str, str]]) -> Any:
    tok = tokenizer()
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False)


MOE_KINDS: dict[str, dict[str, Any]] = {
    "qwen3_moe": {"num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "decoder_sparse_step": 1},
    "mixtral": {"num_local_experts": 8, "num_experts_per_tok": 2},
    "olmoe": {"num_experts": 8, "num_experts_per_tok": 2},
    "granitemoe": {"num_local_experts": 8, "num_experts_per_tok": 2},
    "lfm2_moe": {
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_dense_layers": 0,
        "moe_intermediate_size": 32,
        "layer_types": ["conv", "full_attention"],
    },
    "phimoe": {"num_local_experts": 8, "num_experts_per_tok": 2},
    "afmoe": {
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_dense_layers": 0,
        "moe_intermediate_size": 32,
        "num_shared_experts": 1,
    },
    "gpt_oss": {
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "layer_types": ["sliding_attention", "full_attention"],
    },
}
"""Tiny MoE configs per family (2 layers, 8 experts, top-2), routed the way each family routes."""


def moe_replica(kind: str, seed: int = 0) -> Any:
    """A tiny random-weight MoE model of ``kind`` (see :data:`MOE_KINDS`), float32, eval mode, CPU."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    common = {
        "vocab_size": vocab_size(),
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 2048,
    }
    torch.manual_seed(seed)
    config = AutoConfig.for_model(kind, **{**common, **MOE_KINDS[kind]})
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.config._name_or_path = f"replica/{kind}"
    model.generation_config.pad_token_id = tokenizer().pad_token_id
    model.generation_config.eos_token_id = tokenizer().eos_token_id
    return model.eval()


def inline_moe_replica(layers: int = 2, hidden: int = 32, experts: int = 6, top_k: int = 2, seed: int = 0) -> Any:
    """A tiny decoder whose MoE blocks route inline, the way ``facebook/MobileMoE`` does.

    The block never calls its ``router`` (a linear layer used for its weight only) or its ``experts``
    (a parameter container): sigmoid scores, plus a selection-only bias, top-k, the unbiased scores
    gathered, normalized and scaled, then a loop over experts and an always-on shared expert.
    Written here from the published routing, so the ``topk`` source is tested without remote code.
    """
    import torch
    import torch.nn.functional as F
    from torch import nn

    class Experts(nn.Module):  # parameters only, never called
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = experts
            self.gate_up_proj = nn.Parameter(torch.randn(experts, hidden, 2 * hidden) * 0.1)
            self.down_proj = nn.Parameter(torch.randn(experts, hidden, hidden) * 0.1)
            self.act_fn = nn.SiLU()

    class InlineMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.top_k, self.num_experts, self.hidden_dim = top_k, experts, hidden
            self.routed_scaling_factor = 2.5
            self.experts = Experts()
            self.router = nn.Linear(hidden, experts, bias=False)
            self.expert_bias = nn.Parameter(torch.randn(experts) * 0.05, requires_grad=False)
            self.shared_expert = nn.Linear(hidden, hidden)

        def routing(self, x: Any) -> tuple[Any, Any]:
            scores = torch.sigmoid(F.linear(x.float(), self.router.weight.float()))
            _, index = torch.topk(scores + self.expert_bias, self.top_k, dim=-1)
            weights = scores.gather(1, index)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            return index, weights * self.routed_scaling_factor

        def forward(self, hidden_states: Any) -> Any:
            x = hidden_states.reshape(-1, self.hidden_dim)
            index, weights = self.routing(x)
            out = torch.zeros_like(x)
            for e in range(self.num_experts):
                mask = (index == e).any(dim=-1)
                rows = mask.nonzero(as_tuple=True)[0]
                gate, up = (x[rows] @ self.experts.gate_up_proj[e]).chunk(2, dim=-1)
                y = (self.experts.act_fn(gate) * up) @ self.experts.down_proj[e]
                pos = (index[rows] == e).nonzero(as_tuple=True)
                out[rows] += y * weights[rows][pos[0], pos[1]].unsqueeze(-1).to(y.dtype)
            return (out + self.shared_expert(x)).view_as(hidden_states)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden)
            self.feed_forward = InlineMoE()

        def forward(self, h: Any) -> Any:
            return h + self.feed_forward(self.norm(h))

    class InlineMoEModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size(), hidden)
            self.layers = nn.ModuleList(Layer() for _ in range(layers))
            self.head = nn.Linear(hidden, vocab_size())

        def forward(self, input_ids: Any, attention_mask: Any = None, labels: Any = None) -> Any:
            h = self.embed(input_ids)
            for layer in self.layers:
                h = layer(h)
            logits = self.head(h)
            if labels is None:
                return logits
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

    torch.manual_seed(seed)
    return InlineMoEModel().eval()
