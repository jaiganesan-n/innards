"""Feature analyzers: one per architectural feature, found from the model config.

``innards.observe`` detects the model's features and runs every analyzer that matches. When a new
architecture comes out, it gets a new analyzer: subclass :class:`Analyzer` and :func:`register` it.

=====================  ===================================  =================================================
Analyzer               Reads                                Shows
=====================  ===================================  =================================================
``kv``                 Cache object after each turn         KV per layer and type, growth, predicted vs measured
``gated_deltanet``     β and decay per token and layer      Write strength, retention, half-life, fixed state
``gated_attention``    Output gate                          Gate openness per head and token
``moe``                Router logits, latent projection     Expert load, routing entropy, latent vs hidden size
``mhc``                H_pre, H_post, H_res per layer       Stream mixing, doubly-stochastic check, signal gain
``csa_hca``            Cache layout                         Compressed vs window entries, indexer cache
``dsa``                Indexer keys                         Indexer cache, share of context attended
``mla``                Cache layout                         Latent vs expanded layout
=====================  ===================================  =================================================
"""

from innards.features.base import Analyzer, Detection, Payload, Recorder, TurnData, bars, heatmap, line, section
from innards.features.deltanet import GatedDeltaNetAnalyzer
from innards.features.gates import GatedAttentionAnalyzer
from innards.features.kv import KVAnalyzer
from innards.features.layout import CompressedAttentionAnalyzer, DSAAnalyzer, MLAAnalyzer
from innards.features.mhc import MHCAnalyzer
from innards.features.moe import MoEAnalyzer
from innards.features.registry import KNOWN_FEATURES, analyzers, detect, feature_status, get, register

__all__ = [
    "CSAHCA",
    "DSA",
    "KNOWN_FEATURES",
    "MLA",
    "Analyzer",
    "CompressedAttentionAnalyzer",
    "DSAAnalyzer",
    "Detection",
    "GatedAttention",
    "GatedAttentionAnalyzer",
    "GatedDeltaNet",
    "GatedDeltaNetAnalyzer",
    "KVAnalyzer",
    "KVContext",
    "MHCAnalyzer",
    "MLAAnalyzer",
    "MoE",
    "MoEAnalyzer",
    "Payload",
    "Recorder",
    "TurnData",
    "analyzers",
    "bars",
    "detect",
    "feature_status",
    "get",
    "heatmap",
    "line",
    "mHC",
    "register",
    "section",
]

# The registered instances. Each one is callable on what observe captured:
#     GatedDeltaNet(model="Qwen/Qwen3.5-0.8B", args=watch.last_turn())
KVContext = register(KVAnalyzer())
GatedDeltaNet = register(GatedDeltaNetAnalyzer())
GatedAttention = register(GatedAttentionAnalyzer())
MoE = register(MoEAnalyzer())
mHC = register(MHCAnalyzer())
CSAHCA = register(CompressedAttentionAnalyzer())
DSA = register(DSAAnalyzer())
MLA = register(MLAAnalyzer())
