"""Innards: see inside open-weight models while they run.

Your code loads the model; Innards watches it::

    import innards

    watch = innards.observe(model, tokenizer, model_id="Qwen/Qwen3.5-0.8B", run="deltanet-study")
    out = model.generate(**inputs, max_new_tokens=256)    # one turn

Feature analyzers run automatically for the features the model has, and can be called explicitly::

    from innards import GatedDeltaNet
    GatedDeltaNet(model="Qwen/Qwen3.5-0.8B", args=watch.last_turn())

For MoE models, :mod:`innards.experts` records which experts each token and topic activates, during
inference and training, and which experts a topic needs::

    trace = innards.experts.trace(model, tokenizer, run="topic-experts")
    with trace.topic("math"):
        model.generate(**inputs, max_new_tokens=64)
    trace.map().select("math", coverage=0.9)
    trace.view(chat=True)                                 # Conversation and MoE layers tabs on this machine
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("innards")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

from innards import experts
from innards.features import (
    CSAHCA,
    DSA,
    MLA,
    GatedAttention,
    GatedDeltaNet,
    KVContext,
    MoE,
    mHC,
)
from innards.observer.watch import CapturedTurn, Watch, observe
from innards.schema import ConversationRecord, TurnRecord

__all__ = [
    "CSAHCA",
    "DSA",
    "MLA",
    "CapturedTurn",
    "ConversationRecord",
    "GatedAttention",
    "GatedDeltaNet",
    "KVContext",
    "MoE",
    "TurnRecord",
    "Watch",
    "__version__",
    "experts",
    "mHC",
    "observe",
    "view",
]


def view(
    root: str | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    runner: Any = None,
    open_browser: bool = False,
    block: bool = True,
) -> object:
    """Serve the web view over the expert traces in a folder of runs (default ``./innards_runs``).

    Needs ``innards[view]``. To chat with a model on the Conversation tab, run ``innards view`` (it loads
    the model) or call ``trace.view(chat=True)`` on a model your code traced.
    """
    from innards.view import serve

    return serve(root=root, runner=runner, host=host, port=port, open_browser=open_browser, block=block)
