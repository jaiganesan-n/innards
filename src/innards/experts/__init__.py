"""Expert routing for MoE models: which experts each token and topic activates, and which to keep.

::

    import innards

    trace = innards.experts.trace(model, tokenizer, run="topic-experts")   # read-only hooks
    with trace.topic("math"):
        model.generate(**inputs, max_new_tokens=64)          # inference
    with trace.topic(batch["topic"]):                        # training: one label per sequence
        loss = model(**batch).loss

    experts = trace.map()                                    # ExpertMap: topic x layer x expert
    experts.top("math", layer=12)                            # the experts math tokens chose most
    experts.specialists("math")                              # chosen far more for math than the rest
    experts.select("math", coverage=0.9).save("math.json")   # the experts to keep, per layer

    routing = trace.tokens()                                 # TokenRouting: every token's experts
    routing.token(" integral")                               # where one token goes, per layer
    routing.expert(layer=12, expert=37)                      # the tokens one expert takes

The counts are measured; selection, gate share, lift and the expert sets are derived from them
and labeled so (see :mod:`innards.experts.analysis`). Innards never removes or changes experts:
the selection is what your own code uses to build a smaller model.
"""

from innards.experts.analysis import METRICS, ExpertMap, Selection, TokenRouting, load, traces
from innards.experts.routing import MoELayer, find_moe_layers, routing_facts
from innards.experts.trace import ExpertTrace, trace

__all__ = [
    "METRICS",
    "ExpertMap",
    "ExpertTrace",
    "MoELayer",
    "Selection",
    "TokenRouting",
    "find_moe_layers",
    "load",
    "routing_facts",
    "trace",
    "traces",
]
