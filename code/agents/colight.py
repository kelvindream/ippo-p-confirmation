"""CoLight baseline: graph attention for neighbor coordination with Double-DQN.

This module is the formal CoLight entry point. The implementation is a
lightweight reproduction of the core coordination mechanism used by this repo.
"""

from ._colight_core import GraphAttentionDoubleDQN


class CoLight(GraphAttentionDoubleDQN):
    """CoLight core-mechanism reproduction."""

    formal_name = "CoLight"
    mechanism_name = "CoLight"
    reproduction_scope = "adjacency multi-head attention neighbor communication and Double-DQN target"
