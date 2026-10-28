"""PressLight baseline: phase-pressure state, pressure reward, and Double-DQN target.

This module is the formal PressLight entry point. The implementation is a
lightweight reproduction of the core PressLight mechanism used by this repo.
"""

from ._presslight_core import PressureStateDoubleDQN


class PressLight(PressureStateDoubleDQN):
    """PressLight core-mechanism reproduction."""

    formal_name = "PressLight"
    mechanism_name = "PressLight"
    reproduction_scope = "phase-pressure state, pressure reward, and Double-DQN target"
