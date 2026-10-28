"""Deterministic max-pressure traffic-signal controller."""


class MaxPressureController:
    """Choose at each intersection the phase with the largest net queue pressure."""

    @staticmethod
    def select_actions(traffic_signals):
        return [signal.max_pressure_action() for signal in traffic_signals]
