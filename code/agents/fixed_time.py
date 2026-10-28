"""Deterministic fixed-time traffic-signal controller."""


class FixedTimeController:
    """Cycle every intersection through its green phases at a fixed interval."""

    NAME = "Fixed-Time"

    def __init__(self, green_duration=None, yellow_time=2, delta_rs_update_time=10):
        self.green_duration = int(
            delta_rs_update_time if green_duration is None else green_duration
        )
        if self.green_duration <= 0:
            raise ValueError("green_duration must be positive")
        self.yellow_time = int(yellow_time)
        self.delta_rs_update_time = int(delta_rs_update_time)
        self.phase_index = 0
        self.step_count = 0

    def select_actions(self, traffic_signals):
        """Return the synchronized current phase index for each intersection."""
        actions = [
            self.phase_index % len(signal.all_green_phases)
            for signal in traffic_signals
        ]
        self.step_count += 1
        if self.step_count % self.green_duration == 0:
            self.phase_index += 1
        return actions
