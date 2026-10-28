"""Descriptive non-RL baseline signal controllers for the frozen confirmation protocol.

Purpose
=======
Two descriptive, non-learning reference controllers evaluated through the SAME
environment/metric path as the frozen IPPO confirmation protocol
(``results/ippo_prior_confirmation_v1/protocol_frozen.json``):

- ``FixedTimeProgramController`` (method label ``Fixed-Time-Program``): the
  no-controller anchor. It runs the static pre-timed signal program embedded in
  the network files as-is, with zero traffic-adaptive logic.
- ``MaxPressureDescriptiveController`` (method label ``Max-Pressure-Descriptive``):
  Varaiya-style max-pressure, descriptive (no tuning, fully deterministic).

Both are evaluation-only: 3 scenarios (H1/H2/H3) x 2 controllers x 5 fixed
evaluation seeds (1042-1046) = 30 episodes, NO training. ``paper_protocol.py``
skips training for these method labels; demand_service_time (U), teleport
counts, demand-conservation and completion-rate integrity checks are collected
by the identical ``_evaluate`` path used for the confirmation tasks. Results go
to ``results/baselines_descriptive_v1/`` and are DESCRIPTIVE ONLY: they anchor
the RL results and are not part of the pre-registered confirmatory test.

Realized signal mechanics through SumoEnv (identical for every controller in
this repo, RL or not)
===============================================================================
- one action request per ``delta_rs_update_time`` = 10 s environment step;
- a requested switch is executed only after the current green phase has served
  its net-program duration (minimum green; the env sets
  ``end_time = switch time + yellow + phase duration``);
- switching inserts ``yellow_time`` = 2 s (fast-forwarded by the env);
- ``reset()`` seeds each intersection with a RANDOM initial green phase
  (``skip_range=0``), exactly as in the confirmation evaluations.

FixedTimeProgramController — exact rule as coded
================================================
Stateless replay of the net program by order: at every decision slot, request
the successor of the currently active green phase in the program's green-phase
order (``(current_index + 1) % num_green_phases``). The env enforces the 33 s
minimum green and inserts the 2 s yellow, so each green is served for its
net-program duration (33 s, extended at most to the next 10 s decision
boundary => 33-43 s) and the realized cycle is ~156-172 s. The request sequence
is completely independent of traffic => no adaptive control.

Net programs (verified from the net files, identical at every intersection):
- ``nets/grid-3x3/3x3.net.xml`` (H1; 9 TLS A0-A2/B0-B2/C0-C2): static program 0,
  offset 0, 4 green phases x 33 s + 2 s yellow each => 140 s nominal cycle.
- ``nets/2x2-grid/2x2_oversaturated.net.xml`` (H2/H3; 4 TLS A0/A1/B0/B1):
  the same program.
A literal do-nothing pass-through is not realizable in this harness because
``reset()`` overrides every intersection with a random initial phase via
``setRedYellowGreenState`` (after which SUMO holds that state until the next
TraCI call); the order-replay above is the closest faithful "program as-is"
realizable through the same env/metrics path.

MaxPressureDescriptiveController — exact rule as coded
=======================================================
At every decision slot (10 s grid; a switch only becomes executable after the
current phase served its 33 s minimum green), for each candidate green phase p
of an intersection compute the net queue pressure

    pressure(p) = sum over the phase's controlled (incoming, outgoing) link
                  pairs of [ halting(incoming lane) - halting(receiving lane) ]

(halting = vehicles with speed < 0.1 m/s from ``lane.getLastStepHaltingNumber``).
Request ``argmax_p pressure(p)``; ties resolve to the first phase in program
order (``np.argmax``). The current phase is held while it remains argmax, so
greens extend beyond the minimum green under load. This delegates to
``TrafficSignal.pressure_state() / max_pressure_action()``
(``environment/traffic_signal.py``) — the exact per-signal rule also used by
the harness method ``Max-Pressure`` (``agents/max_pressure.py``), so the two
labels are behaviorally identical by construction; the separate label keeps the
descriptive bundle self-contained and ``verify_harness_binding()`` guards
against drift.
"""

DESCRIPTIVE_METHODS = ('Fixed-Time-Program', 'Max-Pressure-Descriptive')


class FixedTimeProgramController:
    """Replay the net's static program order; no traffic-adaptive logic."""

    NAME = 'Fixed-Time-Program'

    def __init__(self):
        self._green_order_cache = {}

    def _green_order(self, signal):
        order = self._green_order_cache.get(signal.ts_id)
        if order is None:
            order = [phase.state for phase in signal.all_green_phases]
            self._green_order_cache[signal.ts_id] = order
        return order

    def select_actions(self, traffic_signals):
        """Request the successor green phase in program order at every decision.

        Requests issued before the current phase's minimum green expires are
        holds by env construction (``TrafficSignal.change_phase``).
        """
        actions = []
        for signal in traffic_signals:
            order = self._green_order(signal)
            current = order.index(signal.green_phase.state)
            actions.append((current + 1) % len(order))
        return actions


class MaxPressureDescriptiveController:
    """Varaiya-style max-pressure; deterministic, no tuning."""

    NAME = 'Max-Pressure-Descriptive'

    @staticmethod
    def select_actions(traffic_signals):
        return [int(signal.max_pressure_action()) for signal in traffic_signals]


def summarize_net_programs(net_file):
    """Summarize the static signal programs embedded in a SUMO net file."""
    import xml.etree.ElementTree as ElementTree

    root = ElementTree.parse(net_file).getroot()
    programs = {}
    for tls in root.findall('tlLogic'):
        phases = []
        for phase in tls.findall('phase'):
            duration = float(phase.get('duration'))
            state = phase.get('state', '')
            phases.append({
                'duration_s': duration,
                'is_green': 'y' not in state and 'G' in state,
            })
        programs[tls.get('id')] = {
            'type': tls.get('type'),
            'program_id': tls.get('programID'),
            'offset': tls.get('offset'),
            'cycle_length_s': sum(phase['duration_s'] for phase in phases),
            'phases': phases,
        }
    return programs


def verify_harness_binding():
    """Fail fast if the harness no longer routes the descriptive methods
    through the controllers documented in this module.

    Returns a JSON-serializable check report; raises RuntimeError on any
    broken binding so the runner aborts before writing any result.
    """
    import inspect

    import paper_protocol
    import test_algorithms
    from agents.fixed_time import FixedTimeController
    from agents.max_pressure import MaxPressureController

    select_source = inspect.getsource(paper_protocol._select_actions)
    checks = {
        'harness_Max-Pressure_uses_same_per_signal_rule': (
            test_algorithms.MaxPressureController is MaxPressureController
        ),
        'harness_Fixed-Time_class_from_agents': (
            test_algorithms.FixedTimeController is FixedTimeController
        ),
        'descriptive_methods_registered': all(
            method in paper_protocol.METHODS for method in DESCRIPTIVE_METHODS
        ),
        'select_actions_dispatches_Max-Pressure-Descriptive': (
            'MaxPressureDescriptiveController.select_actions' in select_source
        ),
        'select_actions_dispatches_Fixed-Time-Program': (
            'FixedTimeProgramController' in select_source
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f'Descriptive baseline harness binding broken: {checks}')
    return checks
