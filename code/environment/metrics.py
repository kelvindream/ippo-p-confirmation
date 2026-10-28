"""Shared traffic metric definitions for local rewards and network evaluation."""

import math
import os
import xml.etree.ElementTree as ET


class VehicleMetricTracker:
    """Track episode-level vehicle travel-time and completion metrics."""

    def __init__(self, sumo, route_file=None, demand_end=None):
        self.sumo = sumo
        self.departures = {}
        self.free_flow_times = {}
        self.arrivals = {}
        self.teleports = set()
        self.teleport_starts = set()
        self.teleport_ends = set()
        self.total_departed = 0
        self.demand_end = float(demand_end) if demand_end is not None else None
        self.plan = self._load_planned_demand(route_file, demand_end)
        self.event_log = []

    @staticmethod
    def _load_planned_demand(route_file, demand_end):
        if not route_file or not os.path.isfile(route_file):
            return {}
        root = ET.parse(route_file).getroot()
        plan = {}
        for vehicle in root.findall('vehicle'):
            depart = float(vehicle.attrib['depart'])
            if demand_end is None or depart <= float(demand_end):
                plan[vehicle.attrib['id']] = {
                    'scheduled_departure': depart,
                    'state': 'not_yet_inserted',
                }
        for flow in root.findall('flow'):
            begin = float(flow.attrib['begin'])
            end = float(flow.attrib['end'])
            if 'period' in flow.attrib:
                period = float(flow.attrib['period'])
            else:
                period = 3600.0 / float(flow.attrib['vehsPerHour'])
            limit = end if demand_end is None else min(end, float(demand_end))
            index = 0
            depart = begin
            while depart <= limit + 1e-9:
                plan[f"{flow.attrib['id']}.{index}"] = {
                    'scheduled_departure': depart,
                    'state': 'not_yet_inserted',
                }
                index += 1
                depart += period
        return plan

    def _free_flow_time(self, vehicle_id):
        total = 0.0
        for edge_id in self.sumo.vehicle.getRoute(vehicle_id):
            try:
                lane_count = int(self.sumo.edge.getLaneNumber(edge_id))
            except Exception:
                continue
            lane_values = []
            for lane_index in range(lane_count):
                try:
                    lane_id = f'{edge_id}_{lane_index}'
                    length = float(self.sumo.lane.getLength(lane_id))
                    speed = float(self.sumo.lane.getMaxSpeed(lane_id))
                except Exception:
                    continue
                if length > 0.0 and speed > 0.0:
                    lane_values.append(length / speed)
            if lane_values:
                total += min(lane_values)
        return total

    def update(self):
        now = float(self.sumo.simulation.getTime())
        departed_ids = self.sumo.simulation.getDepartedIDList()
        arrived_ids = self.sumo.simulation.getArrivedIDList()
        try:
            teleport_start_ids = self.sumo.simulation.getStartingTeleportIDList()
            teleport_end_ids = self.sumo.simulation.getEndingTeleportIDList()
        except AttributeError:
            teleport_start_ids = ()
            teleport_end_ids = ()

        self.teleport_starts.update(teleport_start_ids)
        self.teleport_ends.update(teleport_end_ids)
        self.teleports.update(teleport_start_ids)
        self.teleports.update(teleport_end_ids)

        for vehicle_id in departed_ids:
            if vehicle_id in self.departures:
                continue
            self.departures[vehicle_id] = now
            self.free_flow_times[vehicle_id] = self._free_flow_time(vehicle_id)
            if vehicle_id in self.plan:
                self.plan[vehicle_id]['state'] = 'in_network'

        for vehicle_id in arrived_ids:
            if vehicle_id in self.departures:
                self.arrivals[vehicle_id] = now
                if vehicle_id in self.plan:
                    self.plan[vehicle_id]['state'] = 'abnormal_arrival' if vehicle_id in self.teleports else 'normal_arrival'
        for vehicle_id in teleport_start_ids:
            if vehicle_id in self.plan:
                self.plan[vehicle_id]['state'] = 'temporarily_out'
        for vehicle_id in teleport_end_ids:
            if vehicle_id in self.plan and vehicle_id not in self.arrivals:
                self.plan[vehicle_id]['state'] = 'in_network'

        self.event_log.append({
            'time': now,
            'departed': list(departed_ids),
            'arrived': list(arrived_ids),
            'teleport_start': list(teleport_start_ids),
            'teleport_end': list(teleport_end_ids),
        })
        self.total_departed = len(self.departures)

    def snapshot(self):
        for vehicle_id, record in self.plan.items():
            if record['state'] == 'in_network' and vehicle_id not in self.departures:
                record['state'] = 'not_yet_inserted'
        completed = [vehicle_id for vehicle_id in self.arrivals if vehicle_id in self.departures]
        travel_times = [
            self.arrivals[vehicle_id] - self.departures[vehicle_id]
            for vehicle_id in completed
        ]
        delays = [
            max(0.0, travel_time - self.free_flow_times.get(vehicle_id, 0.0))
            for vehicle_id, travel_time in zip(completed, travel_times)
        ]
        completed_count = len(completed)
        planned_count = len(self.plan)
        normal_completed = [
            vehicle_id
            for vehicle_id, record in self.plan.items()
            if record['state'] == 'normal_arrival'
        ]
        demand_service_times = []
        if self.demand_end is not None:
            for vehicle_id, record in self.plan.items():
                service_end = self.demand_end
                if record['state'] == 'normal_arrival':
                    service_end = min(self.arrivals.get(vehicle_id, self.demand_end), self.demand_end)
                demand_service_times.append(
                    max(0.0, service_end - record['scheduled_departure'])
                )
        return {
            'travel_time': sum(travel_times) / completed_count if completed_count else None,
            'delay': sum(delays) / completed_count if completed_count else None,
            'demand_service_time': (
                sum(demand_service_times) / planned_count
                if planned_count and self.demand_end is not None else None
            ),
            'completed_vehicles': completed_count,
            'normal_completed_vehicles': len(normal_completed),
            'planned_completion_rate': (
                len(normal_completed) / planned_count if planned_count else 0.0
            ),
            'departed_vehicles': self.total_departed,
            'incomplete_vehicles': max(0, self.total_departed - completed_count),
            'completion_rate': (
                completed_count / self.total_departed if self.total_departed else 0.0
            ),
            'teleport_count': len(self.teleports),
            'planned_vehicles': len(self.plan),
            'demand_states': self.demand_state_counts(),
            'demand_conservation_ok': self.demand_conservation_ok(),
            'event_log': list(self.event_log),
        }

    def demand_state_counts(self):
        counts = {}
        state_names = {
            'normal_arrival': 'normal_compliant_arrival',
            'abnormal_arrival': 'abnormal_arrival',
            'in_network': 'still_in_network',
            'temporarily_out': 'temporarily_out_waiting_reinsertion',
            'not_yet_inserted': 'not_yet_inserted',
            'cancelled_or_permanent': 'cancelled_or_permanently_removed',
        }
        for record in self.plan.values():
            state = state_names.get(record['state'], 'cancelled_or_permanently_removed')
            counts[state] = counts.get(state, 0) + 1
        for state in state_names.values():
            counts.setdefault(state, 0)
        return counts

    def demand_conservation_ok(self):
        return sum(self.demand_state_counts().values()) == len(self.plan)


def collect_lane_metrics(sumo, lane_ids, deduplicate=True):
    """Aggregate speed, queue, and waiting metrics for a lane set."""
    seen_lanes = set()
    seen_vehicles = set()
    total_speed = 0.0
    speed_vehicle_count = 0
    total_queue = 0.0
    total_waiting = 0.0
    waiting_vehicle_count = 0
    lane_count = 0

    for lane_id in lane_ids:
        if deduplicate and lane_id in seen_lanes:
            continue
        seen_lanes.add(lane_id)
        lane_count += 1
        vehicle_ids = list(sumo.lane.getLastStepVehicleIDs(lane_id))
        vehicle_count = len(vehicle_ids)
        if vehicle_count:
            lane_speed = max(0.0, float(sumo.lane.getLastStepMeanSpeed(lane_id)))
            total_speed += lane_speed * vehicle_count
            speed_vehicle_count += vehicle_count

        total_queue += float(sumo.lane.getLastStepHaltingNumber(lane_id)) * 7.5
        for vehicle_id in vehicle_ids:
            if deduplicate and vehicle_id in seen_vehicles:
                continue
            seen_vehicles.add(vehicle_id)
            total_waiting += float(sumo.vehicle.getWaitingTime(vehicle_id))
            waiting_vehicle_count += 1

    return {
        'speed': total_speed / speed_vehicle_count if speed_vehicle_count else 0.0,
        'arrived_number': int(sumo.simulation.getArrivedNumber()),
        'queue': total_queue / lane_count if lane_count else 0.0,
        'waiting': total_waiting / waiting_vehicle_count if waiting_vehicle_count else 0.0,
        'lane_count': lane_count,
        'vehicle_count': waiting_vehicle_count,
        'speed_vehicle_count': speed_vehicle_count,
        'total_speed': total_speed,
        'total_queue': total_queue,
        'total_waiting': total_waiting,
    }


def collect_network_metrics(sumo, traffic_signals):
    """Aggregate network metrics over the unique controlled-lane union."""
    lane_ids = []
    for signal in traffic_signals:
        lane_ids.extend(signal.lanes_id)
    return collect_lane_metrics(sumo, lane_ids, deduplicate=True)
