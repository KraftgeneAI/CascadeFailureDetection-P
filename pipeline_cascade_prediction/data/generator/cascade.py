"""
Cascade Module (Pipeline Digital Twin)
======================================
Cascade failure propagation logic for fluid networks.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from .utils import get_failed_lines_from_nodes


class CascadeSimulator:
    """Simulates cascade failure propagation through a pipeline network."""
    
    def __init__(
        self,
        num_nodes: int,
        adjacency_list: List[List[Tuple[int, int, float]]],
        pressure_failure_threshold: np.ndarray,
        pressure_damage_threshold: np.ndarray,
        flow_failure_threshold: np.ndarray,
        flow_damage_threshold: np.ndarray,
        temperature_failure_threshold: np.ndarray,
        temperature_damage_threshold: np.ndarray
    ):
        self.num_nodes = num_nodes
        self.adjacency_list = adjacency_list
        self.pressure_failure_threshold = pressure_failure_threshold
        self.pressure_damage_threshold = pressure_damage_threshold
        self.flow_failure_threshold = flow_failure_threshold
        self.flow_damage_threshold = flow_damage_threshold
        self.temperature_failure_threshold = temperature_failure_threshold
        self.temperature_damage_threshold = temperature_damage_threshold
    
    def check_node_state(
        self,
        node_id: int,
        pressure: float,
        flow_ratio: float,
        temperature: float
    ) -> Tuple[int, str]:
        """Check node failure state based on fluid physics conditions."""
        # MAOP Overpressure Rupture
        if pressure > self.pressure_failure_threshold[node_id]:
            return 2, "overpressure_rupture"
        # Severe low-pressure cavitation (only when pressure drops near 0)
        if pressure < 100.0:
            return 2, "underpressure_cavitation"
        # Pipe flow velocity/capacity exceeded
        if flow_ratio > self.flow_failure_threshold[node_id]:
            return 2, "flow_capacity_exceeded"
        # Thermal stress
        if temperature > self.temperature_failure_threshold[node_id]:
            return 2, "thermal_stress_failure"
        
        # Warnings / Damage
        if pressure > self.pressure_damage_threshold[node_id]:
            return 1, "pressure_stress"
        if pressure < 200.0:
            return 1, "suction_stress"
        if flow_ratio > self.flow_damage_threshold[node_id]:
            return 1, "erosion_wear"
        if temperature > self.temperature_damage_threshold[node_id]:
            return 1, "thermal_wear"
        
        return 0, "none"
    
    def propagate_cascade_physics(
        self,
        initial_failed_nodes: List[Tuple[int, str]],
        injections: np.ndarray,
        extractions: np.ndarray,
        current_temperature: np.ndarray,
        target_num_failures: int,
        fluid_flow_simulator,
        edge_index: np.ndarray,
        flow_limits: np.ndarray,
        extra_failed_nodes: Optional[Set[int]] = None
    ) -> List[Tuple[int, float, str, Optional[int]]]:
        """Propagate cascade with physics-based hydraulic recomputation."""
        injections = injections.copy()
        extractions = extractions.copy()
        extra_failed = set(extra_failed_nodes or ())

        failed_nodes = set(node[0] for node in initial_failed_nodes)
        failed_reasons = [node[1] for node in initial_failed_nodes]
        failure_sequence = [
            (fail_node, 0.0, fail_reason, None)
            for fail_node, fail_reason in zip(failed_nodes, failed_reasons)
        ]

        queue = [(node[0], 0.0) for node in initial_failed_nodes]
        visited = set(node[0] for node in initial_failed_nodes)
        
        failed_lines = get_failed_lines_from_nodes(edge_index, failed_nodes | extra_failed)
        
        pressures, pipe_flows, pump_heads, is_stable = fluid_flow_simulator.compute_fluid_flow(
            injections, extractions,
            failed_lines=failed_lines,
            failed_nodes=list(failed_nodes | extra_failed)
        )
        
        if not is_stable:
            return failure_sequence
        
        flow_ratios = np.abs(pipe_flows) / (flow_limits + 1e-6)
        node_loading = self._calculate_node_loading(edge_index, flow_ratios)
        
        while queue and len(failed_nodes) < target_num_failures:
            current_node, current_time = queue.pop(0)
            
            for neighbor, edge_idx, propagation_weight in self.adjacency_list[current_node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                
                neighbor_loading = node_loading[neighbor]
                neighbor_pressure = pressures[neighbor]
                neighbor_temperature = current_temperature[neighbor]
                
                failure_state, reason = self.check_node_state(
                    neighbor, neighbor_pressure, neighbor_loading, neighbor_temperature
                )
                
                if failure_state == 2:
                    physical_delay = np.random.uniform(0.1, 0.5)
                    failure_time = current_time + physical_delay

                    failure_sequence.append((neighbor, failure_time, reason, current_node))
                    failed_nodes.add(neighbor)
                    queue.append((neighbor, failure_time))
                    
                    injections[neighbor] = 0.0
                    extractions[neighbor] = 0.0
                    
                    failed_lines = get_failed_lines_from_nodes(edge_index, failed_nodes | extra_failed)

                    pressures, pipe_flows, pump_heads, is_stable = fluid_flow_simulator.compute_fluid_flow(
                        injections, extractions,
                        failed_lines=failed_lines,
                        failed_nodes=list(failed_nodes | extra_failed)
                    )
                    
                    if not is_stable:
                        return failure_sequence
                    
                    flow_ratios = np.abs(pipe_flows) / (flow_limits + 1e-6)
                    node_loading = self._calculate_node_loading(edge_index, flow_ratios)
                    
                    if len(failed_nodes) >= target_num_failures:
                        break
        
        return failure_sequence
    
    def _calculate_node_loading(self, edge_index: np.ndarray, flow_ratios: np.ndarray) -> np.ndarray:
        node_loading = np.zeros(self.num_nodes)
        src, dst = edge_index
        for i in range(len(flow_ratios)):
            s, d = int(src[i]), int(dst[i])
            node_loading[s] = max(node_loading[s], flow_ratios[i])
            node_loading[d] = max(node_loading[d], flow_ratios[i])
        return node_loading


def create_adjacency_list(
    edge_index: np.ndarray,
    node_types: np.ndarray,
    propagation_weights: Optional[np.ndarray] = None
) -> List[List[Tuple[int, int, float]]]:
    num_nodes = node_types.shape[0]
    num_edges = edge_index.shape[1]
    
    adjacency_list = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    
    for i in range(num_edges):
        s, d = int(src[i]), int(dst[i])
        weight = 0.8
        if node_types[s] == 1 and node_types[d] == 2:
            weight = 0.9
        elif node_types[s] == 2 and node_types[d] == 0:
            weight = 0.7
        
        adjacency_list[s].append((d, i, weight))
        adjacency_list[d].append((s, i, weight))
    
    return adjacency_list