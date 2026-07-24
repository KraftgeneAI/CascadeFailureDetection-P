"""
Physics Module (Pipeline Digital Twin)
======================================
Physics-based simulation models for fluid network dynamics.
"""

import numpy as np
from typing import Tuple, Optional, List


class FluidFlowSimulator:
    """
    Hydraulic fluid flow simulator for pipeline networks.
    Solves L * P = F_net for pressure distributions and pipe flows.
    """
    
    def __init__(
        self,
        num_nodes: int,
        edge_index: np.ndarray,
        positions: np.ndarray,
        node_types: np.ndarray,
        pump_capacity: np.ndarray,
        pipe_resistance: np.ndarray,
        flow_limits: np.ndarray,
        reference_pressure: float = 1000.0
    ):
        self.num_nodes = num_nodes
        self.num_edges = edge_index.shape[1]
        self.edge_index = edge_index
        self.node_types = node_types
        self.pump_capacity = pump_capacity
        self.pipe_resistance = pipe_resistance
        self.flow_limits = flow_limits
        self.reference_pressure = reference_pressure
        
    def compute_fluid_flow(
        self,
        injections: np.ndarray,
        extractions: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Compute steady-state fluid pressures and flows.
        """
        failed_line_set = set(failed_lines) if failed_lines else set()
        failed_node_set = set(failed_nodes) if failed_nodes else set()
        
        # Net flow vector (mass conservation)
        net_flow = injections - extractions
        for node in failed_node_set:
            net_flow[node] = 0.0
            
        # Build Hydraulic Laplacian Matrix
        laplacian = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        src, dst = self.edge_index
        
        for i in range(self.num_edges):
            u, v = int(src[i]), int(dst[i])
            if i in failed_line_set or u in failed_node_set or v in failed_node_set:
                continue
                
            conductance = 1.0 / (self.pipe_resistance[i] + 1e-6)
            laplacian[u, u] += conductance
            laplacian[v, v] += conductance
            laplacian[u, v] -= conductance
            laplacian[v, u] -= conductance
            
        # Find active pump nodes
        active_pumps = [i for i in range(self.num_nodes) if self.node_types[i] == 1 and i not in failed_node_set]
        
        if not active_pumps:
            return self._get_default_results()
            
        # Set ALL active pump stations as pressure booster setpoints (1000 PSI)
        for p_node in active_pumps:
            laplacian[p_node, :] = 0.0
            laplacian[p_node, p_node] = 1.0
            net_flow[p_node] = self.reference_pressure
        
        try:
            # Solve L * P = F_net
            pressures = np.linalg.pinv(laplacian) @ net_flow
            is_stable = True
            
            # Pipe flows: Q_ij = (P_u - P_v) / R_ij
            pipe_flows = np.zeros(self.num_edges, dtype=np.float64)
            for i in range(self.num_edges):
                u, v = int(src[i]), int(dst[i])
                if i in failed_line_set or u in failed_node_set or v in failed_node_set:
                    pipe_flows[i] = 0.0
                else:
                    pipe_flows[i] = (pressures[u] - pressures[v]) / (self.pipe_resistance[i] + 1e-6)
                    
            pump_heads = np.zeros(self.num_nodes, dtype=np.float64)
            for p_node in active_pumps:
                pump_heads[p_node] = self.reference_pressure
                
            for node in failed_node_set:
                pressures[node] = 0.0
                
            return pressures, pipe_flows, pump_heads, is_stable
            
        except np.linalg.LinAlgError:
            return self._get_default_results()
            
    def _get_default_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        return (
            np.zeros(self.num_nodes, dtype=np.float64),
            np.zeros(self.num_edges, dtype=np.float64),
            np.zeros(self.num_nodes, dtype=np.float64),
            False
        )


class SurgeDynamicsSimulator:
    """Simulates transient pressure surge dynamics."""
    
    def __init__(self, num_nodes: int, node_types: np.ndarray, pump_capacity: np.ndarray, base_surge_metric: float = 1.0):
        self.num_nodes = num_nodes
        self.node_types = node_types
        self.pump_capacity = pump_capacity
        self.base_surge_metric = base_surge_metric
        self.damping = np.random.uniform(0.05, 0.15, num_nodes)
        self.current_surge = base_surge_metric

    def reset_surge(self):
        self.current_surge = self.base_surge_metric
    
    def update_surge_state(self, injections: np.ndarray, extractions: np.ndarray, current_surge_state: float, dt: float = 1.0) -> Tuple[float, np.ndarray]:
        total_inj = injections.sum()
        total_ext = extractions.sum()
        imbalance = total_inj - total_ext

        if total_inj == 0:
            return 0.0, extractions 

        system_base = max(total_ext, 1.0)
        surge_deviation = np.clip(imbalance / (system_base * np.mean(self.damping)), -0.5, 0.5)
        target_surge = self.base_surge_metric + surge_deviation

        tau_surge = 0.5 
        alpha = 1.0 - np.exp(-dt / tau_surge)
        
        new_surge_state = np.clip(current_surge_state + alpha * (target_surge - current_surge_state), 0.0, 2.0)

        adjusted_extractions = extractions.copy()
        if new_surge_state < 0.8:
            adjusted_extractions *= max(0.5, new_surge_state)

        return new_surge_state, adjusted_extractions


class ThermalDynamicsSimulator:
    """Pipeline thermal dynamics simulator."""
    
    def __init__(self, num_nodes: int, thermal_time_constant: np.ndarray, thermal_capacity: np.ndarray, cooling_effectiveness: np.ndarray, ambient_temperature: float = 15.0):
        self.num_nodes = num_nodes
        self.thermal_time_constant = thermal_time_constant
        self.thermal_capacity = thermal_capacity
        self.cooling_effectiveness = cooling_effectiveness
        self.ambient_temperature = ambient_temperature
        self.temperatures = np.full(num_nodes, ambient_temperature)
    
    def update_temperatures(self, friction_heat: np.ndarray, dt: float = 2.0) -> np.ndarray:
        temp_diff = self.temperatures - self.ambient_temperature
        heat_dissipation = (self.cooling_effectiveness * temp_diff) / self.thermal_time_constant
        
        dT = (friction_heat / self.thermal_capacity - heat_dissipation) * dt
        self.temperatures += dT + np.random.normal(0, 0.2, self.num_nodes)
        self.temperatures = np.clip(self.temperatures, self.ambient_temperature - 5.0, 150.0)
        return self.temperatures
    
    def reset_temperatures(self):
        self.temperatures = np.full(self.num_nodes, self.ambient_temperature)