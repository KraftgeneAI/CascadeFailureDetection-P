"""
Physics-Based Pipeline Simulator Module
=======================================
Orchestrates the complete physics-based simulation of pipeline cascade failures.

SIMULATION PROCESS:
-------------------
1. Initialize pipeline topology and physical properties.
2. Determine scenario type based on stress level.
3. Run time-series loop, updating fluid dynamics (flow, pressure, surge, temperature).
4. Check for initial fluid failures (ruptures, cavitation).
5. Propagate cascade dynamically if failures occur.
6. Package SCADA telemetry and compute ground truth risk labels.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from .topology import PipelineTopologyGenerator, PipelinePropertyInitializer
from .physics import FluidFlowSimulator, SurgeDynamicsSimulator, ThermalDynamicsSimulator
from .cascade import CascadeSimulator, create_adjacency_list
from .utils import get_failed_lines_from_nodes
from .config import Settings


class PhysicsBasedPipelineSimulator:
    """
    Complete physics-based pipeline simulator for SCADA data generation.
    """
    
    def __init__(
        self,
        num_nodes: int = Settings.Topology.DEFAULT_NUM_NODES,
        seed: int = Settings.Scenario.DEFAULT_SEED,
        topology_file: Optional[str] = None
    ):
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize topology
        print(f"Initializing pipeline topology...")
        topo_gen = PipelineTopologyGenerator(num_nodes, seed)
        
        if topology_file:
            topo_data = topo_gen.load_topology(topology_file)
            if topo_data is None:
                topo_data = topo_gen.generate_topology()
                import pickle, pathlib
                pathlib.Path(topology_file).parent.mkdir(parents=True, exist_ok=True)
                with open(topology_file, 'wb') as f:
                    pickle.dump(topo_data, f)
            self.adjacency_matrix = topo_data['adjacency_matrix']
            ei = topo_data['edge_index']
            self.edge_index = torch.from_numpy(ei).long() if isinstance(ei, np.ndarray) else ei.long()
            self.positions = topo_data['positions']
            self.num_nodes = self.adjacency_matrix.shape[0]
        else:
            topo_data = topo_gen.generate_topology()
            self.adjacency_matrix = topo_data['adjacency_matrix']
            self.edge_index = topo_data['edge_index']
            self.positions = topo_data['positions']
        
        self.num_edges = self.edge_index.shape[1]
        
        # Initialize pipeline properties
        print(f"Initializing pipeline properties...")
        prop_init = PipelinePropertyInitializer(self.num_nodes, seed)
        props = prop_init.initialize_properties()
        
        self.node_types = props['node_types']
        self.pump_capacity = props['pump_capacity']
        self.base_flow = props['base_flow']
        self.equipment_age = props['equipment_age']
        self.equipment_condition = props['equipment_condition']
        
        # Thresholds
        self.pressure_fail = props['pressure_failure_threshold']
        self.pressure_dmg = props['pressure_damage_threshold']
        self.flow_fail = props['flow_failure_threshold']
        self.flow_dmg = props['flow_damage_threshold']
        self.temp_fail = props['temperature_failure_threshold']
        self.temp_dmg = props['temperature_damage_threshold']
        
        # Pipe properties (Hydraulic Resistance and Flow Limits)
        src, dst = self.edge_index
        distances = np.linalg.norm(self.positions[src] - self.positions[dst], axis=1)
        dist_norm = distances / (distances.max() + 1e-6)
        
        # Low hydraulic resistance for well-conditioned pressure distribution
        self.pipe_resistance = 0.0005 + dist_norm * 0.002
        avg_flow = self.base_flow.sum() / self.num_edges
        
        # Increase the pipe capacity buffer so normal flows don't trigger erosion/ruptures
        self.flow_limits = avg_flow * np.random.uniform(2.5, 4.0, self.num_edges)
        
        self.decommissioned_nodes = set()
        self._init_simulators()

    def _init_simulators(self) -> None:
        """Initialize physics and cascade sub-modules."""
        print(f"Initializing physics simulators...")
        self.fluid_sim = FluidFlowSimulator(
            self.num_nodes, self.edge_index.numpy(), self.positions, 
            self.node_types, self.pump_capacity, self.pipe_resistance, self.flow_limits
        )
        
        self.surge_sim = SurgeDynamicsSimulator(
            self.num_nodes, self.node_types, self.pump_capacity
        )
        
        self.thermal_sim = ThermalDynamicsSimulator(
            self.num_nodes, 
            thermal_time_constant=np.full(self.num_nodes, 20.0),
            thermal_capacity=np.full(self.num_nodes, 1.0),
            cooling_effectiveness=np.full(self.num_nodes, 0.8)
        )
        
        print(f"Initializing cascade simulator...")
        adjacency_list = create_adjacency_list(self.edge_index.numpy(), self.node_types)
        self.cascade_sim = CascadeSimulator(
            self.num_nodes, adjacency_list,
            self.pressure_fail, self.pressure_dmg,
            self.flow_fail, self.flow_dmg,
            self.temp_fail, self.temp_dmg
        )

    def generate_scenario(self, stress_level: float, sequence_length: int = 30) -> Optional[Dict]:
        """Generate a complete pipeline scenario."""
        print(f"  [INPUT] Generating scenario with stress_level: {stress_level:.3f}")

        scenario_data = self._generate_time_series(stress_level, sequence_length)
        if scenario_data is None:
            return None

        failed_nodes = scenario_data.pop('failed_nodes')
        failure_times = scenario_data.pop('failure_times')
        failure_reasons = scenario_data.pop('failure_reasons')
        cascade_start_time = scenario_data.pop('actual_cascade_start')
        is_cascade = len(failed_nodes) > 0

        ground_truth_risk = self._compute_node_risk_vectors(
            scenario_data['sequence'], failed_nodes, failure_times, sequence_length
        )

        scenario_data['metadata'] = {
            'cascade_start_time': cascade_start_time,
            'failed_nodes': failed_nodes,
            'failure_times': failure_times,
            'failure_reasons': failure_reasons,
            'ground_truth_risk': ground_truth_risk,
            'is_cascade': is_cascade,
            'stress_level': stress_level,
            'num_nodes': self.num_nodes,
            'num_edges': len(self.edge_index[0])
        }

        return scenario_data
    
    def _generate_time_series(self, stress_level: float, sequence_length: int) -> Optional[Dict]:
        sequence = []
        self.surge_sim.reset_surge()
        self.thermal_sim.reset_temperatures()

        injections = np.zeros(self.num_nodes)
        extractions = np.zeros(self.num_nodes)
        cumulative_failed_nodes = set()
        
        failure_record = {}
        cascade_start_time = -1

        for t in range(sequence_length):
            current_stress = stress_level * min(1.0, (t + 1) / (sequence_length * 0.8))
            
            # Add noise based on stress
            noise_level = 0.05 if current_stress > 0.72 else 0.02
            noise = np.clip(np.random.normal(0, noise_level, self.num_nodes), -noise_level, noise_level)
            
            # Setup extractions (deliveries) and injections (pumps)
            extractions = self.base_flow * (1.0 + current_stress * 0.4) * (1 + noise)
            for n in cumulative_failed_nodes | self.decommissioned_nodes:
                extractions[n] = 0.0
                
            total_extraction = extractions.sum()
            active_pumps = [i for i in range(self.num_nodes) if self.node_types[i] == 1 and i not in cumulative_failed_nodes]
            active_pump_cap = sum(self.pump_capacity[i] for i in active_pumps)
            
            injections[:] = 0.0
            for idx in active_pumps:
                if active_pump_cap > 0:
                    injections[idx] = (self.pump_capacity[idx] / active_pump_cap) * total_extraction

            failed_lines_t = get_failed_lines_from_nodes(self.edge_index.numpy(), cumulative_failed_nodes)
            
            # 1. Physics Solvers
            pressures, pipe_flows, pump_heads, is_stable = self.fluid_sim.compute_fluid_flow(
                injections, extractions, failed_lines_t, list(cumulative_failed_nodes | self.decommissioned_nodes)
            )
            
            if not is_stable:
                if len(cumulative_failed_nodes) > 0:
                    print(f"  [COLLAPSE] System collapsed at t={t}")
                else:
                    return None
                    
            surge_metric, extractions = self.surge_sim.update_surge_state(injections, extractions, self.surge_sim.current_surge, 1.0)
            
            # Friction heat proxy based on flow velocity
            flow_ratios = np.abs(pipe_flows) / (self.flow_limits + 1e-6)
            friction_heat = np.zeros(self.num_nodes)
            np.add.at(friction_heat, self.edge_index[0].numpy(), flow_ratios**2)
            np.add.at(friction_heat, self.edge_index[1].numpy(), flow_ratios**2)
            equipment_temps = self.thermal_sim.update_temperatures(friction_heat, 1.0)
            
            # 2. Trigger Check
            node_flow_ratios = np.zeros(self.num_nodes)
            np.maximum.at(node_flow_ratios, self.edge_index[0].numpy(), flow_ratios)
            np.maximum.at(node_flow_ratios, self.edge_index[1].numpy(), flow_ratios)
            
            new_failures = []
            for n in range(self.num_nodes):
                if n in cumulative_failed_nodes or n in self.decommissioned_nodes:
                    continue
                    
                state, reason = self.cascade_sim.check_node_state(
                    n, pressures[n], node_flow_ratios[n], equipment_temps[n]
                )
                
                if state == 2 or (surge_metric > 1.8 and self.node_types[n] == 1):
                    reason = "surge_failure" if surge_metric > 1.8 else reason
                    new_failures.append((n, reason))
                    
            # 3. Cascade Propagation
            if new_failures:
                target_max = min(len(new_failures) + int(self.num_nodes * 0.3), self.num_nodes)
                cascade_seq = self.cascade_sim.propagate_cascade_physics(
                    new_failures, injections, extractions, equipment_temps, target_max,
                    self.fluid_sim, self.edge_index.numpy(), self.flow_limits, self.decommissioned_nodes
                )
                
                for fail_node, fail_time, fail_reason, fail_parent in cascade_seq:
                    if fail_node not in cumulative_failed_nodes:
                        if cascade_start_time < 0:
                            cascade_start_time = t
                        failure_record[fail_node] = (t, fail_reason, fail_parent)
                        cumulative_failed_nodes.add(fail_node)
                        injections[fail_node] = 0.0
                        extractions[fail_node] = 0.0

            # 4. Package Data
            timing = np.full(self.num_nodes, -1.0, dtype=np.float32)
            for n, (ft, _, _) in failure_record.items():
                timing[n] = float(t - ft)
                
            sequence.append({
                'scada_data': np.column_stack([
                    pressures,                            # 0: Pressure
                    pump_heads,                           # 1: Pump Head
                    injections,                           # 2: Fluid injected
                    extractions,                          # 3: Fluid extracted
                    injections - extractions,             # 4: Net Flow
                    equipment_temps,                      # 5: Temp
                    np.full(self.num_nodes, surge_metric),# 6: Surge state
                    self.equipment_age,                   # 7
                    self.equipment_condition,             # 8
                    self.pump_capacity,                   # 9
                    self.base_flow,                       # 10
                    self.node_types,                      # 11
                    np.full(self.num_nodes, t/sequence_length), # 12
                    np.full(self.num_nodes, current_stress),    # 13
                    pressures / self.pressure_fail,       # 14
                    equipment_temps / self.temp_fail,     # 15
                    node_flow_ratios,                     # 16
                    np.full(self.num_nodes, surge_metric / 1.5) # 17
                ]).astype(np.float32),
                'edge_attr': np.column_stack([
                    self.pipe_resistance, self.flow_limits, pipe_flows
                ]).astype(np.float32),
                'node_labels': np.array([1.0 if n in cumulative_failed_nodes else 0.0 for n in range(self.num_nodes)], dtype=np.float32),
                'cascade_timing': timing
            })

        failed_nodes_out = sorted(failure_record.keys(), key=lambda n: failure_record[n][0])
        return {
            'sequence': sequence,
            'edge_index': self.edge_index.numpy(),
            'failed_nodes': failed_nodes_out,
            'failure_times': [failure_record[n][0] for n in failed_nodes_out],
            'failure_reasons': [failure_record[n][1] for n in failed_nodes_out],
            'actual_cascade_start': cascade_start_time
        }

    def _compute_node_risk_vectors(self, sequence, failed_nodes, failure_times, sequence_length):
        last_scada = sequence[-1]['scada_data']
        p_ratio = last_scada[:, 14]
        f_ratio = last_scada[:, 16]
        
        risk = np.zeros((self.num_nodes, 7), dtype=np.float32)
        risk[:, 0] = np.clip(np.maximum(p_ratio, f_ratio), 0.0, 1.0) # Threat
        risk[:, 1] = np.clip(1.0 - self.equipment_condition, 0.0, 1.0) # Vulnerability
        risk[:, 2] = np.clip(self.base_flow / (self.base_flow.max() + 1e-6), 0.0, 1.0) # Impact
        risk[:, 3] = np.clip(f_ratio, 0.0, 1.0) # Prob
        risk[failed_nodes, 3] = 1.0
        risk[:, 4] = np.clip(1.0 - f_ratio, 0.0, 1.0) # Headroom
        risk[:, 5] = np.clip(1.0 - np.maximum(p_ratio, f_ratio), 0.0, 1.0) # Safety Margin
        for n, t_fail in zip(failed_nodes, failure_times):
            risk[n, 6] = np.clip(1.0 - (t_fail / max(sequence_length, 1)), 0.0, 1.0) # Urgency
            
        return risk