"""
Topology Module (Pipeline Digital Twin)
=======================================
Pipeline topology generation and initialization.

Handles:
- Realistic pipeline network generation (zones and tie-lines)
- Node assignment: 0=Delivery (Tank Farm), 1=Source (Pump Station), 2=Valve/Junction
- Pipe feature initialization (Length, Diameter, Friction, Flow Capacity)
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import Settings


class PipelineTopologyGenerator:
    """
    Generates realistic meshed pipeline networks.
    
    Creates pipeline transmission networks with:
    - Multiple zones (gathering/regional areas)
    - Intra-zone connections (meshed)
    - Inter-zone tie lines (critical transmission connections)
    - Guaranteed fluid connectivity
    """
    
    def __init__(self, num_nodes: int = Settings.Topology.DEFAULT_NUM_NODES, seed: int = Settings.Scenario.DEFAULT_SEED):
        """
        Initialize topology generator.
        
        Args:
            num_nodes: Number of nodes in the pipeline network
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_topology(self) -> Dict:
        """
        Generate complete pipeline topology.
        
        Returns:
            Dictionary containing:
            - adjacency_matrix: Node connectivity
            - edge_index: Edge list format
            - positions: Geographic positions
            - num_nodes: Number of nodes
            - num_edges: Number of edges
        """
        # Generate adjacency matrix
        adj = self._generate_realistic_topology()
        
        # Ensure connectivity
        adj = self._check_and_fix_connectivity(adj)
        
        # Convert to edge index
        edge_index = self._adjacency_to_edge_index(adj)
        
        # Generate positions
        positions = self._generate_geographic_positions()
        
        return {
            'adjacency_matrix': adj,
            'edge_index': edge_index,
            'positions': positions,
            'num_nodes': self.num_nodes,
            'num_edges': edge_index.shape[1]
        }

    @classmethod
    def load_topology(
        cls,
        topology_file: str = Settings.Dataset.DEFAULT_TOPOLOGY_FILE,
    ) -> Optional[Dict]:
        """
        Load a previously saved pipeline topology from disk[cite: 18].
        """
        path = Path(topology_file)
        if not path.exists():
            print(f"  [PipelineTopologyGenerator.load_topology] File not found: {path}")
            return None

        with open(path, "rb") as f:
            topology = pickle.load(f)

        if "num_nodes" not in topology:
            topology["num_nodes"] = int(topology["adjacency_matrix"].shape[0])
        if "num_edges" not in topology:
            ei = topology["edge_index"]
            topology["num_edges"] = int(ei.shape[1] if hasattr(ei, "shape") else len(ei[0]))

        print(
            f"  Loaded topology: {topology['num_nodes']} nodes, "
            f"{topology['num_edges']} edges \u2190 {path}"
        )
        return topology

    def _generate_realistic_topology(self) -> np.ndarray:
        """
        Generate realistic meshed pipeline topology[cite: 18].
        """
        adj = np.zeros((self.num_nodes, self.num_nodes))
        
        num_zones = Settings.Topology.NUM_ZONES
        nodes_per_zone = self.num_nodes // num_zones

        for zone in range(num_zones):
            start = zone * nodes_per_zone
            end = start + nodes_per_zone if zone < num_zones - 1 else self.num_nodes

            for i in range(start, end):
                num_connections = np.random.randint(
                    Settings.Topology.INTRA_ZONE_CONN_MIN,
                    Settings.Topology.INTRA_ZONE_CONN_MAX,
                )
                possible_neighbors = list(range(start, end))
                possible_neighbors.remove(i)
                neighbors = np.random.choice(
                    possible_neighbors,
                    size=min(num_connections, len(possible_neighbors)),
                    replace=False
                )
                for j in neighbors:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        for zone in range(num_zones - 1):
            zone_end = (zone + 1) * nodes_per_zone
            next_zone_start = zone_end
            for _ in range(np.random.randint(Settings.Topology.TIE_LINES_MIN, Settings.Topology.TIE_LINES_MAX)):
                i = np.random.randint(zone * nodes_per_zone, zone_end)
                j = np.random.randint(
                    next_zone_start,
                    min(next_zone_start + nodes_per_zone, self.num_nodes)
                )
                adj[i, j] = 1
                adj[j, i] = 1
        
        return adj
    
    def _check_and_fix_connectivity(self, adj: np.ndarray) -> np.ndarray:
        """
        Ensure the graph is fully connected by adding tie-lines to prevent hydraulic solver failures[cite: 18].
        """
        num_nodes = adj.shape[0]
        visited = np.zeros(num_nodes, dtype=bool)
        q = [0]  # Start BFS from main injection node
        visited[0] = True
        component = [0]
        
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in range(num_nodes):
                if adj[u, v] > 0 and not visited[v]:
                    visited[v] = True
                    q.append(v)
                    component.append(v)
        
        if len(component) == num_nodes:
            print("  Pipeline topology is fully connected.")
            return adj
        
        print(f"  [WARNING] Pipeline topology not connected. "
              f"Found {len(component)} nodes in main component.")
        print("  Adding tie lines to connect isolated segments...")
        
        all_nodes = set(range(num_nodes))
        main_component_set = set(component)
        island_nodes = list(all_nodes - main_component_set)
        
        while island_nodes:
            island_q = [island_nodes[0]]
            visited[island_nodes[0]] = True
            current_island_component = [island_nodes[0]]
            
            head = 0
            while head < len(island_q):
                u = island_q[head]
                head += 1
                neighbors = np.where(adj[u, :] > 0)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        island_q.append(v)
                        current_island_component.append(v)
            
            island_node = current_island_component[0]
            main_node = component[np.random.randint(len(component))]
            
            adj[island_node, main_node] = 1
            adj[main_node, island_node] = 1
            print(f"    Added pipe: Node {island_node} (isolated) <-> "
                  f"Node {main_node} (main network)")
            
            island_nodes = [
                n for n in island_nodes 
                if n not in current_island_component
            ]
        
        print("  Pipeline connectivity fixed.")
        return adj
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> torch.Tensor:
        """
        Convert adjacency matrix to edge index format[cite: 18].
        """
        edges = np.where(adj > 0)
        return torch.tensor(np.vstack(edges), dtype=torch.long)
    
    def _generate_geographic_positions(self) -> np.ndarray:
        """
        Generate realistic geographic positions[cite: 18].
        """
        positions = []
        num_zones = Settings.Topology.NUM_ZONES
        nodes_per_zone = self.num_nodes // num_zones

        for zone_idx, (cx, cy) in enumerate(Settings.Topology.ZONE_CENTERS):
            start = zone_idx * nodes_per_zone
            end = start + nodes_per_zone if zone_idx < num_zones - 1 else self.num_nodes
            num_in_zone = end - start

            zone_positions = np.random.randn(num_in_zone, 2) * Settings.Topology.ZONE_SPREAD_STD + np.array([cx, cy])
            positions.append(zone_positions)
        
        return np.vstack(positions)


class PipelinePropertyInitializer:
    """
    Initializes physical pipeline properties and fluid failure thresholds.
    """
    
    def __init__(self, num_nodes: int, seed: int = Settings.Scenario.DEFAULT_SEED):
        self.num_nodes = num_nodes
        np.random.seed(seed)
    
    def initialize_properties(self) -> Dict:
        """Initialize all node and pipe properties."""
        self.node_types, pump_indices = self._assign_node_types() # <-- Added 'self.'
        base_flow = self._calculate_base_flow(self.node_types)
        pump_capacity = self._size_pumps(base_flow, pump_indices)
        thresholds = self._initialize_failure_thresholds()
        
        return {
            'node_types': self.node_types,
            'pump_capacity': pump_capacity,
            'base_flow': base_flow,
            'equipment_age': np.random.uniform(0, 40, self.num_nodes),
            'equipment_condition': np.random.uniform(0.6, 1.0, self.num_nodes),
            **thresholds
        }
    
    def _assign_node_types(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign node types: 
        0 = Delivery (Tank Farm / Refinery / Export)
        1 = Source (Pump Station / Injection Point)
        2 = Junction (Block Valve / Routing)
        """
        node_types = np.zeros(self.num_nodes, dtype=int)
        
        node_types[0] = 1
        
        num_pumps = max(int(self.num_nodes * 0.20) - 1, 3)
        possible_pump_indices = list(range(1, self.num_nodes))
        pump_indices = np.random.choice(possible_pump_indices, num_pumps, replace=False)
        node_types[pump_indices] = 1
        
        all_pump_indices = np.concatenate([[0], pump_indices])
        
        num_valves = int(self.num_nodes * 0.30)
        possible_valve_indices = [i for i in range(1, self.num_nodes) if i not in all_pump_indices]
        valve_indices = np.random.choice(possible_valve_indices, num_valves, replace=False)
        node_types[valve_indices] = 2
        
        return node_types, all_pump_indices
    
    def _calculate_base_flow(self, node_types: np.ndarray) -> np.ndarray:
        """Calculate base fluid extraction/injection rates for each node."""
        base_flow = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            if node_types[i] == 1:    # Pump Station
                base_flow[i] = 0.0
            elif node_types[i] == 2:  # Block Valve
                base_flow[i] = 0.0
            else:                     # Delivery Node
                base_flow[i] = np.random.uniform(100, 1000)
        return base_flow
    
    def _size_pumps(self, base_flow: np.ndarray, pump_indices: np.ndarray) -> np.ndarray:
        """Size pump capacity to ensure the network can meet delivery demand with a margin."""
        total_demand = base_flow[self.node_types == 0].sum()
        target_total_capacity = total_demand * 1.50
        
        pump_capacity = np.zeros(self.num_nodes)
        weights = np.random.dirichlet(np.ones(len(pump_indices)) * 2.0)
        
        for i, idx in enumerate(pump_indices):
            pump_capacity[idx] = target_total_capacity * weights[i]
            
        return pump_capacity
    
    def _initialize_failure_thresholds(self) -> Dict:
        """
        Initialize fluid-dynamic failure thresholds.
        Replaces electrical thresholds with Pressure/Flow limits.
        """
        pressure_failure_threshold = np.random.uniform(1200, 1440, self.num_nodes)
        pressure_damage_threshold = pressure_failure_threshold - np.random.uniform(50, 100)
        
        flow_failure_threshold = np.random.uniform(1.10, 1.25, self.num_nodes)
        flow_damage_threshold = flow_failure_threshold - 0.05
        
        temperature_failure_threshold = np.random.uniform(80.0, 95.0, self.num_nodes)
        temperature_damage_threshold = temperature_failure_threshold - 10.0
        
        return {
            'pressure_failure_threshold': pressure_failure_threshold,
            'pressure_damage_threshold': pressure_damage_threshold,
            'flow_failure_threshold': flow_failure_threshold,
            'flow_damage_threshold': flow_damage_threshold,
            'temperature_failure_threshold': temperature_failure_threshold,
            'temperature_damage_threshold': temperature_damage_threshold,
        }