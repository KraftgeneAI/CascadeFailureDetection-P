"""
Topology Module (Pipeline Digital Twin)
=======================================
Pipeline topology generation and initialization.

Handles:
- Linear, long-haul transmission pipeline generation (Single continuous line).
- Node assignment: 0=Delivery (Terminal), 1=Source/Booster (Pump Station), 2=Pipe Segment/Valve.
- Pipe feature initialization (Length, Flow Capacity, etc.).
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import Settings


class PipelineTopologyGenerator:
    """
    Generates a linear, long-haul pipeline transmission network.
    
    Creates a single continuous pipeline with:
    - Node 0 connected to Node 1, Node 1 to Node 2, ..., Node N-1 to Node N.
    - No meshed zones or random tie-lines.
    """
    
    def __init__(self, num_nodes: int = Settings.Topology.DEFAULT_NUM_NODES, seed: int = Settings.Scenario.DEFAULT_SEED):
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_topology(self) -> Dict:
        """
        Generate complete linear pipeline topology.
        """
        # Generate linear adjacency matrix
        adj = self._generate_linear_topology()
        
        # Convert to edge index
        edge_index = self._adjacency_to_edge_index(adj)
        
        # Generate snaking geographic positions
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

    def _generate_linear_topology(self) -> np.ndarray:
        """
        Generate a strictly linear point-to-point topology.
        Node 0 <-> Node 1 <-> Node 2 ... <-> Node N-1
        """
        adj = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1  # Keep bidirectional for hydraulic matrix symmetry
            
        return adj
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> torch.Tensor:
        edges = np.where(adj > 0)
        return torch.tensor(np.vstack(edges), dtype=torch.long)
    
    def _generate_geographic_positions(self) -> np.ndarray:
        """
        Generate coordinates that form a snaking path (like a real pipeline routing).
        """
        positions = np.zeros((self.num_nodes, 2))
        
        # Start at an arbitrary point
        current_pos = np.array([-50.0, -50.0])
        positions[0] = current_pos
        
        # Initial direction vector (e.g., heading North-East)
        direction = np.array([1.0, 0.5]) 
        direction = direction / np.linalg.norm(direction)

        for i in range(1, self.num_nodes):
            # Step forward with some distance (e.g., 5 to 15 km per segment)
            step_size = np.random.uniform(5.0, 15.0) 
            
            # Add minor lateral jitter for terrain variations
            jitter = np.random.randn(2) * 1.5 
            
            current_pos = current_pos + (direction * step_size) + jitter
            positions[i] = current_pos
            
            # Slowly curve the main direction to make a natural snaking pipeline
            theta = np.random.normal(0, 0.15) # Small angle rotation in radians
            c, s = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[c, -s], [s, c]])
            direction = np.dot(rot_matrix, direction)
            
        return positions


class PipelinePropertyInitializer:
    """
    Initializes physical pipeline properties along the single line.
    """
    
    def __init__(self, num_nodes: int, seed: int = Settings.Scenario.DEFAULT_SEED):
        self.num_nodes = num_nodes
        np.random.seed(seed)
    
    def initialize_properties(self) -> Dict:
        self.node_types, pump_indices = self._assign_node_types()
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
        Assign node types along the linear path:
        Node 0 = Main Injection Pump Station (Type 1)
        Node N-1 = Delivery Terminal (Type 0)
        Middle Nodes = Pipe Segments/Valves (Type 2), with occasional Booster Pumps (Type 1)
        """
        node_types = np.full(self.num_nodes, 2, dtype=int)
        
        # Start is the main injection pump
        node_types[0] = 1
        
        # End is the delivery terminal
        node_types[-1] = 0
        
        # Place booster pump stations approximately every 20-30 nodes
        booster_indices = []
        for i in range(25, self.num_nodes - 10, 25):
            # Add some randomness to exact placement
            idx = i + np.random.randint(-5, 5)
            node_types[idx] = 1
            booster_indices.append(idx)
            
        all_pump_indices = np.array([0] + booster_indices)
        
        return node_types, all_pump_indices
    
    def _calculate_base_flow(self, node_types: np.ndarray) -> np.ndarray:
        """Only the final terminal extracts fluid."""
        base_flow = np.zeros(self.num_nodes)
        # All demand is pulled from the very end of the line
        base_flow[-1] = np.random.uniform(1000, 2000)
        return base_flow
    
    def _size_pumps(self, base_flow: np.ndarray, pump_indices: np.ndarray) -> np.ndarray:
        """Distribute pumping capacity across the main injection and boosters."""
        total_demand = base_flow[-1]
        target_total_capacity = total_demand * 1.50 # 50% headroom
        
        pump_capacity = np.zeros(self.num_nodes)
        
        # Give the main injection pump slightly more weight, distribute rest to boosters
        weights = np.ones(len(pump_indices))
        weights[0] = 2.0 
        weights = weights / weights.sum()
        
        for i, idx in enumerate(pump_indices):
            pump_capacity[idx] = target_total_capacity * weights[i]
            
        return pump_capacity
    
    def _initialize_failure_thresholds(self) -> Dict:
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