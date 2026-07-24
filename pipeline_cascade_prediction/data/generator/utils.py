"""
Generator Utilities (Pipeline Digital Twin)
===========================================
Utility functions for pipeline data generation.
"""

import psutil
import warnings
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class MemoryMonitor:
    """
    Monitor memory usage to prevent Out-Of-Memory (OOM) errors.
    """
    
    @staticmethod
    def get_memory_usage() -> float:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def check_threshold(threshold_mb: float = 8000) -> bool:
        current = MemoryMonitor.get_memory_usage()
        if current > threshold_mb:
            warnings.warn(f"High memory usage: {current:.1f} MB")
            return True
        return False


def save_scenarios(scenarios: List[Dict], output_path: str, batch_idx: int):
    """Save pipeline scenarios to a pickle file."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"scenarios_batch_{batch_idx}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(scenarios, f)
    
    print(f"Saved batch {batch_idx}: {len(scenarios)} scenarios -> {filename}")


def load_topology(topology_file: str) -> Optional[Dict]:
    if not Path(topology_file).exists():
        return None
    with open(topology_file, 'rb') as f:
        topology = pickle.load(f)
    return topology


def save_topology(adjacency_matrix: np.ndarray, edge_index: np.ndarray, positions: np.ndarray, output_path: str):
    topology = {
        'adjacency_matrix': adjacency_matrix,
        'edge_index': edge_index,
        'positions': positions,
        'num_nodes': adjacency_matrix.shape[0],
        'num_edges': edge_index.shape[1]
    }
    with open(output_path, 'wb') as f:
        pickle.dump(topology, f)
    print(f"Saved pipeline topology: {topology['num_nodes']} nodes, {topology['num_edges']} pipes -> {output_path}")


def split_scenarios(scenarios: List[Dict], train_split: float = 0.8, val_split: float = 0.1, test_split: float = 0.1, seed: int = 42) -> Dict[str, List[Dict]]:
    assert abs(train_split + val_split + test_split - 1.0) < 0.01, "Splits must sum to 1.0"
    
    np.random.seed(seed)
    indices = np.random.permutation(len(scenarios))
    
    n_train = int(len(scenarios) * train_split)
    n_val = int(len(scenarios) * val_split)
    
    return {
        'train': [scenarios[i] for i in indices[:n_train]],
        'val': [scenarios[i] for i in indices[n_train:n_train + n_val]],
        'test': [scenarios[i] for i in indices[n_train + n_val:]]
    }


def validate_scenario(scenario: Dict) -> bool:
    """Validate SCADA pipeline scenario structure and data quality."""
    try:
        for key in ['sequence', 'edge_index', 'metadata']:
            if key not in scenario:
                print(f"Missing key: {key}")
                return False
        
        if not scenario['sequence']:
            print("Empty sequence")
            return False
        
        timestep = scenario['sequence'][0]
        if 'scada_data' not in timestep or 'edge_attr' not in timestep or 'node_labels' not in timestep:
            print(f"Missing essential pipeline timestep keys")
            return False
            
        scada = timestep['scada_data']
        if np.any(np.isnan(scada)) or np.any(np.isinf(scada)):
            print("NaN or Inf detected in SCADA telemetry")
            return False
        
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def get_failed_lines_from_nodes(edge_index: np.ndarray, failed_nodes: set) -> List[int]:
    """Get pipe indices connected to any closed valve or failed pump."""
    failed_lines = []
    for line_idx in range(edge_index.shape[1]):
        source, target = edge_index[:, line_idx]
        if source in failed_nodes or target in failed_nodes:
            failed_lines.append(line_idx)
    return failed_lines