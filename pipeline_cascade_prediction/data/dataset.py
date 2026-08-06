"""
Dataset Module (Pipeline Digital Twin)
======================================
Memory-efficient dataset loader for pre-generated pipeline cascade data.
"""

import torch
from torch.utils.data import Dataset
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import glob
import json

from pipeline_cascade_prediction.data.generator.config import Settings

# --- Embedded Preprocessing Helpers ---
def normalize_pressure(p, base=Settings.PipelineSystem.REFERENCE_PRESSURE_PSI): return p / base
def normalize_flow(f, base=Settings.PipelineSystem.BASE_FLOW_BBL_HR): return f / base

def calculate_truncation_window(seq_len: int, cascade_start_time: int, is_cascade: bool, min_ratio: float = 0.3) -> Tuple[int, int]:
    """Ensures model only sees PRE-cascade data to learn prediction, not detection."""
    min_len = max(10, int(seq_len * min_ratio))
    if is_cascade:
        hard_limit = max(min_len, int(cascade_start_time) - 5)
        end_idx = np.random.randint(min_len, hard_limit + 1)
    else:
        global_max = max(min_len + 1, int(seq_len * 0.85) - 5)
        end_idx = np.random.randint(min_len, global_max + 1)
    
    max_start = end_idx - 10
    start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
    return start_idx, end_idx

class CascadeDataset(Dataset):
    """Memory-efficient PyTorch Dataset for pipeline cascade scenarios."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.scenario_files = sorted(glob.glob(str(self.data_dir / "scenarios_batch_*.pkl")))
        
        cache_file = self.data_dir / "metadata_cache.json"
        self.cascade_labels = []

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.cascade_labels = json.load(f)

        if not self.cascade_labels:
            print(f"Scanning files in {data_dir} for metadata cache...")
            for f in self.scenario_files:
                try:
                    with open(f, 'rb') as file:
                        batch = pickle.load(file)
                    self.cascade_labels.extend([bool(s['metadata'].get('is_cascade', False)) for s in batch])
                except Exception:
                    pass
            with open(cache_file, 'w') as f:
                json.dump(self.cascade_labels, f)

    def __len__(self) -> int:
        return len(self.scenario_files)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        file = self.scenario_files[idx]
        try:
            with open(file, 'rb') as f:
                batch = pickle.load(f)
            # Return processed list (Collation will flatten this)
            return [self._process_sequence(s) for s in batch]
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return []
            
    def _process_sequence(self, scenario: Dict) -> Dict[str, Any]:
        seq_original = scenario['sequence']
        metadata = scenario['metadata']
        edge_index = scenario['edge_index']
        num_nodes = metadata['num_nodes']
        num_edges = metadata['num_edges']
        
        start_idx, end_idx = calculate_truncation_window(
            len(seq_original), metadata.get('cascade_start_time', -1), metadata.get('is_cascade', False)
        )
        seq = seq_original[start_idx:end_idx] if len(seq_original[start_idx:end_idx]) > 0 else seq_original[:10]
        
        T = len(seq)
        
        # 1. Initialize padded tensors for UnifiedModel compatibility
        # Model expects: node_features [T, N, 124], edge_attr [T, E, 7]
        node_features = torch.zeros((T, num_nodes, 124), dtype=torch.float32)
        edge_attr_padded = torch.zeros((T, num_edges, 7), dtype=torch.float32)
        edge_mask = torch.ones((T, num_edges), dtype=torch.float32)
        
        for t, step in enumerate(seq):
            # Process Nodes
            scada_raw = step.get('scada_data', np.zeros((num_nodes, 18)))
            # Normalizing pressure (col 0, 1) and flow (col 2, 3, 4)
            scada_raw[:, 0] = normalize_pressure(scada_raw[:, 0])
            scada_raw[:, 1] = normalize_pressure(scada_raw[:, 1])
            scada_raw[:, 2:5] = normalize_flow(scada_raw[:, 2:5])
            
            # Map SCADA into the 124-dim tensor (Mapping to base slots 0:18)
            node_features[t, :, :min(18, scada_raw.shape[1])] = torch.from_numpy(scada_raw[:, :18]).float()
            
            # Process Edges
            ea_raw = step.get('edge_attr', np.zeros((num_edges, 3)))
            # Normalizing flow limits and active flows
            ea_raw[:, 1:3] = normalize_flow(ea_raw[:, 1:3])
            
            # Map edge attributes into 7-dim tensor
            edge_attr_padded[t, :, :min(3, ea_raw.shape[1])] = torch.from_numpy(ea_raw[:, :3]).float()
            
            # Calculate Edge Mask based on loading ratio (Flow / Limit)
            thermal = np.abs(ea_raw[:, 1]) + 1e-6
            loading = np.abs(ea_raw[:, 2]) / thermal
            edge_mask[t] = torch.from_numpy(np.clip(1.0 - loading, 0.0, 1.0)).float()

        # Labels
        final_labels = torch.from_numpy(seq_original[-1].get('node_labels', np.zeros(num_nodes))).float()
        
        timing_tensor = torch.full((num_nodes,), -1.0, dtype=torch.float32)
        parent_labels = torch.full((num_nodes,), -1, dtype=torch.long)

        if metadata.get('is_cascade', False):
            failed = list(metadata.get('failed_nodes', []))
            # Sub-timestep resolution when available; integer timesteps otherwise.
            times = metadata.get('failure_times_exact')
            if not times:
                times = metadata.get('failure_times', [])

            if failed:
                # Positional assignment. A boolean mask would write in ascending
                # node-index order while `times` is in failure-time order, silently
                # pairing each node with another node's failure time.
                idx = torch.as_tensor(failed, dtype=torch.long)
                norm_times = torch.clamp(
                    torch.as_tensor(times, dtype=torch.float32) / max(len(seq_original), 1),
                    0.0, 1.0,
                )
                timing_tensor[idx] = norm_times

                # Causal parent targets for ParentPredictionHead. Class `num_nodes`
                # is the "trigger / no parent" class; -1 means ignore (see
                # PhysicsInformedLoss, which masks on label != -1). Without this the
                # parent loss never fires and the head goes unsupervised.
                parents = metadata.get('failure_parents')
                if parents is not None:
                    for node, parent in zip(failed, parents):
                        parent_labels[node] = (
                            num_nodes if parent is None or int(parent) < 0 else int(parent)
                        )

        return {
            'scada_data': node_features[:, :, :18], # Keep raw scada for physics targets
            'node_features': node_features,
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': edge_attr_padded,
            'edge_mask': edge_mask,
            'node_failure_labels': final_labels,
            'cascade_timing': timing_tensor,
            'parent_labels': parent_labels,
            'ground_truth_risk': torch.zeros(7, dtype=torch.float32),
            'graph_properties': {},
            'temporal_sequence': node_features[:, :, :18],
            'sequence_length': T,
        }