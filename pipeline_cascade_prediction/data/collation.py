"""
Collation Module (Pipeline Digital Twin)
========================================
Batch collation functions for DataLoader.
"""

import torch
import numpy as np
from typing import List, Dict, Any

_TEMPORAL_KEYS = {
    'scada_data', 'edge_mask', 'temporal_sequence',
    'edge_attr', 'node_features'
}

def collate_cascade_batch(batch: List[Any]) -> Dict[str, torch.Tensor]:
    """Collate a batch by truncating all temporal sequences to the minimum sequence length."""
    
    # 1. FLATTEN THE BATCH (Fix for the list-of-lists error)
    flat_batch = []
    for item in batch:
        if isinstance(item, list):
            flat_batch.extend(item)
        else:
            flat_batch.append(item)
            
    # 2. Filter out empty items
    flat_batch = [item for item in flat_batch if item]
    batch_dict = {}
    if not flat_batch:
        return batch_dict

    # Now it's safe to get keys!
    keys = flat_batch[0].keys()

    # Pass 1: compute a single global min_len across ALL temporal keys
    global_min_len = None
    for key in _TEMPORAL_KEYS:
        if not all(key in item for item in flat_batch):
            continue
        items = [item[key] for item in flat_batch]
        if not isinstance(items[0], torch.Tensor) or items[0].dim() < 1:
            continue
        key_min = min(item.shape[0] for item in items)
        if global_min_len is None or key_min < global_min_len:
            global_min_len = key_min

    # Pass 2: collate each key
    for key in keys:
        if key == 'temporal_sequence':
            continue

        if key == 'edge_index':
            edge_index = flat_batch[0]['edge_index']
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            batch_dict['edge_index'] = edge_index

        elif key == 'sequence_length':
            batch_dict['sequence_length'] = torch.tensor(
                [item['sequence_length'] for item in flat_batch], dtype=torch.long
            )

        elif key == 'parent_labels':
            items = [item[key] for item in flat_batch if key in item]
            if items:
                batch_dict[key] = torch.stack(items, dim=0)

        elif key == 'graph_properties':
            graph_props_batch = {}
            if flat_batch[0]['graph_properties']:
                for prop_key in flat_batch[0]['graph_properties'].keys():
                    props = [
                        item['graph_properties'][prop_key] for item in flat_batch
                        if prop_key in item['graph_properties']
                    ]
                    if props:
                        if isinstance(props[0], torch.Tensor):
                            try:
                                graph_props_batch[prop_key] = torch.stack(props, dim=0)
                            except RuntimeError:
                                graph_props_batch[prop_key] = props[0]
                        else:
                            graph_props_batch[prop_key] = torch.from_numpy(np.array(props)).float()
            batch_dict[key] = graph_props_batch

        else:
            if not all(key in item for item in flat_batch):
                continue

            items = [item[key] for item in flat_batch]

            if not isinstance(items[0], torch.Tensor):
                try:
                    if isinstance(items[0], np.ndarray):
                        items = [torch.from_numpy(np.array(items)[i]).float() for i in range(len(items))]
                    else:
                        items = [torch.tensor(item, dtype=torch.float32) if not isinstance(item, torch.Tensor) else item for item in items]
                except Exception:
                    continue

            # Temporal keys: truncate every item to global_min_len
            if key in _TEMPORAL_KEYS and global_min_len is not None and items[0].dim() >= 1:
                batch_dict[key] = torch.stack([item[:global_min_len] for item in items], dim=0)
                continue

            # Non-temporal keys: stack directly
            try:
                batch_dict[key] = torch.stack(items, dim=0)
            except Exception:
                try:
                    batch_dict[key] = torch.cat(items, dim=0)
                except Exception:
                    pass

    if 'scada_data' in batch_dict:
        batch_dict['temporal_sequence'] = batch_dict['scada_data']

    return batch_dict
