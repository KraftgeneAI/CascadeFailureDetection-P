"""
Pipeline State Module
=====================
Topology-embedded scenario support for fluid networks.
Maps frontend pipeline edits to the simulator's internal mathematical variables.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

TOPOLOGY_BLOCK_VERSION = 1

# Internal simulator array keys (Restored to match the physics engine's actual properties)
NODE_PROP_KEYS = [
    'node_types', 'base_flow', 'pump_capacity',
    'equipment_age', 'equipment_condition',
    'thermal_capacity', 'cooling_effectiveness', 'thermal_time_constant',
]
THRESHOLD_KEYS = [
    'pressure_failure_threshold', 'pressure_damage_threshold',
    'flow_failure_threshold', 'flow_damage_threshold',
    'temperature_failure_threshold', 'temperature_damage_threshold',
]
LINE_KEYS = [
    'pipe_resistance', 'thermal_limit_mw'
]

def build_topology_block(sim) -> Dict:
    """
    Extract the full grid parameterization from a PhysicsBasedPipelineSimulator
    into a plain-numpy topology block for embedding in a scenario file.
    Safely handles attributes missing from the pipeline simulator.
    """
    edge_index = sim.edge_index
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()

    num_nodes = int(sim.num_nodes)
    num_edges = edge_index.shape[1]

    block = {
        'version': TOPOLOGY_BLOCK_VERSION,
        'source': 'generated',
        'seed': getattr(sim, 'seed', None),
        'num_nodes': num_nodes,
        'edge_index': np.asarray(edge_index, dtype=np.int64).copy(),
        'positions': np.asarray(sim.positions, dtype=np.float64).copy(),
        'adjacency_matrix': np.asarray(sim.adjacency_matrix).copy(),
        'client_id_map': {},
    }

    # Safely extract Node & Threshold properties (fallback to zero arrays if simulator lacks attribute)
    for key in NODE_PROP_KEYS + THRESHOLD_KEYS:
        if hasattr(sim, key):
            block[key] = np.asarray(getattr(sim, key)).copy()
        else:
            block[key] = np.zeros(num_nodes, dtype=np.float64)

    # Safely extract Line/Edge properties (fallback to zero arrays if simulator lacks attribute)
    for key in LINE_KEYS:
        if hasattr(sim, key):
            block[key] = np.asarray(getattr(sim, key)).copy()
        else:
            block[key] = np.zeros(num_edges, dtype=np.float64)

    return block

def validate_topology_block(block: Dict) -> List[str]:
    problems: List[str] = []
    for key in ['version', 'num_nodes', 'edge_index', 'positions']:
        if key not in block:
            problems.append(f"missing key: {key}")
    if problems:
        return problems

    n = int(block['num_nodes'])
    ei = np.asarray(block['edge_index'])
    if ei.ndim != 2 or ei.shape[0] != 2:
        problems.append(f"edge_index must be [2, E], got {ei.shape}")
        return problems
    e = ei.shape[1]

    if np.asarray(block['positions']).shape[0] != n:
        problems.append("positions length != num_nodes")
    if e > 0 and (ei.max() >= n or ei.min() < 0):
        problems.append("edge_index references nodes outside [0, num_nodes)")

    for key in NODE_PROP_KEYS + THRESHOLD_KEYS:
        arr = block.get(key)
        if arr is None:
            problems.append(f"missing node array: {key}")
        elif np.asarray(arr).shape[0] != n:
            problems.append(f"{key} length != num_nodes")

    for key in LINE_KEYS:
        arr = block.get(key)
        if arr is None:
            problems.append(f"missing line array: {key}")
        elif np.asarray(arr).shape[0] != e:
            problems.append(f"{key} length != num_edges")

    return problems


def edge_index_to_adjacency(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def grid_is_connected(edge_index: np.ndarray, num_nodes: int,
                      removed_nodes: Optional[set] = None) -> bool:
    removed = removed_nodes or set()
    alive = [n for n in range(num_nodes) if n not in removed]
    if not alive:
        return False
    adj: Dict[int, List[int]] = {n: [] for n in alive}
    for s, d in zip(edge_index[0], edge_index[1]):
        s, d = int(s), int(d)
        if s in removed or d in removed:
            continue
        adj[s].append(d)
        adj[d].append(s)
    seen = {alive[0]}
    stack = [alive[0]]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(alive)


def _rng_for(client_id: str, base_seed: Optional[int]) -> np.random.Generator:
    mix = abs(hash((str(client_id), int(base_seed or 0)))) % (2**32)
    return np.random.default_rng(mix)


def _new_node_params(block: Dict, node_type: int, flow_rate: float,
                     rng: np.random.Generator) -> Dict[str, float]:
    types = np.asarray(block['node_types'])
    donors = np.nonzero(types == node_type)[0]
    if len(donors) == 0:
        donors = np.arange(len(types))
    donor = int(rng.choice(donors))

    params = {key: float(np.asarray(block[key])[donor])
              for key in NODE_PROP_KEYS + THRESHOLD_KEYS
              if key != 'node_types'}
    params['node_types'] = float(node_type)
    
    # Map frontend UI payloads to the simulator's internal math variables
    if node_type == 1:      # Supply / Pump Station
        params['pump_capacity'] = float(flow_rate)
        params['base_flow'] = 0.0
    elif node_type == 0:    # Delivery / Load
        params['pump_capacity'] = 0.0
        params['base_flow'] = float(flow_rate)
    return params


def _new_edge_params(block: Dict, s: int, d: int,
                     rng: np.random.Generator) -> Dict[str, float]:
    pos = np.asarray(block['positions'])
    ei = np.asarray(block['edge_index'])
    new_dist = float(np.linalg.norm(pos[s] - pos[d]))
    dists = np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)
    donor = int(np.argmin(np.abs(dists - new_dist)))
    return {key: float(np.asarray(block[key])[donor]) for key in LINE_KEYS}


def apply_topology_edits(block: Dict, edits: Dict) -> Tuple[Dict, Dict[str, int]]:
    # NATIVE PIPELINE TERMS
    TYPE_CODES = {'delivery': 0, 'pump station': 1, 'valve': 2}
    
    new_block = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in block.items()}
    seed = new_block.get('seed')
    n = int(new_block['num_nodes'])

    id_map: Dict[str, int] = dict(new_block.get('client_id_map', {}))
    for spec in edits.get('added_nodes', []):
        cid = str(spec['id'])
        if cid in id_map:
            raise ValueError(f"duplicate draft node id: {cid}")
            
        # Parse the native pipeline term
        node_type = TYPE_CODES.get(str(spec.get('type', 'delivery')).lower().strip())
        if node_type is None:
            raise ValueError(f"unknown node type: {spec.get('type')}")
            
        rng = _rng_for(cid, seed)
        
        # Parse flow_rate instead of p_mw
        params = _new_node_params(new_block, node_type,
                                  float(spec.get('flow_rate', 0.0)), rng)
        id_map[cid] = n
        n += 1
        
        # Safely pad 2D coordinates to 3D to match the elevation matrix
        new_pos = np.asarray(spec['position'], dtype=np.float64)
        if new_pos.shape[0] == 2 and new_block['positions'].shape[1] == 3:
            new_pos = np.array([new_pos[0], new_pos[1], -2.0], dtype=np.float64) # Default underground Z
            
        new_block['positions'] = np.vstack([
            new_block['positions'],
            new_pos])
        
        for key in NODE_PROP_KEYS + THRESHOLD_KEYS:
            new_block[key] = np.append(new_block[key], params[key])
    new_block['num_nodes'] = n
    new_block['client_id_map'] = id_map

    def resolve(ref) -> int:
        if isinstance(ref, str):
            if ref not in id_map:
                raise ValueError(f"unknown node reference: {ref}")
            return id_map[ref]
        idx = int(ref)
        if not 0 <= idx < n:
            raise ValueError(f"node index out of range: {idx}")
        return idx

    for spec in edits.get('moved_nodes', []):
        idx = resolve(spec['id'])
        new_pos = np.asarray(spec['position'], dtype=np.float64)
        
        # Keep the node's original Z-coordinate if the frontend only sends X and Y
        if new_pos.shape[0] == 2 and new_block['positions'].shape[1] == 3:
            new_pos = np.array([new_pos[0], new_pos[1], new_block['positions'][idx][2]], dtype=np.float64)
            
        new_block['positions'][idx] = new_pos

    ei = np.asarray(new_block['edge_index'])
    keep = np.ones(ei.shape[1], dtype=bool)
    for pair in edits.get('removed_edges', []):
        a, b = resolve(pair[0]), resolve(pair[1])
        hit = (((ei[0] == a) & (ei[1] == b)) | ((ei[0] == b) & (ei[1] == a)))
        if not hit.any():
            raise ValueError(f"removed edge ({a},{b}) does not exist")
        keep &= ~hit
    ei = ei[:, keep]
    for key in LINE_KEYS:
        new_block[key] = np.asarray(new_block[key])[keep]

    existing = {(int(s), int(d)) for s, d in zip(ei[0], ei[1])}
    for spec in edits.get('added_edges', []):
        s, d = resolve(spec['from']), resolve(spec['to'])
        if s == d:
            raise ValueError("self-loop edges are not allowed")
        if (s, d) in existing or (d, s) in existing:
            raise ValueError(f"edge ({s},{d}) already exists")
        rng = _rng_for(f"edge:{spec['from']}->{spec['to']}", seed)
        params = _new_edge_params(new_block, s, d, rng)
        ei = np.hstack([ei, np.array([[s, d], [d, s]], dtype=ei.dtype).T])
        existing.update({(s, d), (d, s)})
        for key in LINE_KEYS:
            new_block[key] = np.append(
                new_block[key], [params[key], params[key]])
    new_block['edge_index'] = ei

    removed = {resolve(r) for r in edits.get('removed_nodes', [])}
    if removed:
        ei = np.asarray(new_block['edge_index'])
        keep = ~(np.isin(ei[0], list(removed)) | np.isin(ei[1], list(removed)))
        new_block['edge_index'] = ei[:, keep]
        for key in LINE_KEYS:
            new_block[key] = np.asarray(new_block[key])[keep]
        for idx in removed:
            new_block['pump_capacity'][idx] = 0.0
            new_block['base_flow'][idx] = 0.0
        new_block['removed_nodes'] = sorted(
            set(new_block.get('removed_nodes', [])) | removed)

    # if not grid_is_connected(np.asarray(new_block['edge_index']), n,
    #                          set(new_block.get('removed_nodes', []))):
    #     raise ValueError("edit would disconnect the pipeline")
    new_block['adjacency_matrix'] = edge_index_to_adjacency(
        np.asarray(new_block['edge_index']), n)
    problems = validate_topology_block(new_block)
    if problems:
        raise ValueError(f"invalid topology after edit: {problems}")

    new_block['source'] = 'edited'
    return new_block, id_map