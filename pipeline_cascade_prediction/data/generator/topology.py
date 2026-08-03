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

#: 1 cubic metre = 6.2898 US oil barrels (used to convert the Seaway
#: throughput figure, which the reference paper quotes in m3/h).
BBL_PER_M3 = 6.2898


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
            'num_edges': edge_index.shape[1],
            # Structural metadata so node roles and geometry can be derived
            # from the graph instead of from node index.
            'trunk_nodes': list(getattr(self, 'trunk_nodes', [])),
            'twin_nodes': list(getattr(self, 'twin_nodes', [])),
            'pump_nodes': list(getattr(self, 'pump_nodes', [])),
            'crossovers': list(getattr(self, 'crossovers', [])),
            'branch_of': dict(getattr(self, 'branch_of', {})),
            'route_frac': dict(getattr(self, 'route_frac', {})),
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
        Build the Seaway-derived graph: one long-haul trunk, parallel
        redundancy loops around pump stations, and lateral delivery spurs at
        the real terminal groups.

        Structure is decided here and *recorded* on the instance so that
        geometry and node roles can be derived from it rather than from node
        index. Storing `self.trunk_nodes`, `self.pump_nodes` and
        `self.branch_of` is what lets `_generate_geographic_positions` place a
        spur next to the trunk node it actually taps - previously positions
        were laid out in index order while branches attached to random trunk
        nodes, producing "pipes" up to 602 km long and a 3.2x resistance skew.
        """
        cfg = Settings.Topology
        n = self.num_nodes
        adj = np.zeros((n, n))

        corridor = max(4, min(int(round(n * cfg.TRUNK_FRACTION)), n))
        # Split the corridor budget between the original line and the twin,
        # which covers TWIN_FRACTION of the route at the same node spacing.
        t_main = int(round(corridor / (1.0 + cfg.TWIN_FRACTION)))
        t_main = max(3, min(t_main, corridor - 2))
        t_twin = corridor - t_main

        self.trunk_nodes = list(range(t_main))                       # original line
        self.twin_nodes = list(range(t_main, t_main + t_twin))       # Seaway Twin
        self.route_frac = {}          # node -> position along the route in [0,1]

        # -- 1. original line: Cushing (node 0) -> Freeport ------------------
        for i in range(t_main - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1
        for k, node in enumerate(self.trunk_nodes):
            self.route_frac[node] = k / max(1, t_main - 1)

        # -- 2. Seaway Twin: parallel line over the first TWIN_FRACTION -----
        for i in range(t_twin - 1):
            a, b = self.twin_nodes[i], self.twin_nodes[i + 1]
            adj[a, b] = adj[b, a] = 1
        for k, node in enumerate(self.twin_nodes):
            self.route_frac[node] = cfg.TWIN_FRACTION * (k / max(1, t_twin - 1))

        # -- 3. pump stations, evenly spaced along the route (~108 km) ------
        k = min(cfg.NUM_PUMP_STATIONS, t_main)
        self.pump_nodes = sorted({int(round(f * (t_main - 1)))
                                  for f in np.linspace(0.0, 1.0, k, endpoint=True)})

        # -- 4. crossovers tying the two lines together at each station -----
        # Real twinned corridors are cross-connected at stations so either line
        # can carry flow past a failure on the other.
        self.crossovers = []

        def _tie(main_node: int) -> None:
            f = self.route_frac[main_node]
            if f > cfg.TWIN_FRACTION:
                return                       # past the end of the twin line
            j = int(round((f / cfg.TWIN_FRACTION) * (t_twin - 1)))
            w = self.twin_nodes[int(np.clip(j, 0, t_twin - 1))]
            if not adj[main_node, w]:
                adj[main_node, w] = adj[w, main_node] = 1
                self.crossovers.append((main_node, w))

        for p in self.pump_nodes:            # every station is cross-connected
            _tie(p)

        n_extra = int(cfg.TWIN_ROUTE_KM // cfg.CROSSOVER_SPACING_KM)
        for f in np.linspace(0.0, cfg.TWIN_FRACTION, max(2, n_extra)):
            _tie(int(np.clip(round(f * (t_main - 1)), 0, t_main - 1)))

        # -- 5. delivery spurs at the real terminal groups -------------------
        self.branch_of = {}          # branch node -> node it taps
        spur_nodes = list(range(corridor, n))
        if spur_nodes:
            groups = cfg.DELIVERY_GROUPS
            weights = np.array([g[1] for g in groups], dtype=float)
            weights /= weights.sum()
            counts = np.floor(weights * len(spur_nodes)).astype(int)
            while counts.sum() < len(spur_nodes):           # distribute remainder
                counts[int(np.argmax(weights - counts / max(1, len(spur_nodes))))] += 1

            cursor = 0
            for (frac, _), cnt in zip(groups, counts):
                tap = int(round(frac * (t_main - 1)))
                tap = int(np.clip(tap, 1, t_main - 1))
                for j in range(cnt):
                    if cursor >= len(spur_nodes):
                        break
                    b = spur_nodes[cursor]; cursor += 1
                    # First spur of a group hangs off the corridor; later ones
                    # extend it, so a group is a short lateral chain rather
                    # than a star of parallel stubs.
                    parent = tap if j == 0 else spur_nodes[cursor - 2]
                    adj[b, parent] = adj[parent, b] = 1
                    self.branch_of[b] = parent
                    self.route_frac[b] = self.route_frac.get(parent, frac)

        return adj
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> torch.Tensor:
        edges = np.where(adj > 0)
        return torch.tensor(np.vstack(edges), dtype=torch.long)
    
    def _generate_geographic_positions(self) -> np.ndarray:
        """
        Place nodes in 3D (x, y km; z elevation m) *following the graph*.

        Trunk nodes are spaced evenly along a 968 km Cushing -> Freeport route
        so each trunk segment is ROUTE_LENGTH_KM / (trunk_size - 1) ~= 11 km,
        matching real block-valve spacing. Each delivery spur is placed as a
        short lateral offset from the trunk node it actually taps, so its pipe
        length is the spur length rather than an artefact of node numbering.

        Elevation follows the real profile direction: Cushing sits on the
        Oklahoma plateau (~320 m) and the line descends to the Gulf coast
        (~0 m), with local terrain noise. That gradient matters because the
        Bernoulli/Leibenzon head loss the reference model uses includes an
        elevation term; the previous +/-10 m range made gravity negligible.

        Requires `_generate_linear_topology` to have run first (it records
        trunk_nodes / branch_of).
        """
        cfg = Settings.Topology
        n = self.num_nodes
        positions = np.zeros((n, 3))

        trunk = getattr(self, 'trunk_nodes', list(range(n)))
        twin = getattr(self, 'twin_nodes', [])
        branch_of = getattr(self, 'branch_of', {})
        t_count = len(trunk)

        # -- 1. trace the route once, as a gently meandering polyline -------
        # Both lines share this corridor; the twin is laid a short distance
        # off the original, as a real loop line is within the same easement.
        route_km = cfg.ROUTE_LENGTH_KM
        seg = route_km / max(1, t_count - 1)
        heading = np.deg2rad(-72.0)          # roughly N -> SSE, Cushing to the Gulf
        xy = np.array([0.0, 0.0])
        centre, headings = [], []
        for k in range(t_count):
            if k > 0:
                heading += np.random.normal(0.0, 0.045)   # slow, smooth curvature
                xy = xy + seg * np.array([np.cos(heading), np.sin(heading)])
            centre.append(xy.copy())
            headings.append(heading)
        centre = np.asarray(centre)

        def elevation(frac: float) -> float:
            """Monotone descent from the Cushing plateau to the Gulf coast."""
            base = cfg.ELEVATION_MAX_M + frac * (cfg.ELEVATION_MIN_M - cfg.ELEVATION_MAX_M)
            return base + np.random.normal(0.0, 8.0)

        def point_at(frac: float):
            """Interpolate the route polyline at a fractional position."""
            x = float(np.clip(frac, 0.0, 1.0)) * (t_count - 1)
            i = int(np.floor(x)); j = min(i + 1, t_count - 1); w = x - i
            return centre[i] * (1 - w) + centre[j] * w, headings[min(i, t_count - 1)]

        for k, node in enumerate(trunk):
            positions[node] = [centre[k][0], centre[k][1],
                               elevation(k / max(1, t_count - 1))]

        # -- 2. twin line: offset ~0.3 km perpendicular to the corridor -----
        route_frac = getattr(self, 'route_frac', {})
        for node in twin:
            f = route_frac.get(node, 0.0)
            pt, hd = point_at(f)
            nx, ny = -np.sin(hd), np.cos(hd)        # unit normal to the route
            positions[node] = [pt[0] + 0.3 * nx, pt[1] + 0.3 * ny, elevation(f)]

        # -- 3. spurs: short lateral offset from the node they tap ----------
        for b, parent in branch_of.items():
            p = positions[parent]
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            L = np.random.uniform(cfg.BRANCH_LENGTH_KM_MIN, cfg.BRANCH_LENGTH_KM_MAX)
            positions[b] = [p[0] + L * np.cos(ang),
                            p[1] + L * np.sin(ang),
                            p[2] + np.random.normal(0.0, 12.0)]

        # any node not covered above (defensive) sits on the route start
        placed = set(trunk) | set(twin) | set(branch_of)
        for i in range(n):
            if i not in placed:
                positions[i] = positions[trunk[0]]

        return positions

class PipelinePropertyInitializer:
    """
    Initializes physical pipeline properties along the single line.

    When a `topology` dict is supplied (as produced by
    PipelineTopologyGenerator.generate_topology) node roles are derived from
    the graph: pump stations sit at the trunk positions where the real Seaway
    system has them, and delivery terminals sit at spur ends. Without it the
    initializer falls back to the previous index-based random assignment,
    which placed pump stations on branch leaves and delivery terminals
    mid-trunk - backwards for a long-haul line.
    """

    def __init__(self, num_nodes: int, seed: int = Settings.Scenario.DEFAULT_SEED,
                 topology: Optional[Dict] = None):
        self.num_nodes = num_nodes
        self.topology = topology or {}
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
        Assign node types: pump stations (1), delivery terminals (0),
        and pipe junctions / block valves (2).

        With a topology present, roles follow the graph: pump stations at the
        recorded trunk stations, terminals at spur ends, valves everywhere
        else. This is how a long-haul liquids line is actually laid out.
        """
        node_types = np.full(self.num_nodes, 2, dtype=int)

        pump_nodes = list(self.topology.get('pump_nodes', []))
        branch_of = self.topology.get('branch_of', {})

        if pump_nodes:
            # ---- topology-driven (Seaway-derived) assignment --------------
            for idx in pump_nodes:
                if 0 <= idx < self.num_nodes:
                    node_types[idx] = 1
            node_types[0] = 1                      # Cushing origin injection

            # Delivery terminals sit at the *ends* of lateral spurs: a branch
            # node that nothing else taps off is a terminal.
            parents = set(branch_of.values())
            for b in branch_of:
                if b not in parents and node_types[b] != 1:
                    node_types[b] = 0
            if not (node_types == 0).any():        # degenerate: no spurs
                node_types[self.num_nodes - 1] = 0

            all_pump_indices = np.array(sorted({0, *pump_nodes}))
            return node_types, all_pump_indices

        # ---- legacy fallback: index-based random assignment ---------------
        node_types[0] = 1
        booster_indices = np.random.choice(range(1, self.num_nodes), size=int(self.num_nodes * 0.1), replace=False)
        for idx in booster_indices:
            node_types[idx] = 1

        terminal_indices = np.random.choice(range(int(self.num_nodes * 0.5), self.num_nodes), size=int(self.num_nodes * 0.15), replace=False)
        for idx in terminal_indices:
            if node_types[idx] != 1:  # Don't overwrite pumps
                node_types[idx] = 0

        all_pump_indices = np.array([0] + list(booster_indices))
        return node_types, all_pump_indices

    def _calculate_base_flow(self, node_types: np.ndarray) -> np.ndarray:
        """Distribute delivery demand across all terminal nodes.

        Total offtake is scaled to the real Seaway throughput
        (THROUGHPUT_M3_H = 6300 m3/h ~= 39,600 bbl/hr, i.e. 950,000 bbl/day)
        rather than an arbitrary per-terminal range, then split unevenly
        across terminals so no two deliver the same volume.
        """
        base_flow = np.zeros(self.num_nodes)
        terminals = np.where(node_types == 0)[0]
        if len(terminals) == 0:
            return base_flow

        total_bph = Settings.Topology.THROUGHPUT_M3_H * BBL_PER_M3
        shares = np.random.uniform(0.5, 1.5, len(terminals))
        shares /= shares.sum()
        base_flow[terminals] = shares * total_bph
        return base_flow

    def _size_pumps(self, base_flow: np.ndarray, pump_indices: np.ndarray) -> np.ndarray:
        """Distribute pumping capacity across the main injection and boosters.

        Sized against TOTAL system demand. This previously read
        `base_flow[-1]` - the demand of the last node by index - which was a
        valve in the old topology, so target capacity came out as 0.0, every
        pump got zero capacity, and the simulator injected no fluid at all.
        """
        total_demand = float(np.sum(base_flow))
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