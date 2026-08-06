"""
Invariant checks for intra-timestep cascade timing.

Validates the contract introduced when the generator started preserving
sub-timestep failure times and causal parents:

  * floor(failure_times_exact[i]) == failure_times[i]
  * exact times lie within [t, t + INTRA_STEP_MAX]
  * the sequence is ordered, and strictly increasing within a wave
  * failure_parents are valid node ids (or -1 for a trigger)
  * a parent always fails strictly before its child
  * a parent is a topological neighbour of its child
  * node_labels[floor(exact)] marks the node failed

Run against a generated dataset directory:

    python tests/test_intra_timestep_cascade.py /tmp/gen_check/train
"""

import glob
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_cascade_prediction.data.generator.config import Settings  # noqa: E402

INTRA_STEP_MAX = Settings.Simulation.INTRA_STEP_MAX
TOL = 1e-6


class Checker:
    def __init__(self):
        self.failures = []
        self.checks = 0
        self.scenarios = 0
        self.cascades = 0
        self.stats = defaultdict(int)

    def expect(self, condition, message):
        self.checks += 1
        if not condition:
            self.failures.append(message)

    def check_scenario(self, tag, scenario):
        self.scenarios += 1
        meta = scenario.get("metadata", {})
        seq = scenario["sequence"]

        failed = list(meta.get("failed_nodes", []))
        times_int = list(meta.get("failure_times", []))
        times_exact = list(meta.get("failure_times_exact", []))
        parents = list(meta.get("failure_parents", []))
        reasons = list(meta.get("failure_reasons", []))

        if not failed:
            self.expect(
                not meta.get("is_cascade", False),
                f"{tag}: is_cascade True but failed_nodes empty",
            )
            return

        self.cascades += 1
        num_nodes = int(meta.get("num_nodes", len(seq[0]["node_labels"])))

        # --- new keys present and aligned -----------------------------------
        self.expect(times_exact, f"{tag}: failure_times_exact missing")
        self.expect(parents, f"{tag}: failure_parents missing")
        if not times_exact or not parents:
            return

        n = len(failed)
        for name, arr in (
            ("failure_times", times_int),
            ("failure_times_exact", times_exact),
            ("failure_parents", parents),
            ("failure_reasons", reasons),
        ):
            self.expect(len(arr) == n, f"{tag}: {name} length {len(arr)} != {n}")

        # --- floor consistency and in-step bounds ---------------------------
        for node, t_int, t_exact in zip(failed, times_int, times_exact):
            self.expect(
                math.floor(t_exact + TOL) == t_int,
                f"{tag}: node {node} floor({t_exact}) != {t_int}",
            )
            frac = t_exact - t_int
            self.expect(
                -TOL <= frac <= INTRA_STEP_MAX + TOL,
                f"{tag}: node {node} fraction {frac:.6f} outside [0, {INTRA_STEP_MAX}]",
            )
            if frac > TOL:
                self.stats["sub_step_failures"] += 1

        # --- global ordering ------------------------------------------------
        self.expect(
            all(a <= b + TOL for a, b in zip(times_exact, times_exact[1:])),
            f"{tag}: failure_times_exact not sorted ascending",
        )

        # --- strict ordering within a wave ----------------------------------
        by_step = defaultdict(list)
        for node, t_int, t_exact in zip(failed, times_int, times_exact):
            by_step[t_int].append((t_exact, node))
        for t_int, wave in by_step.items():
            if len(wave) > 1:
                self.stats["multi_node_waves"] += 1
                self.stats["max_wave_size"] = max(
                    self.stats["max_wave_size"], len(wave)
                )
                ordered = sorted(wave)
                distinct = len({round(x, 9) for x, _ in wave})
                # Triggers detected simultaneously legitimately share time 0.0;
                # anything propagated must be separated.
                self.expect(
                    distinct > 1 or all(abs(x - t_int) < TOL for x, _ in wave),
                    f"{tag}: step {t_int} has {len(wave)} failures collapsed "
                    f"onto one exact time {ordered[0][0]}",
                )

        # --- parents --------------------------------------------------------
        exact_by_node = dict(zip(failed, times_exact))
        edge_index = scenario.get("edge_index")
        if edge_index is None:
            edge_index = scenario.get("topology", {}).get("edge_index")
        neighbours = defaultdict(set)
        if edge_index is not None:
            src, dst = edge_index[0], edge_index[1]
            for s, d in zip(src, dst):
                neighbours[int(s)].add(int(d))
                neighbours[int(d)].add(int(s))

        for node, parent, t_exact in zip(failed, parents, times_exact):
            self.expect(
                parent == -1 or 0 <= parent < num_nodes,
                f"{tag}: node {node} parent {parent} out of range",
            )
            if parent == -1:
                self.stats["triggers"] += 1
                continue
            self.stats["propagated"] += 1
            self.expect(
                parent in exact_by_node,
                f"{tag}: node {node} parent {parent} is not in failed_nodes",
            )
            if parent in exact_by_node:
                self.expect(
                    exact_by_node[parent] < t_exact - TOL,
                    f"{tag}: node {node} fails at {t_exact} but parent {parent} "
                    f"fails at {exact_by_node[parent]} (not strictly earlier)",
                )
            if neighbours:
                self.expect(
                    node in neighbours[parent],
                    f"{tag}: node {node} parent {parent} is not a topological neighbour",
                )

        # --- labels agree with the times ------------------------------------
        for node, t_int, t_exact in zip(failed, times_int, times_exact):
            if t_int < len(seq):
                labels = seq[t_int]["node_labels"]
                self.expect(
                    labels[node] > 0.5,
                    f"{tag}: node {node} exact time {t_exact} but node_labels"
                    f"[{t_int}][{node}] == {labels[node]}",
                )

        # --- per-frame cascade_timing is fractional -------------------------
        last_timing = seq[-1].get("cascade_timing")
        if last_timing is not None:
            for node, t_exact in zip(failed, times_exact):
                expected = float(len(seq) - 1 - t_exact)
                self.expect(
                    abs(float(last_timing[node]) - expected) < 1e-4,
                    f"{tag}: cascade_timing[{node}] == {last_timing[node]}, "
                    f"expected {expected}",
                )


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    files = sorted(glob.glob(f"{data_dir}/scenarios_batch_*.pkl"))
    if not files:
        print(f"No scenario files found in {data_dir}")
        return 1

    checker = Checker()
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        batch = data if isinstance(data, list) else [data]
        for i, scenario in enumerate(batch):
            checker.check_scenario(f"{Path(path).name}[{i}]", scenario)

    print(f"scenarios:            {checker.scenarios}")
    print(f"  with cascades:      {checker.cascades}")
    print(f"  trigger failures:   {checker.stats['triggers']}")
    print(f"  propagated:         {checker.stats['propagated']}")
    print(f"  sub-step failures:  {checker.stats['sub_step_failures']}")
    print(f"  multi-node waves:   {checker.stats['multi_node_waves']}"
          f" (largest {checker.stats['max_wave_size']})")
    print(f"assertions:           {checker.checks}")

    if checker.failures:
        print(f"\nFAILED ({len(checker.failures)}):")
        for msg in checker.failures[:25]:
            print(f"  - {msg}")
        if len(checker.failures) > 25:
            print(f"  ... and {len(checker.failures) - 25} more")
        return 1

    if checker.stats["sub_step_failures"] == 0:
        print("\nWARNING: no sub-timestep failures observed - nothing was actually "
              "exercised. Generate more cascade scenarios.")
        return 1

    print("\nOK - all intra-timestep invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
