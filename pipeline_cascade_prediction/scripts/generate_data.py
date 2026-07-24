import argparse
import sys
import os

# Dynamically find the root folder (CASCADEFAILUREDETECTION-P) and add it to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Updated imports matching your new package name
from pipeline_cascade_prediction.data.generator.simulator import PhysicsBasedPipelineSimulator
from pipeline_cascade_prediction.data.generator.scenario import ScenarioOrchestrator
from pipeline_cascade_prediction.data.generator.config import Settings

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pipeline digital twin data.")
    
    parser.add_argument("--normal", type=int, default=Settings.Scenario.DEFAULT_NUM_NORMAL)
    parser.add_argument("--cascade", type=int, default=Settings.Scenario.DEFAULT_NUM_CASCADE)
    parser.add_argument("--stressed", type=int, default=Settings.Scenario.DEFAULT_NUM_STRESSED)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--topology-file", type=str, default=Settings.Dataset.DEFAULT_TOPOLOGY_FILE)
    parser.add_argument("--sequence-length", type=int, default=Settings.Scenario.DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=Settings.Scenario.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=Settings.Scenario.DEFAULT_SEED)
    
    args = parser.parse_args()

    print("==================================================")
    print(" PIPELINE DIGITAL TWIN: SYNTHETIC DATA GENERATOR")
    print("==================================================")

    simulator = PhysicsBasedPipelineSimulator(
        num_nodes=Settings.Topology.DEFAULT_NUM_NODES,
        seed=args.seed,
        topology_file=args.topology_file
    )
    
    orchestrator = ScenarioOrchestrator(
        simulator=simulator,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    print("\nStarting dataset generation...")
    orchestrator.generate_dataset(
        num_normal=args.normal,
        num_cascade=args.cascade,
        num_stressed=args.stressed,
        sequence_length=args.sequence_length
    )

if __name__ == "__main__":
    main()