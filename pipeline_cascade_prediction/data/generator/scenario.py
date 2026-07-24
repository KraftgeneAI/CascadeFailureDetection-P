"""
Scenario Orchestrator Module (Pipeline Digital Twin)
====================================================
High-level orchestration for batch pipeline scenario generation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import pickle
import gc
import sys

from .simulator import PhysicsBasedPipelineSimulator
from .utils import MemoryMonitor
from .config import Settings


class ScenarioOrchestrator:
    """Orchestrates batch generation of pipeline cascade scenarios."""
    
    def __init__(
        self,
        simulator: PhysicsBasedPipelineSimulator,
        output_dir: str = 'data',
        batch_size: int = Settings.Scenario.DEFAULT_BATCH_SIZE,
        train_ratio: float = Settings.Dataset.TRAIN_RATIO,
        val_ratio: float = Settings.Dataset.VAL_RATIO,
        test_ratio: float = Settings.Dataset.TEST_RATIO
    ):
        self.simulator = simulator
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < Settings.Dataset.RATIO_TOLERANCE
        
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_dataset(
        self,
        num_normal: int = Settings.Scenario.DEFAULT_NUM_NORMAL,
        num_cascade: int = Settings.Scenario.DEFAULT_NUM_CASCADE,
        num_stressed: int = Settings.Scenario.DEFAULT_NUM_STRESSED,
        sequence_length: int = Settings.Scenario.DEFAULT_SEQUENCE_LENGTH,
        start_batch: int = 0
    ) -> Dict[str, int]:
        total_scenarios = num_normal + num_cascade + num_stressed
        if total_scenarios == 0:
            return {}
        
        splits = self._calculate_splits(num_normal, num_cascade, num_stressed)
        
        train_stats = self._generate_split(splits['train'], self.train_dir, 'TRAIN', sequence_length, start_batch)
        val_stats = self._generate_split(splits['val'], self.val_dir, 'VALIDATION', sequence_length, start_batch)
        test_stats = self._generate_split(splits['test'], self.test_dir, 'TEST', sequence_length, start_batch)
        
        return {'train': train_stats, 'val': val_stats, 'test': test_stats}
    
    def _calculate_splits(self, num_normal: int, num_cascade: int, num_stressed: int) -> Dict[str, Dict[str, int]]:
        return {
            'train': {
                'normal': int(num_normal * self.train_ratio),
                'cascade': int(num_cascade * self.train_ratio),
                'stressed': int(num_stressed * self.train_ratio),
                'total': int(num_normal * self.train_ratio) + int(num_cascade * self.train_ratio) + int(num_stressed * self.train_ratio)
            },
            'val': {
                'normal': int(num_normal * self.val_ratio),
                'cascade': int(num_cascade * self.val_ratio),
                'stressed': int(num_stressed * self.val_ratio),
                'total': int(num_normal * self.val_ratio) + int(num_cascade * self.val_ratio) + int(num_stressed * self.val_ratio)
            },
            'test': {
                'normal': num_normal - int(num_normal * self.train_ratio) - int(num_normal * self.val_ratio),
                'cascade': num_cascade - int(num_cascade * self.train_ratio) - int(num_cascade * self.val_ratio),
                'stressed': num_stressed - int(num_stressed * self.train_ratio) - int(num_stressed * self.val_ratio),
                'total': (num_normal - int(num_normal * self.train_ratio) - int(num_normal * self.val_ratio)) +
                         (num_cascade - int(num_cascade * self.train_ratio) - int(num_cascade * self.val_ratio)) +
                         (num_stressed - int(num_stressed * self.train_ratio) - int(num_stressed * self.val_ratio))
            }
        }
    
    def _generate_split(self, split_counts: Dict[str, int], output_dir: Path, split_name: str, sequence_length: int, start_batch: int) -> Dict[str, int]:
        total = split_counts['total']
        if total == 0:
            return {'generated': 0, 'failed': 0}
        
        print(f"\nGENERATING {split_name} SET ({total} scenarios)")
        
        types_to_gen = (['normal'] * split_counts['normal'] + ['cascade'] * split_counts['cascade'] + ['stressed'] * split_counts['stressed'])
        np.random.shuffle(types_to_gen)
        
        current_batch = []
        batch_count = start_batch
        generated_count = 0
        failed_count = 0
        
        for i in range(total):
            gen_type = types_to_gen[i]
            scenario = self._generate_with_retry(gen_type, sequence_length)
            
            if scenario is not None:
                current_batch.append(scenario)
                generated_count += 1
            else:
                failed_count += 1
            
            if len(current_batch) >= self.batch_size or i == total - 1:
                if len(current_batch) > 0:
                    batch_file = output_dir / f'scenarios_batch_{batch_count}.pkl'
                    with open(batch_file, 'wb') as f:
                        pickle.dump(current_batch, f)
                    print(f"  [SAVED] Batch {batch_count}: {len(current_batch)} scenarios -> {batch_file}")
                    
                    # Log memory usage and warn if it gets too high
                    print(f"  Memory: {MemoryMonitor.get_memory_usage():.1f} MB")
                    MemoryMonitor.check_threshold(8000)
                    
                    batch_count += 1
                    current_batch = []
                    gc.collect()
        
        return {'generated': generated_count, 'failed': failed_count}
    
    def _generate_with_retry(self, scenario_type: str, sequence_length: int, max_retries: int = Settings.Scenario.MAX_RETRIES) -> Optional[Dict]:
        if scenario_type == 'cascade':
            stress_level = np.random.uniform(Settings.Scenario.CASCADE_STRESS_MIN, Settings.Scenario.CASCADE_STRESS_MAX)
        elif scenario_type == 'stressed':
            stress_level = np.random.uniform(Settings.Scenario.STRESSED_STRESS_MIN, Settings.Scenario.STRESSED_STRESS_MAX)
        else:
            stress_level = np.random.uniform(Settings.Scenario.NORMAL_STRESS_MIN, Settings.Scenario.NORMAL_STRESS_MAX)

        for retry in range(max_retries):
            scenario = self.simulator.generate_scenario(stress_level=stress_level, sequence_length=sequence_length)
            if scenario is None:
                continue
            
            is_cascade = scenario['metadata']['is_cascade']
            if scenario_type == 'cascade' and not is_cascade:
                stress_level += 0.05
                continue
            if scenario_type in ['normal', 'stressed'] and is_cascade:
                stress_level = max(0.0, stress_level - 0.05)  # Prevent negative stress
                continue
            
            return scenario
        return None


def main():
    """CLI Entry point for generating the dataset."""
    print("Initializing Physics-Based Pipeline Simulator...")
    simulator = PhysicsBasedPipelineSimulator(
        num_nodes=Settings.Scenario.DEFAULT_NUM_NODES,
        seed=Settings.Scenario.DEFAULT_SEED,
        topology_file=Settings.Dataset.DEFAULT_TOPOLOGY_FILE
    )
    
    orchestrator = ScenarioOrchestrator(
        simulator=simulator,
        output_dir='data',
        batch_size=Settings.Scenario.DEFAULT_BATCH_SIZE
    )
    
    print("Starting dataset generation...")
    orchestrator.generate_dataset(
        num_normal=Settings.Scenario.DEFAULT_NUM_NORMAL,
        num_cascade=Settings.Scenario.DEFAULT_NUM_CASCADE,
        num_stressed=Settings.Scenario.DEFAULT_NUM_STRESSED,
        sequence_length=Settings.Scenario.DEFAULT_SEQUENCE_LENGTH
    )
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()