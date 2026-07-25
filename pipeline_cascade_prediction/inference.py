"""
Inference Module (Pipeline Digital Twin)
========================================
Provides the CascadePredictor class for the FastAPI backend.
Handles data loading, truncation, and GNN inference.
"""

import os
import pickle
import torch
import numpy as np
import glob
from typing import Dict, Any

from pipeline_cascade_prediction.model.unified_model import UnifiedPipelinePredictionModel
from pipeline_cascade_prediction.data.collation import collate_cascade_batch
from pipeline_cascade_prediction.data.dataset import calculate_truncation_window
from pipeline_cascade_prediction.data.generator.config import Settings

class CascadePredictor:
    def __init__(self, model_path: str, topology_path: str, device: torch.device):
        self.device = device
        
        # 1. Initialize the Pipeline Model
        self.model = UnifiedPipelinePredictionModel(
            embedding_dim=128,
            hidden_dim=128
        ).to(self.device)
        
        # 2. Load the trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded trained Pipeline GNN model from {model_path}")
        else:
            print(f"⚠️ Warning: Model not found at {model_path}. Using random initialization.")
            
        self.model.eval()
        self.node_threshold = Settings.Training.NODE_THRESHOLD
        self.cascade_threshold = Settings.Training.CASCADE_THRESHOLD

    def _load_and_preprocess(self, data_path: str, scenario_idx: int, end_step: int = None) -> Dict[str, torch.Tensor]:
        """Manually loads a scenario and applies physical normalizations for the GNN."""
        
        # Locate the scenario file
        files = sorted(glob.glob(os.path.join(data_path, "scenarios_batch_*.pkl")))
        if not files:
            files = sorted(glob.glob(os.path.join(data_path, "scenario_*.pkl")))
            
        file_path = files[scenario_idx]
        
        with open(file_path, 'rb') as f:
            batch_data = pickle.load(f)
        
        # Extract sequence and metadata
        scenario = batch_data[0] if isinstance(batch_data, list) else batch_data
        seq_original = scenario['sequence']
        metadata = scenario['metadata']
        
        # Slicing Logic: Streaming (strict window) vs Compare (reproducible seed)
        if end_step is not None:
            start_idx, end_idx = 0, end_step
        else:
            start_idx, end_idx = calculate_truncation_window(
                len(seq_original), 
                metadata.get('cascade_start_time', -1), 
                metadata.get('is_cascade', False)
            )
        
        seq = seq_original[start_idx:end_idx] if len(seq_original[start_idx:end_idx]) > 0 else seq_original[:10]
        T = len(seq)
        num_nodes, num_edges = metadata['num_nodes'], metadata['num_edges']
        
        # Initialize Tensors
        node_features = torch.zeros((T, num_nodes, 124), dtype=torch.float32)
        edge_attr_padded = torch.zeros((T, num_edges, 3), dtype=torch.float32)
        edge_mask = torch.ones((T, num_edges), dtype=torch.float32)
        
        # Apply Pipeline Physics Normalization (Pressure & Flow)
        for t, step in enumerate(seq):
            scada = step.get('scada_data', np.zeros((num_nodes, 18)))
            scada[:, 0] = scada[:, 0] / Settings.PipelineSystem.REFERENCE_PRESSURE_PSI
            scada[:, 1] = scada[:, 1] / Settings.PipelineSystem.REFERENCE_PRESSURE_PSI
            scada[:, 2:5] = scada[:, 2:5] / Settings.PipelineSystem.BASE_FLOW_BBL_HR
            node_features[t, :, :min(18, scada.shape[1])] = torch.from_numpy(scada[:, :18]).float()
            
            ea = step.get('edge_attr', np.zeros((num_edges, 3)))
            ea[:, 1:3] = ea[:, 1:3] / Settings.PipelineSystem.BASE_FLOW_BBL_HR
            edge_attr_padded[t, :, :min(3, ea.shape[1])] = torch.from_numpy(ea[:, :3]).float()
            
            loading = np.abs(ea[:, 2]) / (np.abs(ea[:, 1]) + 1e-6)
            edge_mask[t] = torch.from_numpy(np.clip(1.0 - loading, 0.0, 1.0)).float()

        # Format as batch item
        item = {
            'node_features': node_features,
            'edge_index': torch.tensor(scenario['edge_index'], dtype=torch.long),
            'edge_attr': edge_attr_padded,
            'edge_mask': edge_mask,
            'sequence_length': T,
        }
        
        batch = collate_cascade_batch([item])
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def predict_scenario(self, data_path: str, scenario_idx: int) -> Dict[str, Any]:
        """Runs inference for a full scenario (Used by CompareService & CascadeService)."""
        batch = self._load_and_preprocess(data_path, scenario_idx)
        
        with torch.no_grad():
            outputs = self.model(batch)
            
        node_probs = outputs['failure_probability'].squeeze(-1).sigmoid().cpu().numpy()[0]
        timing = outputs['cascade_timing'].squeeze(-1).cpu().numpy()[0]
        
        cascade_prob = float(node_probs.max())
        cascade_detected = cascade_prob > self.cascade_threshold
        
        path = []
        if cascade_detected:
            failed_nodes = np.where(node_probs > self.node_threshold)[0]
            for node_id in failed_nodes:
                path.append({
                    "node_id": int(node_id),
                    "ranking_score": float(node_probs[node_id]),
                    "pred_time_minutes": float(timing[node_id] * Settings.Thermal.DT_MINUTES)
                })
            path = sorted(path, key=lambda x: x["pred_time_minutes"])
            for i, p in enumerate(path):
                p["order"] = i + 1
                
        return {
            "cascade_detected": cascade_detected,
            "cascade_probability": cascade_prob,
            "cascade_path": path,
            "cascade_sequence": path,
            "top_nodes": path,
        }

    def predict_window(self, data_path: str, scenario_idx: int, end_step: int) -> Dict[str, Any]:
        """Runs windowed inference for Live Streaming mode."""
        batch = self._load_and_preprocess(data_path, scenario_idx, end_step=end_step)
        
        with torch.no_grad():
            outputs = self.model(batch)
            
        node_probs = outputs['failure_probability'].squeeze(-1).sigmoid().cpu().numpy()[0]
        timing = outputs['cascade_timing'].squeeze(-1).cpu().numpy()[0]
        
        cascade_prob = float(node_probs.max())
        cascade_detected = cascade_prob > self.cascade_threshold
        
        risky_nodes = []
        if cascade_detected:
            failed_nodes = np.where(node_probs > self.node_threshold)[0]
            for node_id in failed_nodes:
                risky_nodes.append({
                    "node_id": int(node_id),
                    "score": float(node_probs[node_id]),
                    "pred_time_minutes": float(timing[node_id] * Settings.Thermal.DT_MINUTES)
                })
        
        return {
            "cascade_detected": cascade_detected,
            "cascade_probability": cascade_prob,
            "risky_nodes": risky_nodes
        }