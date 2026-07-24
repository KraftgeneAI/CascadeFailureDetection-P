"""
Trainer Module (Pipeline Digital Twin)
======================================
Core training loop and checkpointing manager.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict

from pipeline_cascade_prediction.data.generator.config import Settings
from .metrics import calculate_batch_metrics, aggregate_epoch_metrics, find_best_f1, find_best_fbeta
from .visualization import save_history, plot_training_curves

class Trainer:
    def __init__(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        criterion: nn.Module, device: torch.device, learning_rate: float = Settings.Training.LEARNING_RATE,
        output_dir: str = "checkpoints", max_grad_norm: float = Settings.Training.TRAINER_MAX_GRAD_NORM,
        patience: int = Settings.Training.PATIENCE, use_amp: bool = False
    ):
        self.model = model.to(device)
        self.train_loader, self.val_loader, self.criterion, self.device = train_loader, val_loader, criterion, device
        self.max_grad_norm, self.output_dir, self.patience, self.use_amp = max_grad_norm, output_dir, patience, use_amp
        
        os.makedirs(output_dir, exist_ok=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Settings.Training.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=Settings.Training.SCHEDULER_PATIENCE)
        self.scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
        
        self.history = {k: [] for k in [
            'train_loss', 'val_loss', 'train_cascade_acc', 'val_cascade_acc', 'train_cascade_f1', 'val_cascade_f1',
            'train_cascade_precision', 'val_cascade_precision', 'train_cascade_recall', 'val_cascade_recall',
            'train_node_acc', 'val_node_acc', 'train_node_f1', 'val_node_f1', 'train_node_precision', 'val_node_precision',
            'train_node_recall', 'val_node_recall', 'train_time_mae', 'val_time_mae', 'train_risk_mse', 'val_risk_mse', 'learning_rate'
        ]}
        
        self.start_epoch, self.epochs_without_improvement = 0, 0
        self.best_val_loss, self.best_val_timing_loss, self.best_val_f1 = float('inf'), float('inf'), 0.0
        self.cascade_threshold, self.node_threshold = Settings.Training.CASCADE_THRESHOLD, Settings.Training.NODE_THRESHOLD
    
    def _prepare_targets(self, batch_device: Dict) -> Dict[str, torch.Tensor]:
        targets = {
            'failure_label': batch_device['node_failure_labels'], 'ground_truth_risk': batch_device.get('ground_truth_risk'),
            'cascade_timing': batch_device.get('cascade_timing'), 'parent_labels': batch_device.get('parent_labels'),
        }
        scada, edge_attr = batch_device.get('scada_data'), batch_device.get('edge_attr')
        if scada is not None and scada.dim() == 4:
            targets['physics_pressure_target'] = scada[:, -1, :, 0]
            targets['physics_temp_target'] = scada[:, -1, :, 5]
        if edge_attr is not None:
            targets['physics_flow_target'] = edge_attr[:, -1, :, 2] if edge_attr.dim() == 4 else edge_attr[:, :, 2]
        return targets
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss, grad_norms, metric_sums, total_timing_batches = 0.0, [], {}, 0

        pbar = tqdm(self.train_loader, desc="Training", mininterval=240.0)
        for batch_idx, batch in enumerate(pbar):
            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if 'node_failure_labels' not in batch_device: continue

            self.optimizer.zero_grad()
            graph_properties = batch_device.get('graph_properties', {})
            graph_properties.setdefault('edge_index', batch_device['edge_index'])
            
            targets = self._prepare_targets(batch_device)
            edge_mask = batch_device.get('edge_mask')
            if edge_mask is not None and edge_mask.dim() == 3: edge_mask = edge_mask[:, -1, :]

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch_device)
                    loss, _ = self.criterion(outputs, targets, graph_properties, edge_mask=edge_mask)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norms.append(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item())
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_device)
                loss, _ = self.criterion(outputs, targets, graph_properties, edge_mask=edge_mask)
                loss.backward()
                grad_norms.append(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item())
                self.optimizer.step()
            
            total_loss += loss.item()
            with torch.no_grad():
                batch_metrics = calculate_batch_metrics(outputs, batch_device, self.node_threshold, self.cascade_threshold)
            for k, v in batch_metrics.items(): metric_sums[k] = metric_sums.get(k, 0) + v
            if batch_metrics['valid_timing_nodes'] > 0: total_timing_batches += 1

            pbar.set_description(f"Training (Loss: {loss.item():.4f})")
            
        epoch_metrics = aggregate_epoch_metrics(metric_sums, len(self.train_loader), total_timing_batches)
        epoch_metrics['loss'] = total_loss / (len(self.train_loader) + 1e-7)
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_timing_batches = 0.0, 0
        all_node_probs, all_node_labels, all_cascade_probs, all_cascade_labels = [], [], [], []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", mininterval=240.0)
            for batch in pbar:
                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if 'node_failure_labels' not in batch_device: continue

                outputs = self.model(batch_device)
                graph_properties = batch_device.get('graph_properties', {})
                graph_properties.setdefault('edge_index', batch_device['edge_index'])
                
                targets = self._prepare_targets(batch_device)
                edge_mask = batch_device.get('edge_mask')
                if edge_mask is not None and edge_mask.dim() == 3: edge_mask = edge_mask[:, -1, :]

                loss, _ = self.criterion(outputs, targets, graph_properties, edge_mask=edge_mask)
                total_loss += loss.item()
                
                node_probs = outputs['failure_probability'].squeeze(-1).sigmoid() 
                all_node_probs.append(node_probs.flatten())
                all_node_labels.append(batch_device['node_failure_labels'].flatten())
                all_cascade_probs.append(node_probs.max(dim=1)[0])
                all_cascade_labels.append((batch_device['node_failure_labels'].max(dim=1)[0] > 0.5).float())

        # BULLETPROOF FIX: Prevent torch.cat crash if the validation folder is empty
        if not all_node_probs:
            print("\n  [Warning] Validation set is empty! Skipping validation metrics.")
            return {
                'loss': float('inf'),
                'node_f1': 0.0, 'cascade_f1': 0.0,
                'best_cascade_threshold': self.cascade_threshold, 'best_node_threshold': self.node_threshold,
            }

        global_node_probs, global_node_labels = torch.cat(all_node_probs), torch.cat(all_node_labels)
        global_cascade_probs, global_cascade_labels = torch.cat(all_cascade_probs), torch.cat(all_cascade_labels)
        
        best_c_f1, best_c_thresh = find_best_f1(global_cascade_probs, global_cascade_labels)
        best_n_score, best_n_thresh = find_best_fbeta(global_node_probs, global_node_labels, beta=Settings.Training.FBETA)
        
        return {
            'loss': total_loss / (len(self.val_loader) + 1e-7),
            'node_f1': best_n_score, 'cascade_f1': best_c_f1,
            'best_cascade_threshold': best_c_thresh, 'best_node_threshold': best_n_thresh,
        }
    
    def train(self, num_epochs: int):
        patience_counter = 0
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}\n" + "-" * 80)
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step(val_metrics['loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            for k in ['loss', 'cascade_f1', 'node_f1']:
                self.history[f'train_{k}'].append(train_metrics.get(k, 0))
                self.history[f'val_{k}'].append(val_metrics.get(k, 0))

            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            self.cascade_threshold, self.node_threshold = val_metrics['best_cascade_threshold'], val_metrics['best_node_threshold']
            
            if val_metrics['node_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['node_f1']
                torch.save({'model_state_dict': self.model.state_dict()}, f"{self.output_dir}/best_f1_model.pth")
                print(f"  ★ SAVED BEST F1 MODEL (nF1: {self.best_val_f1:.3f})")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save({'model_state_dict': self.model.state_dict()}, f"{self.output_dir}/best_model.pth")
                print(f"  ✓ Saved best loss model (Loss: {self.best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience: break

        save_history(self.history, self.output_dir)
        plot_training_curves(self.history, self.output_dir)