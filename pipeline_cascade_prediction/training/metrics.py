"""
Metrics Module
==============
Mathematical calculations for model evaluation and dynamic thresholding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

def calculate_batch_metrics(outputs: Dict, batch: Dict, node_threshold: float, cascade_threshold: float) -> Dict:
    node_probs = outputs['failure_probability'].squeeze(-1).sigmoid()
    node_pred = (node_probs > node_threshold).float()
    node_labels = batch['node_failure_labels']
    
    cascade_prob = node_probs.max(dim=1)[0]
    cascade_pred = (cascade_prob > cascade_threshold).float()
    cascade_labels = (node_labels.max(dim=1)[0] > 0.5).float()
    
    cascade_tp = ((cascade_pred == 1) & (cascade_labels == 1)).sum().item()
    cascade_fp = ((cascade_pred == 1) & (cascade_labels == 0)).sum().item()
    cascade_tn = ((cascade_pred == 0) & (cascade_labels == 0)).sum().item()
    cascade_fn = ((cascade_pred == 0) & (cascade_labels == 1)).sum().item()
    
    node_tp = ((node_pred == 1) & (node_labels == 1)).sum().item()
    node_fp = ((node_pred == 1) & (node_labels == 0)).sum().item()
    node_tn = ((node_pred == 0) & (node_labels == 0)).sum().item()
    node_fn = ((node_pred == 0) & (node_labels == 1)).sum().item()
    
    risk_mse = 0.0
    if 'risk_scores' in outputs and 'ground_truth_risk' in batch and batch['ground_truth_risk'] is not None:
        pred_risk = outputs['risk_scores']
        targ_risk = batch['ground_truth_risk']
        
        # Align shapes for logging: expand graph-level target [B, 7] to node-level [B, N, 7]
        if targ_risk.dim() == 2 and pred_risk.dim() == 3:
            targ_risk = targ_risk.unsqueeze(1).expand_as(pred_risk)
            
        risk_mse = nn.functional.mse_loss(pred_risk, targ_risk).item()
    
    time_mae_normed, valid_timing_nodes = 0.0, 0
    if 'cascade_timing' in outputs and 'cascade_timing' in batch:
        pred_times = outputs['cascade_timing'].squeeze(-1)   
        target_times = batch['cascade_timing']               
        total_abs_err, total_valid = 0.0, 0

        for b in range(pred_times.shape[0]):
            mask = (target_times[b] >= 0)
            n_valid = mask.sum().item()
            if n_valid == 0: continue
            total_abs_err += (pred_times[b][mask] - target_times[b][mask]).abs().sum().item()
            total_valid += n_valid

        if total_valid > 0:
            time_mae_normed = total_abs_err / total_valid
            valid_timing_nodes = 1

    return {
        'cascade_tp': cascade_tp, 'cascade_fp': cascade_fp,
        'cascade_tn': cascade_tn, 'cascade_fn': cascade_fn,
        'node_tp': node_tp, 'node_fp': node_fp,
        'node_tn': node_tn, 'node_fn': node_fn,
        'risk_mse': risk_mse, 'time_mae': time_mae_normed,
        'valid_timing_nodes': valid_timing_nodes
    }

def aggregate_epoch_metrics(metric_sums: Dict, total_batches: int, total_timing_batches: int) -> Dict:
    c_tp, c_fp = metric_sums.get('cascade_tp', 0), metric_sums.get('cascade_fp', 0)
    c_tn, c_fn = metric_sums.get('cascade_tn', 0), metric_sums.get('cascade_fn', 0)
    
    n_tp, n_fp = metric_sums.get('node_tp', 0), metric_sums.get('node_fp', 0)
    n_tn, n_fn = metric_sums.get('node_tn', 0), metric_sums.get('node_fn', 0)
    
    eps = 1e-7
    c_prec = c_tp / (c_tp + c_fp + eps)
    c_rec = c_tp / (c_tp + c_fn + eps)
    c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec + eps)
    c_acc = (c_tp + c_tn) / (c_tp + c_tn + c_fp + c_fn + eps)
    
    n_prec = n_tp / (n_tp + n_fp + eps)
    n_rec = n_tp / (n_tp + n_fn + eps)
    n_f1 = 2 * n_prec * n_rec / (n_prec + n_rec + eps)
    n_acc = (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn + eps)
    
    risk_mse = metric_sums.get('risk_mse', 0) / (total_batches + eps)
    time_mae = metric_sums.get('time_mae', 0) / total_timing_batches if total_timing_batches > 0 else 0.0
    
    return {
        'cascade_acc': c_acc, 'cascade_f1': c_f1, 'cascade_precision': c_prec, 'cascade_recall': c_rec,
        'node_acc': n_acc, 'node_f1': n_f1, 'node_precision': n_prec, 'node_recall': n_rec,
        'risk_mse': risk_mse, 'time_mae': time_mae
    }

def find_best_f1(probs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.005, 0.96, 0.005):  # <--- CHANGED HERE
        preds = (probs > t).float()
        tp = (preds * targets).sum()
        fp = (preds * (1-targets)).sum()
        fn = ((1-preds) * targets).sum()
        f1 = 2*tp / (2*tp + fp + fn + 1e-7)
        if f1 > best_f1:
            best_f1 = f1.item()
            best_thresh = t
    return best_f1, best_thresh

def find_best_fbeta(probs: torch.Tensor, targets: torch.Tensor, beta: float = 0.5) -> Tuple[float, float]:
    best_score, best_thresh = 0.0, 0.5
    beta_sq = beta**2
    for t in np.arange(0.005, 0.96, 0.005):  # <--- CHANGED HERE
        preds = (probs > t).float()
        tp = (preds * targets).sum()
        fp = (preds * (1-targets)).sum()
        fn = ((1-preds) * targets).sum()
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        score = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-7)
        if score > best_score:
            best_score = score.item()
            best_thresh = t
    return best_score, best_thresh

