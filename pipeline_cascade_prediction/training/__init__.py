"""
Training Package
================
Modular training loop, metrics, and visualization utilities.
"""

from .trainer import Trainer
from .metrics import calculate_batch_metrics, aggregate_epoch_metrics
from .visualization import plot_training_curves, save_history

__all__ = [
    'Trainer',
    'calculate_batch_metrics',
    'aggregate_epoch_metrics',
    'plot_training_curves',
    'save_history'
]