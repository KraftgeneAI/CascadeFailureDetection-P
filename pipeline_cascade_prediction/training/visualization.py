"""
Visualization Module
====================
Saves training history and plots 3x3 progress grids.
"""

import json
import os

def save_history(history: dict, output_dir: str):
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to {history_path}")

def plot_training_curves(history: dict, output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
        return
        
    if not history['train_loss']:
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Losses
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Total Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cascade F1
    axes[0, 1].plot(epochs, history['train_cascade_f1'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_cascade_f1'], 'r-', label='Validation')
    axes[0, 1].set_title('Cascade Detection F1'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Node F1
    axes[0, 2].plot(epochs, history['train_node_f1'], 'b-', label='Train')
    axes[0, 2].plot(epochs, history['val_node_f1'], 'r-', label='Validation')
    axes[0, 2].set_title('Node Failure F1'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Cascade Prec/Recall
    axes[1, 0].plot(epochs, history['train_cascade_precision'], 'b--', label='Train Precision')
    axes[1, 0].plot(epochs, history['val_cascade_precision'], 'r--', label='Val Precision')
    axes[1, 0].plot(epochs, history['train_cascade_recall'], 'b:', label='Train Recall')
    axes[1, 0].plot(epochs, history['val_cascade_recall'], 'r:', label='Val Recall')
    axes[1, 0].set_title('Cascade Prec/Recall'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Node Prec/Recall
    axes[1, 1].plot(epochs, history['train_node_precision'], 'b--', label='Train Precision')
    axes[1, 1].plot(epochs, history['val_node_precision'], 'r--', label='Val Precision')
    axes[1, 1].plot(epochs, history['train_node_recall'], 'b:', label='Train Recall')
    axes[1, 1].plot(epochs, history['val_node_recall'], 'r:', label='Val Recall')
    axes[1, 1].set_title('Node Prec/Recall'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Timing
    axes[1, 2].plot(epochs, history.get('train_time_mae', [0]*len(epochs)), 'b-', label='Train')
    axes[1, 2].plot(epochs, history.get('val_time_mae', [0]*len(epochs)), 'r-', label='Validation')
    axes[1, 2].set_title('Cascade Timing MAE'); axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Risk MSE
    axes[2, 0].plot(epochs, history.get('train_risk_mse', [0]*len(epochs)), 'b-', label='Train')
    axes[2, 0].plot(epochs, history.get('val_risk_mse', [0]*len(epochs)), 'r-', label='Validation')
    axes[2, 0].set_title('Risk Score MSE'); axes[2, 0].legend(); axes[2, 0].grid(True, alpha=0.3)

    # 8. LR
    axes[2, 1].plot(epochs, history['learning_rate'], 'g-', label='Learning Rate')
    axes[2, 1].set_title('Learning Rate'); axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

    # 9. Accuracy
    axes[2, 2].plot(epochs, history['train_cascade_acc'], 'b-', label='Train Cascade')
    axes[2, 2].plot(epochs, history['val_cascade_acc'], 'r-', label='Val Cascade')
    axes[2, 2].plot(epochs, history['train_node_acc'], 'b--', label='Train Node')
    axes[2, 2].plot(epochs, history['val_node_acc'], 'r--', label='Val Node')
    axes[2, 2].set_title('Accuracy'); axes[2, 2].legend(); axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {plot_path}")