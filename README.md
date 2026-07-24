# Pipeline Digital Twin: Cascade Failure Prediction

A physics-informed digital twin and machine learning pipeline engineered to simulate, detect, and predict catastrophic cascade failures in fluid networks (e.g., oil, natural gas, and water distribution systems).

This framework couples hydraulic fluid dynamics (steady-state flow, transient surge pressures, and thermal friction) with spatio-temporal deep learning—combining **Graph Attention Networks (GAT)** and **Long Short-Term Memory (LSTM)** networks—to predict failure probabilities, identify causal trigger nodes, estimate time-to-failure (TTF), and evaluate 7-dimensional risk profiles in real-time.

---

## Key Features

- **Physics-Based Hydraulics Simulator:** Solves network fluid dynamics across a 118-node topology to simulate pressure gradients, volumetric flow, flow restrictions, and thermal losses.
- **Cascade Simulator Engine:** Replicates critical pipeline failure modes, including overpressure ruptures, low-pressure cavitation, surge pressures, and equipment overloading.
- **124-Dimensional SCADA Telemetry:** Encodes rich per-node time-series features comprising pressure, temperature, volumetric flow, acoustic/transient sensors, valve/compressor statuses, temporal deltas, and time-to-failure metrics.
- **Spatio-Temporal GNN Architecture:** Uses a custom batched `GraphAttentionLayer` to capture spatial pressure/flow dynamics and a `TemporalGNNCell` (LSTM) to track temporal stress accumulation across time steps.
- **Multi-Task Prediction Heads:**
  - **Failure Probability:** Predicts per-node failure probability logits.
  - **Cascade Timing:** Predicts normalized time-to-failure (TTF) using smoothed L1 and pairwise ranking losses.
  - **7D Risk Assessment:** Generates multidimensional risk scores bounded in $[0, 1]$.
  - **Causal Parent Prediction:** Identifies the originating trigger node responsible for a propagating failure.
  - **Physics Supervision:** Reconstructs node pressures (using `Softplus` for non-negativity), temperatures, and edge fluid flow rates.
- **Automated Training Engine:** Includes dynamic threshold search for optimal F1 and $F_\beta$ scores, automatic learning rate decay, mixed-precision support (`torch.amp`), and automated generation of 3x3 evaluation progress grids (`training_curves.png`).

---

## Project Structure

```text
CascadeFailureDetection-P/
├── data/                                 # Root directory for generated data (Git-ignored)
│   ├── train/                            # Generated training scenario batches (.pkl)
│   ├── val/                              # Validation scenario batches (.pkl)
│   └── test/                             # Testing scenario batches (.pkl)
├── checkpoints/                          # Saved model weights & training artifacts
│   ├── best_model.pth                    # Weights with lowest validation loss
│   ├── best_f1_model.pth                 # Weights with highest node F1 score
│   ├── training_history.json             # Per-epoch training & validation metrics
│   └── training_curves.png               # 3x3 visual summary of training progress
├── pipeline_cascade_prediction/
│   ├── data/
│   │   ├── generator/                    # Physics simulator & scenario orchestrator
│   │   ├── collation.py                  # Batch collation & temporal truncation
│   │   └── dataset.py                    # Memory-efficient PyTorch Dataset loader
│   ├── model/
│   │   ├── graph_attention.py            # Batched Graph Attention Network (GAT) layer
│   │   ├── node_mlp.py                   # 124-dim SCADA feature encoder
│   │   ├── physics_informed.py           # Multi-task Physics-Informed Loss module
│   │   ├── prediction_heads.py           # Specialized task-specific neural heads
│   │   ├── temporal_gnn.py               # Spatio-temporal GNN + LSTM cell
│   │   └── unified_model.py              # Unified Pipeline Digital Twin architecture
│   ├── training/
│   │   ├── metrics.py                    # Dynamic thresholding, F1/F-beta, & MAE metrics
│   │   ├── trainer.py                    # Training and validation orchestrator loop
│   │   └── visualization.py              # Visual progress logger & curve plotter
│   └── scripts/
│       ├── generate_data.py              # CLI for synthetic scenario generation
│       └── train_model.py                # CLI for model training
├── requirements.txt
└── README.md
```

---

## Model Architecture

The `UnifiedPipelinePredictionModel` integrates spatial message-passing with temporal sequence modeling:

1. **Input Representation:** 124-dimensional SCADA vectors are passed through a 3-layer `NodeFeatureMLP` to generate node embeddings.
2. **Spatio-Temporal Core:** A `TemporalGNNCell` passes node embeddings and edge attributes through multi-head Graph Attention (`GAT`) layers to capture spatial fluid interaction, followed by an `LSTM` cell to track pressure accumulation over time.
3. **Multi-Task Prediction & Loss Formulation:** The total optimization objective combines long-range topological risk with short-range fluid dynamics:

$$L_{\text{total}} = \lambda_{\text{pred}} L_{\text{focal}} + \lambda_{\text{risk}} L_{\text{MSE}} + \lambda_{\text{timing}} L_{\text{timing}} + \lambda_{\text{parent}} L_{\text{CE}} + \lambda_{\text{phys}} L_{\text{physics}}$$

---

## Installation

Ensure you have **Python 3.9+** and **PyTorch 2.0+** installed.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/YOUR-USERNAME/CascadeFailureDetection-P.git
   cd CascadeFailureDetection-P
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Workflow & Usage

### 1. Synthetic Data Generation

Generate synthetic scenarios ("Normal", "Stressed", and "Cascade") across the 118-node pipeline network:

```bash
python pipeline_cascade_prediction/scripts/generate_data.py \
    --normal 250 \
    --cascade 500 \
    --output-dir data \
    --topology-file data/pipeline_topology.pkl \
    --start-batch 1
```

### 2. Model Training

Train the `UnifiedPipelinePredictionModel` using the generated datasets:

```bash
python pipeline_cascade_prediction/scripts/train_model.py \
    --epochs 30 \
    --batch-size 8 \
    --lr 0.001 \
    --data-dir data \
    --output-dir checkpoints
```

### 3. Monitoring Training Progress

During training, progress metrics are automatically recorded. Upon completion, check the `checkpoints/` directory for:

- **`best_f1_model.pth`**: Model state dict saved at peak validation Node $F_1$ score.
- **`training_curves.png`**: A 3x3 plot displaying Total Loss, Cascade $F_1$, Node $F_1$, Precision/Recall, Timing MAE, Risk MSE, and Learning Rate curves over all epochs.