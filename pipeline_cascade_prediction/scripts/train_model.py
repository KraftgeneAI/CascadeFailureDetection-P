import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Dynamically add the root directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import our custom modules
from pipeline_cascade_prediction.data.dataset import CascadeDataset
from pipeline_cascade_prediction.data.collation import collate_cascade_batch
from pipeline_cascade_prediction.model.unified_model import UnifiedPipelinePredictionModel
from pipeline_cascade_prediction.model.physics_informed import PhysicsInformedLoss
from pipeline_cascade_prediction.training.trainer import Trainer
from pipeline_cascade_prediction.data.generator.config import Settings

def main():
    parser = argparse.ArgumentParser(description="Train the Pipeline Digital Twin Model.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory of generated data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to save model weights")
    args = parser.parse_args()

    print("==================================================")
    print(" PIPELINE DIGITAL TWIN: MODEL TRAINING")
    print("==================================================")

    # 1. Setup Device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Datasets
    print("\nLoading datasets...")
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    train_dataset = CascadeDataset(data_dir=train_dir)
    val_dataset = CascadeDataset(data_dir=val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_cascade_batch, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cascade_batch)

    # 3. Initialize Model
    print("\nInitializing Neural Network...")
    model = UnifiedPipelinePredictionModel(
        embedding_dim=128,
        hidden_dim=128
    ).to(device)

    # 4. Initialize Physics-Informed Loss
    criterion = PhysicsInformedLoss(
        base_pressure=Settings.PipelineSystem.REFERENCE_PRESSURE_PSI,
        base_flow=Settings.PipelineSystem.BASE_FLOW_BBL_HR
    ).to(device)

    # 5. Initialize Modular Trainer
    print(f"\nInitializing Trainer (Saving to {args.output_dir}/)...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        use_amp=torch.cuda.is_available() # Fast mixed-precision training if on GPU
    )

    # 6. Start Training Loop
    print("\nStarting Training Loop...")
    trainer.train(num_epochs=args.epochs)

if __name__ == "__main__":
    main()