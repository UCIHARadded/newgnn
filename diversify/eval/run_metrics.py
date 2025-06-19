import sys
import os
sys.path.append(os.path.abspath("."))  # Make sure ./eval etc. are importable

import argparse
from pathlib import Path
import torch
import pickle
import numpy as np

# ✅ Imports
from utils.util import get_args
from datautil.getdataloader_single import get_act_dataloader
from alg import alg
from eval.metrics import (
    compute_accuracy, compute_silhouette, compute_davies_bouldin,
    compute_h_divergence, extract_features_labels, plot_metrics
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_env', type=int, required=True)
    parser.add_argument('--dataset', type=str, default='emg')
    args_extra = parser.parse_args()

    # ✅ Core Args
    args = get_args()
    # FIX: Set num_classes based on dataset
    if args_extra.dataset == 'emg':
        args.num_classes = 6  # EMG has 6 classes
    else:
        args.num_classes = 36  # For other datasets
    args.data_dir = './data/'
    args.dataset = args_extra.dataset
    args.output = args_extra.output_dir
    args.test_envs = [args_extra.test_env]
    
    # CRITICAL: Match training configuration
    args.use_gnn = False  # Must match training configuration
    args.layer = 'bn'     # Must match training configuration
    args.latent_domain_num = 10  # Must match training configuration

    # ✅ Load data
    train_loader, _, _, target_loader, _, _, _ = get_act_dataloader(args)

    # ✅ Model
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    model = algorithm_class(args).cuda()
    model.eval()
    
    # ✅ Load trained model weights
    model_path = Path(args.output) / "model.pth"
    if model_path.exists():
        # Load with strict=False to handle mismatched keys
        model.load_state_dict(torch.load(model_path), strict=False)
        print(f"✅ Loaded trained model from {model_path}")
        print("Note: Some layers were not loaded due to architecture changes")
    else:
        print("⚠️ Warning: No trained model found. Using random weights")

    # ✅ History
    history_path = Path(args.output) / "training_history.pkl"
    history = {}
    if history_path.exists():
        with open(history_path, "rb") as f:
            history = pickle.load(f)

    print("\n=== Evaluation Metrics on Target Domain ===")
    print(f"Dataset: {args.dataset}, Num classes: {args.num_classes}")
    print(f"Model config: use_gnn={args.use_gnn}, layer={args.layer}, latent_domains={args.latent_domain_num}")

    # ✅ Accuracy
    acc = compute_accuracy(model, target_loader)
    print("Test Accuracy (OOD):", acc)

    # ✅ Feature extraction
    try:
        # Debug: Check label ranges
        _, sample_labels = next(iter(target_loader))
        unique_labels = torch.unique(sample_labels)
        print(f"Unique labels in target: {unique_labels}")
        print(f"Label range: {unique_labels.min().item()} to {unique_labels.max().item()}")
        assert unique_labels.max() < args.num_classes, "Labels exceed class dimension!"
        
        train_feats, train_labels = extract_features_labels(model, train_loader)
        target_feats, target_labels = extract_features_labels(model, target_loader)

        print("Silhouette Score:", compute_silhouette(train_feats, train_labels))
        print("Davies-Bouldin Score:", compute_davies_bouldin(train_feats, train_labels))
        print("H-divergence:", compute_h_divergence(
            torch.tensor(train_feats).cuda(),
            torch.tensor(target_feats).cuda(),
            model.discriminator
        ))
    except Exception as e:
        print(f"⚠️ Feature metric computation failed: {e}")

    # ✅ Plot training curve
    if history:
        print("Plotting training metrics...")
        plot_metrics({"GNN": history}, save_dir=args.output)

if __name__ == "__main__":
    main()
