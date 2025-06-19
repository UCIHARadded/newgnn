import sys
import os
sys.path.append(os.path.abspath("."))  # Make sure ./eval etc. are importable

import argparse
from pathlib import Path
import torch
import pickle
import numpy as np
import collections

# ✅ Imports
from utils.util import get_args
from datautil.getdataloader_single import get_act_dataloader
from alg import alg
from eval.metrics import (
    compute_accuracy, compute_silhouette, compute_davies_bouldin,
    compute_h_divergence, extract_features_labels, plot_metrics
)

# Helper function to limit batches
def limit_batches(loader, max_batches):
    count = 0
    for batch in loader:
        yield batch
        count += 1
        if count >= max_batches:
            break

def analyze_distribution(loader, name, max_batches=10):
    """Analyze label and prediction distribution"""
    print(f"\n=== {name} Distribution Analysis ===")
    
    # Label distribution
    label_counter = collections.Counter()
    for batch in limit_batches(loader, max_batches):
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            labels = batch[1].numpy()
            label_counter.update(labels)
    
    print("Label distribution:")
    for label, count in label_counter.most_common():
        print(f"  Class {label}: {count} samples")
    
    return label_counter

def analyze_predictions(model, loader, name, max_batches=10):
    """Analyze model prediction distribution"""
    print(f"\n=== {name} Prediction Analysis ===")
    
    # Prediction distribution
    pred_counter = collections.Counter()
    model.eval()
    with torch.no_grad():
        for batch in limit_batches(loader, max_batches):
            data = batch[0].cuda()
            outputs = model.predict(data)
            _, preds = torch.max(outputs, 1)
            pred_counter.update(preds.cpu().numpy())
    
    print("Prediction distribution:")
    for pred, count in pred_counter.most_common():
        print(f"  Class {pred}: {count} samples")
    
    return pred_counter

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

    # ✅ Analyze data distributions
    _ = analyze_distribution(target_loader, "Target Data")
    _ = analyze_predictions(model, target_loader, "Target Predictions")

    # ✅ Accuracy (on full dataset)
    acc = compute_accuracy(model, target_loader)
    print("\nTest Accuracy (OOD):", acc)

    # ✅ Feature extraction (on limited batches)
    MAX_BATCHES_FOR_FEATURES = 50
    try:
        train_feats, train_labels = extract_features_labels(model, limit_batches(train_loader, MAX_BATCHES_FOR_FEATURES))
        target_feats, target_labels = extract_features_labels(model, limit_batches(target_loader, MAX_BATCHES_FOR_FEATURES))

        print(f"\nExtracted {len(train_labels)} train samples and {len(target_labels)} target samples for metrics")
        
        # Silhouette Score
        sil_score = compute_silhouette(train_feats, train_labels)
        print("Silhouette Score:", sil_score)
        
        # Davies-Bouldin Score
        db_score = compute_davies_bouldin(train_feats, train_labels)
        print("Davies-Bouldin Score:", db_score)
        
        # H-divergence
        if len(train_feats) > 0 and len(target_feats) > 0:
            h_div = compute_h_divergence(train_feats, target_feats, model.discriminator)
            print("H-divergence:", h_div)
        else:
            print("⚠️ Skipping H-divergence due to insufficient data")
    except Exception as e:
        print(f"⚠️ Feature metric computation failed: {e}")

    # ✅ Plot training curve
    if history:
        print("\nPlotting training metrics...")
        plot_metrics({"GNN": history}, save_dir=args.output)

if __name__ == "__main__":
    main()
