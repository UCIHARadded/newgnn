import sys
import os
import argparse
from pathlib import Path
import torch
import pickle
import numpy as np
import collections
import traceback

# Make sure ./eval etc. are importable
sys.path.append(os.path.abspath("."))

# Imports
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
    """Analyze label distribution"""
    print(f"\n=== {name} Distribution Analysis ===")
    
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
    # Create a parser for evaluation-specific arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the training output directory')
    parser.add_argument('--test_env', type=int, required=True,
                        help='Test environment index')
    parser.add_argument('--dataset', type=str, default='emg',
                        help='Dataset name (default: emg)')
    
    # Parse evaluation arguments
    args_extra = parser.parse_args()
    
    # Initialize variables
    model = None
    history = {}
    target_loader = None
    train_loader = None

    try:
        # Get base arguments without parsing command line
        args = get_args()
        
        # Override specific arguments for evaluation
        if args_extra.dataset == 'emg':
            args.num_classes = 6
        else:
            args.num_classes = 36
        args.data_dir = './data/'
        args.dataset = args_extra.dataset
        args.output = args_extra.output_dir
        args.test_envs = [args_extra.test_env]
        args.use_gnn = False
        args.layer = 'bn'
        args.latent_domain_num = 10

        # Load data
        train_loader, _, _, target_loader, _, _, _ = get_act_dataloader(args)

        # Model
        algorithm_class = alg.get_algorithm_class(args.algorithm)
        model = algorithm_class(args).cuda()
        model.eval()
        
        # ✅ Load trained model weights - UPDATED TO LOAD BEST MODEL
        best_model_path = Path(args.output) / "best_model.pth"
        model_path = Path(args.output) / "model.pth"
        
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path), strict=False)
            print(f"✅ Loaded BEST trained model from {best_model_path}")
        elif model_path.exists():
            model.load_state_dict(torch.load(model_path), strict=False)
            print(f"✅ Loaded trained model from {model_path}")
        else:
            print("⚠️ Warning: No trained model found. Using random weights")

        # History
        history_path = Path(args.output) / "training_history.pkl"
        if history_path.exists():
            with open(history_path, "rb") as f:
                history = pickle.load(f)

        print("\n=== Evaluation Metrics on Target Domain ===")
        print(f"Dataset: {args.dataset}, Num classes: {args.num_classes}")
        print(f"Model config: use_gnn={args.use_gnn}, layer={args.layer}, latent_domains={args.latent_domain_num}")

        # Analyze distributions
        if target_loader:
            analyze_distribution(target_loader, "Target Data")
        if model and target_loader:
            analyze_predictions(model, target_loader, "Target Predictions")

        # Accuracy
        if model and target_loader:
            acc = compute_accuracy(model, target_loader)
            print("\nTest Accuracy (OOD):", acc)
        else:
            print("\n⚠️ Skipping accuracy calculation - model or data loader missing")

        # Feature extraction
        MAX_BATCHES_FOR_FEATURES = 50
        if model and train_loader and target_loader:
            try:
                train_feats_raw, train_labels = extract_features_labels(model, limit_batches(train_loader, MAX_BATCHES_FOR_FEATURES), use_bottleneck=False)
                target_feats_raw, target_labels = extract_features_labels(model, limit_batches(target_loader, MAX_BATCHES_FOR_FEATURES), use_bottleneck=False)
                train_feats_bn, _ = extract_features_labels(model, limit_batches(train_loader, MAX_BATCHES_FOR_FEATURES), use_bottleneck=True)
                target_feats_bn, _ = extract_features_labels(model, limit_batches(target_loader, MAX_BATCHES_FOR_FEATURES), use_bottleneck=True)

                print(f"\nExtracted {len(train_labels)} train samples and {len(target_labels)} target samples for metrics")
                
                # Cluster metrics
                if len(train_feats_raw) > 0 and len(train_labels) > 0:
                    sil_score = compute_silhouette(train_feats_raw, train_labels)
                    print("Silhouette Score:", sil_score)
                    db_score = compute_davies_bouldin(train_feats_raw, train_labels)
                    print("Davies-Bouldin Score:", db_score)
                else:
                    print("⚠️ Skipping cluster metrics due to insufficient data")
                
                # H-divergence
                if len(train_feats_bn) > 0 and len(target_feats_bn) > 0:
                    h_div = compute_h_divergence(train_feats_bn, target_feats_bn, model.discriminator)
                    print("H-divergence:", h_div)
                else:
                    print("⚠️ Skipping H-divergence due to insufficient data")
            except Exception as e:
                print(f"⚠️ Feature metric computation failed: {str(e)}")
                traceback.print_exc()
        else:
            print("\n⚠️ Skipping feature metrics - model or data loader missing")
            
    except Exception as e:
        print(f"⚠️ Critical error in main execution: {str(e)}")
        traceback.print_exc()
    
    # Plot training curve (safe even if previous steps failed)
    if history:
        print("\nPlotting training metrics...")
        save_dir = args.output if 'args' in locals() else args_extra.output_dir
        plot_metrics({"GNN": history}, save_dir=save_dir)

if __name__ == "__main__":
    main()
