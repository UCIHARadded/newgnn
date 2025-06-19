import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

def extract_features_labels(model, loader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Handle any batch structure
            if isinstance(batch, (list, tuple)):
                # First element is always data
                data = batch[0]
                
                # Find the label tensor in the batch
                for item in batch:
                    if isinstance(item, torch.Tensor) and (item.dtype == torch.int64 or item.dtype == torch.int32):
                        label = item
                        break
                else:  # If no label tensor found
                    label = batch[1] if len(batch) > 1 else None
            else:
                data = batch
                label = None
                
            if label is None:
                print("⚠️ Warning: Could not find labels in batch")
                continue
                
            data = data.cuda()
            feat = model.featurizer(data)
            features.append(feat.cpu().numpy())
            labels.append(label.cpu().numpy())
            
    features = np.concatenate(features, axis=0) if features else np.array([])
    labels = np.concatenate(labels, axis=0) if labels else np.array([])
    return features, labels

def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            # Handle any batch structure
            if isinstance(batch, (list, tuple)):
                # First element is always data
                data = batch[0]
                
                # Find the label tensor in the batch
                for item in batch:
                    if isinstance(item, torch.Tensor) and (item.dtype == torch.int64 or item.dtype == torch.int32):
                        labels = item
                        break
                else:  # If no label tensor found
                    labels = batch[1] if len(batch) > 1 else None
            else:
                data = batch
                labels = None
                
            if labels is None:
                print("⚠️ Warning: Could not find labels in batch")
                continue
                
            data, labels = data.cuda(), labels.cuda()
            outputs = model.predict(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0

def compute_silhouette(features, labels):
    if len(features) == 0 or len(labels) == 0:
        print("⚠️ Warning: No features or labels for silhouette score")
        return 0.0
    if len(features) > 5000:
        indices = np.random.choice(len(features), 5000, replace=False)
        features = features[indices]
        labels = labels[indices]
    return silhouette_score(features, labels)

def compute_davies_bouldin(features, labels):
    if len(features) == 0 or len(labels) == 0:
        print("⚠️ Warning: No features or labels for Davies-Bouldin score")
        return 0.0
    if len(features) > 5000:
        indices = np.random.choice(len(features), 5000, replace=False)
        features = features[indices]
        labels = labels[indices]
    return davies_bouldin_score(features, labels)

def compute_h_divergence(features1, features2, discriminator):
    if len(features1) == 0 or len(features2) == 0:
        print("⚠️ Warning: Empty features for H-divergence")
        return 0.0
    features1 = torch.tensor(features1).cuda()
    features2 = torch.tensor(features2).cuda()
    inputs = torch.cat([features1, features2], dim=0)
    outputs = discriminator(inputs)
    d = outputs[:,0]
    d1 = d[:features1.size(0)]
    d2 = d[features1.size(0):]
    return torch.abs(torch.mean(d1) - torch.mean(d2)).item()

def plot_metrics(history_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Accuracy plot
    plt.figure()
    for name, history in history_dict.items():
        if 'train_acc' in history and 'valid_acc' in history:
            plt.plot(history['train_acc'], label=f'{name} Train')
            plt.plot(history['valid_acc'], label=f'{name} Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()
    
    # Loss plot
    plt.figure()
    for name, history in history_dict.items():
        if 'losses' in history:
            for loss_key in history['losses']:
                plt.plot(history['losses'][loss_key], label=f'{name} {loss_key}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
