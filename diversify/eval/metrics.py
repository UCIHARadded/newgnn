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
            # Handle both 2-item and 3-item returns
            if len(batch) == 3:
                data, label, _ = batch
            elif len(batch) == 2:
                data, label = batch
            else:
                raise ValueError(f"Unexpected number of items in batch: {len(batch)}")
                
            data = data.cuda()
            feat = model.featurizer(data)
            features.append(feat.cpu().numpy())
            labels.append(label.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            # Handle both 2-item and 3-item returns
            if len(batch) == 3:
                data, labels, _ = batch
            elif len(batch) == 2:
                data, labels = batch
            else:
                raise ValueError(f"Unexpected number of items in batch: {len(batch)}")
                
            data, labels = data.cuda(), labels.cuda()
            outputs = model.predict(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def compute_silhouette(features, labels):
    if len(features) > 5000:
        indices = np.random.choice(len(features), 5000, replace=False)
        features = features[indices]
        labels = labels[indices]
    return silhouette_score(features, labels)

def compute_davies_bouldin(features, labels):
    if len(features) > 5000:
        indices = np.random.choice(len(features), 5000, replace=False)
        features = features[indices]
        labels = labels[indices]
    return davies_bouldin_score(features, labels)

def compute_h_divergence(features1, features2, discriminator):
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
        for loss_key in history['losses']:
            plt.plot(history['losses'][loss_key], label=f'{name} {loss_key}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
