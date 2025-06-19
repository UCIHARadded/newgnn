import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

# Force synchronous CUDA error reporting for easier debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 🔍 Initial label sanity check
y_all = np.load('/content/GNN/diversify/data/emg/emg_y.npy')
print("✅ Unique labels in dataset:", np.unique(y_all))
print("🔢 Max label in dataset:", int(np.max(y_all)))


def compute_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]

            # 🧾 Show dtype before any operation
            print("🧾 y original dtype:", y.dtype)

            # 🔁 Convert labels properly
            try:
                y = y.type(torch.LongTensor)  # Ensure long type for classification
                y = y.to(x.device)             # Move to same device as input
                print("✅ y converted dtype:", y.dtype)

                # 🚨 Check for invalid label values
                if y.min() < 0 or y.max() >= model.args.num_classes:
                    print(f"⚠️ Label out of range: min={y.min().item()} max={y.max().item()} (expected 0 to {model.args.num_classes - 1})")
                    continue
            except Exception as e:
                print(f"❌ Label handling error: {e}")
                print("🧾 y content:", y)
                continue

            # ✅ Convert x
            try:
                x = x.cuda().float()
            except Exception as e:
                print(f"❌ x CUDA move failed: {e}")
                continue

            batch_size = x.size(0)
            device = x.device

            try:
                featurizer_params = model.featurizer.forward.__code__.co_varnames
                if 'edge_index' in featurizer_params and 'batch_size' in featurizer_params:
                    edge_index = torch.tensor([
                        list(range(batch_size - 1)),
                        list(range(1, batch_size))
                    ], dtype=torch.long).to(device)
                    preds = model.predict(x, edge_index=edge_index, batch_size=batch_size)
                else:
                    preds = model.predict(x)
            except Exception as e:
                print(f"🚨 Prediction failed: {e}")
                continue

            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"✅ Final accuracy: {acc:.4f}")
    return acc


def extract_features_labels(model, loader):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            try:
                x = x.cuda().float()
                y = y.type(torch.LongTensor).cuda()
                feats = model.extract_features(x)
                all_feats.append(feats.cpu().numpy())
                all_labels.append(y.cpu().numpy())
            except Exception as e:
                print(f"❌ Feature extraction failed: {e}")
                continue
    return np.concatenate(all_feats), np.concatenate(all_labels)


def compute_h_divergence(source_feats, target_feats, discriminator):
    source = torch.tensor(source_feats).cuda()
    target = torch.tensor(target_feats).cuda()
    feats = torch.cat([source, target], dim=0)
    labels = torch.cat([
        torch.zeros(source.shape[0], dtype=torch.long),
        torch.ones(target.shape[0], dtype=torch.long)
    ]).cuda()
    preds = discriminator(feats)
    return F.cross_entropy(preds, labels).item()


def compute_silhouette(features, labels):
    try:
        return silhouette_score(features, labels)
    except Exception as e:
        print(f"Silhouette error: {e}")
        return -1


def compute_davies_bouldin(features, labels):
    try:
        return davies_bouldin_score(features, labels)
    except Exception as e:
        print(f"Davies-Bouldin error: {e}")
        return -1


def plot_metrics(history_dict, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    for metric in ["train_acc", "valid_acc", "target_acc", "class_loss", "dis_loss"]:
        plt.figure()
        for label, values in history_dict.items():
            if metric in values:
                plt.plot(values[metric], label=label)
        plt.title(f"{metric} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/{metric}.png")
        plt.close()
