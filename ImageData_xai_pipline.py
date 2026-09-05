import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, DeepLift, GradientShap  # ← Changed
from medmnist import BreastMNIST
from copy import deepcopy
import os
import warnings

warnings.filterwarnings("ignore")

# ====================== CONFIG ======================
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = 5
ROAR_FRACTIONS = [0.2, 0.4, 0.6, 0.8]
STEPS = 8
SAMPLE_SIZE = 100

os.makedirs("Image_data_results/results_images", exist_ok=True)
os.makedirs("Image_data_results/models_images", exist_ok=True)
os.makedirs("Image_data_results/xai_images", exist_ok=True)
os.makedirs("Image_data_results/figures_images", exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================== CNN MODEL ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ====================== DATA ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

print("Loading BreastMNIST from local files...")
full_train = BreastMNIST(split='train', transform=transform, download=False)
full_val = BreastMNIST(split='val', transform=transform, download=False)
full_test = BreastMNIST(split='test', transform=transform, download=False)

np.random.seed(42)
train_idx = np.random.choice(len(full_train), min(SAMPLE_SIZE, len(full_train)), replace=False)
val_idx = np.random.choice(len(full_val), min(SAMPLE_SIZE, len(full_val)), replace=False)
test_idx = np.random.choice(len(full_test), min(SAMPLE_SIZE, len(full_test)), replace=False)

train_dataset = Subset(full_train, train_idx)
val_dataset = Subset(full_val, val_idx)
test_dataset = Subset(full_test, test_idx)

print(f"Using small subset → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


# ====================== MASKED DATASET ======================
class MaskedDataset(Dataset):
    def __init__(self, base_dataset, mask):
        self.base = base_dataset
        self.mask = mask

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = img.clone()
        img[0][~self.mask] = 0.0
        return img, label


# ====================== TRAIN ======================
def train_cnn(train_ds, val_ds, seed, epochs=EPOCHS):
    set_seed(seed)
    model = SimpleCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_val_loss = float("inf")
    best_state = None
    patience, counter = 5, 0

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.float().to(DEVICE)
                val_loss += criterion(model(images), labels).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ====================== EVALUATE ======================
def evaluate_cnn(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().flatten())

    return {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall": recall_score(all_labels, all_preds, zero_division=0),
        "F1": f1_score(all_labels, all_preds, zero_division=0),
        "AUC-ROC": roc_auc_score(all_labels, all_probs)
    }


# ====================== CAPTUM (with GradientShap) ======================
def get_image_attributions(model, method_name, images):
    model.eval()
    images = images.to(DEVICE).requires_grad_()

    if method_name == "IntegratedGradients":
        attr = IntegratedGradients(model).attribute(images, n_steps=20)

    elif method_name == "DeepLift":
        attr = DeepLift(model).attribute(images)

    elif method_name == "GradientShap":
        # GradientShap requires baselines
        baselines = torch.zeros_like(images).to(DEVICE)
        attr = GradientShap(model).attribute(
            images,
            baselines=baselines,
            n_samples=25
        )
    else:
        raise ValueError("Unknown method")

    return attr.detach().cpu()


# ====================== DELETION + INSERTION ======================
def deletion_insertion_images(model, images, attributions, steps=STEPS):
    model.eval()
    batch_size = images.shape[0]
    n_pixels = 28 * 28

    deletion_scores = []
    insertion_scores = []

    for b in range(batch_size):
        img = images[b:b + 1].clone()
        attr = attributions[b].abs().squeeze().detach().cpu().numpy()
        order = np.argsort(-attr.flatten())

        with torch.no_grad():
            orig_prob = torch.sigmoid(model(img.to(DEVICE))).item()

        # Deletion
        del_probs = [orig_prob]
        img_del = img.clone()
        for i in range(1, steps + 1):
            k = int(i * n_pixels / steps)
            mask = np.ones(n_pixels, dtype=bool)
            mask[order[:k]] = False
            mask = mask.reshape(28, 28)
            img_del[0, 0][~torch.from_numpy(mask)] = 0.0
            with torch.no_grad():
                prob = torch.sigmoid(model(img_del.to(DEVICE))).item()
            del_probs.append(prob)
        deletion_scores.append(np.trapz(del_probs) / steps)

        # Insertion
        ins_probs = [0.5]
        img_ins = torch.zeros_like(img)
        for i in range(1, steps + 1):
            k = int(i * n_pixels / steps)
            mask = np.zeros(n_pixels, dtype=bool)
            mask[order[:k]] = True
            mask = mask.reshape(28, 28)
            img_ins[0, 0][torch.from_numpy(mask)] = img[0, 0][torch.from_numpy(mask)]
            with torch.no_grad():
                prob = torch.sigmoid(model(img_ins.to(DEVICE))).item()
            ins_probs.append(prob)
        insertion_scores.append(np.trapz(ins_probs) / steps)

    return np.mean(deletion_scores), np.mean(insertion_scores)


# ====================== ROAR ======================
def compute_roar_images(model, method_name, seed):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    images, _ = next(iter(train_loader))
    attr = get_image_attributions(model, method_name, images)

    importance = attr.abs().mean(dim=0).squeeze().detach().cpu().numpy()
    order = np.argsort(-importance.flatten())

    roar_results = {}
    metrics = evaluate_cnn(model, test_dataset)
    roar_results[0.0] = metrics["AUC-ROC"]
    print(f"      ROAR 0% removed → AUC: {metrics['AUC-ROC']:.4f}")

    for frac in ROAR_FRACTIONS:
        k = int(frac * 28 * 28)
        mask = np.ones(28 * 28, dtype=bool)
        mask[order[:k]] = False
        mask = mask.reshape(28, 28)

        train_masked = MaskedDataset(train_dataset, mask)
        val_masked = MaskedDataset(val_dataset, mask)
        test_masked = MaskedDataset(test_dataset, mask)

        model_red = train_cnn(train_masked, val_masked, seed, epochs=10)
        metrics_red = evaluate_cnn(model_red, test_masked)
        roar_results[frac] = metrics_red["AUC-ROC"]
        print(f"      ROAR {int(frac * 100)}% removed → AUC: {metrics_red['AUC-ROC']:.4f}")

    return roar_results


# ====================== MAIN ======================
all_results = []

print(f"\nStarting BreastMNIST experiment with GradientShap ({N_SEEDS} seeds)...\n")

for seed in range(N_SEEDS):
    print(f"\n{'=' * 60}")
    print(f"SEED {seed + 1} of {N_SEEDS}")
    print(f"{'=' * 60}")

    model = train_cnn(train_dataset, val_dataset, seed)
    torch.save(model.state_dict(), f"Image_data_results/models_images/cnn_seed{seed}.pt")

    metrics = evaluate_cnn(model, test_dataset)
    print("Baseline:", {k: round(v, 4) for k, v in metrics.items()})

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    sample_images, _ = next(iter(test_loader))

    # ← Updated methods
    for method in ["IntegratedGradients", "DeepLift", "GradientShap"]:
        print(f"\n  → {method}")
        attr = get_image_attributions(model, method, sample_images)
        torch.save(attr, f"Image_data_results/xai_images/{method}_seed{seed}.pt")

        # Visualisation
        plt.figure(figsize=(10, 4))
        for i in range(min(4, len(sample_images))):
            plt.subplot(2, 4, i + 1)
            plt.imshow(sample_images[i].squeeze().detach().cpu().numpy(), cmap="gray")
            plt.title("Original")
            plt.axis("off")
            plt.subplot(2, 4, i + 5)
            plt.imshow(attr[i].squeeze().detach().cpu().numpy(), cmap="hot")
            plt.title(method)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"Image_data_results/figures_images/{method}_seed{seed}.png", dpi=150)
        plt.close()

        del_score, ins_score = deletion_insertion_images(model, sample_images, attr)
        print(f"     Deletion: {del_score:.4f} | Insertion: {ins_score:.4f}")

        print("     Running ROAR...")
        roar_scores = compute_roar_images(model, method, seed)

        row = {
            "Seed": seed,
            "Method": method,
            **metrics,
            "Deletion_Score": del_score,
            "Insertion_Score": ins_score
        }
        for frac, auc in roar_scores.items():
            row[f"ROAR_{int(frac * 100)}%"] = auc
        all_results.append(row)

# ====================== SAVE ======================
results_df = pd.DataFrame(all_results)
results_df.to_csv("Image_data_results/results_images/breastmnist_gradientshap.csv", index=False)

summary = results_df.groupby("Method").agg(["mean", "std"]).round(4)
summary.to_csv("Image_data_results/results_images/summary_gradientshap.csv")

print("\n" + "=" * 60)
print("FINAL SUMMARY (with GradientShap)")
print("=" * 60)
print(summary)
print("\nResults saved!")