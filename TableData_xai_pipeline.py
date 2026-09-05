import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, DeepLift, GradientShap  # ← Changed
from copy import deepcopy
import os
import warnings

warnings.filterwarnings("ignore")

# ====================== CONFIGURATION ======================
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
HIDDEN_SIZES = [128, 64, 32]
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = 5
TARGET_COLUMN = "survived"
DATA_PATH = "simulacrum_cleaned.csv"
SAMPLE_SIZE = 100  # ← Using only 100 samples
ROAR_FRACTIONS = [0.2, 0.4, 0.6, 0.8]

os.makedirs("Table_data_results/results", exist_ok=True)
os.makedirs("Table_data_results/models", exist_ok=True)
os.makedirs("Table_data_results/xai_outputs", exist_ok=True)
os.makedirs("Table_data_results/figures", exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================== NEURAL NETWORK ======================
class TabularNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ====================== DATA LOADING ======================
def load_and_preprocess(data_path, target_col):
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    drop_cols = [c for c in df.columns if "ID" in c.upper() or "LINKNUMBER" in c.upper()]
    df = df.drop(columns=drop_cols, errors="ignore")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col].values
    X = df.drop(columns=[target_col])

    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist()


# ====================== TRAIN ======================
def train_model(X_train, y_train, X_val, y_val, input_size, seed, epochs=EPOCHS):
    set_seed(seed)
    model = TabularNN(input_size, HIDDEN_SIZES, DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1)),
        batch_size=BATCH_SIZE
    )

    best_val_loss = float("inf")
    patience, counter = 6, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()
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


# ====================== EVALUATE + PLOTS ======================
def evaluate_and_plot(model, X_test, y_test, seed):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(DEVICE)).cpu()
        probs = torch.sigmoid(logits).numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, probs)
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Seed {seed}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"Table_data_results/figures/confusion_matrix_seed{seed}.png", dpi=200)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC-ROC']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Seed {seed}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Table_data_results/figures/roc_curve_seed{seed}.png", dpi=200)
    plt.close()

    return metrics


# ====================== CAPTUM (with GradientShap) ======================
def get_attributions(model, method_name, X_sample):
    model.eval()
    inputs = torch.FloatTensor(X_sample).to(DEVICE).requires_grad_()

    if method_name == "IntegratedGradients":
        attr = IntegratedGradients(model).attribute(inputs, n_steps=30)

    elif method_name == "DeepLift":
        attr = DeepLift(model).attribute(inputs)

    elif method_name == "GradientShap":
        baselines = torch.zeros_like(inputs).to(DEVICE)
        attr = GradientShap(model).attribute(inputs, baselines=baselines, n_samples=20)

    else:
        raise ValueError("Unknown method")

    return attr.detach().cpu().numpy()


def plot_feature_importance(attr, feature_names, method, seed, top_n=10):
    importance = np.abs(attr).mean(axis=0)
    idx = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(8, 5))
    plt.barh(range(top_n), importance[idx])
    plt.yticks(range(top_n), [feature_names[i] for i in idx])
    plt.xlabel("Mean Absolute Attribution")
    plt.title(f"{method} - Top Features (Seed {seed})")
    plt.tight_layout()
    plt.savefig(f"Table_data_results/figures/{method}_importance_seed{seed}.png", dpi=200)
    plt.close()


# ====================== DELETION / INSERTION ======================
def deletion_insertion_score(model, X, y, attributions, steps=8):
    n_features = X.shape[1]
    order = np.argsort(-np.abs(attributions).mean(axis=0))

    model.eval()
    with torch.no_grad():
        base_probs = torch.sigmoid(model(torch.FloatTensor(X).to(DEVICE))).cpu().numpy().flatten()
    base_auc = roc_auc_score(y, base_probs)

    # Deletion
    X_del = X.copy()
    del_aucs = [base_auc]
    for i in range(1, steps + 1):
        k = int(i * n_features / steps)
        X_del[:, order[:k]] = 0
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.FloatTensor(X_del).to(DEVICE))).cpu().numpy().flatten()
        del_aucs.append(roc_auc_score(y, probs))
    deletion_score = np.trapz(del_aucs) / steps

    # Insertion
    X_ins = np.zeros_like(X)
    ins_aucs = [0.5]
    for i in range(1, steps + 1):
        k = int(i * n_features / steps)
        X_ins[:, order[:k]] = X[:, order[:k]]
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.FloatTensor(X_ins).to(DEVICE))).cpu().numpy().flatten()
        ins_aucs.append(roc_auc_score(y, probs))
    insertion_score = np.trapz(ins_aucs) / steps

    return deletion_score, insertion_score


# ====================== ROAR ======================
def compute_roar(X_train, y_train, X_val, y_val, X_test, y_test,
                 attributions, seed, fractions=ROAR_FRACTIONS):
    n_features = X_train.shape[1]
    feature_importance = np.abs(attributions).mean(axis=0)
    order = np.argsort(-feature_importance)

    roar_results = {}

    model_full = train_model(X_train, y_train, X_val, y_val, n_features, seed, epochs=15)
    metrics_full = evaluate_and_plot(model_full, X_test, y_test, seed)
    roar_results[0.0] = metrics_full["AUC-ROC"]

    for frac in fractions:
        k = int(frac * n_features)
        keep_idx = order[k:]

        X_train_red = X_train[:, keep_idx]
        X_val_red = X_val[:, keep_idx]
        X_test_red = X_test[:, keep_idx]

        model_red = train_model(X_train_red, y_train, X_val_red, y_val,
                                X_train_red.shape[1], seed, epochs=15)
        metrics_red = evaluate_and_plot(model_red, X_test_red, y_test, seed)
        roar_results[frac] = metrics_red["AUC-ROC"]
        print(f"      ROAR {int(frac * 100)}% removed → AUC: {metrics_red['AUC-ROC']:.4f}")

    return roar_results


# ====================== MAIN ======================
print("Loading cleaned Simulacrum data...")
X, y, feature_names = load_and_preprocess(DATA_PATH, TARGET_COLUMN)
print(f"Original samples: {len(y)} | Features: {len(feature_names)}")

# Sample only 100 rows
if len(y) > SAMPLE_SIZE:
    print(f"Sampling {SAMPLE_SIZE} rows...")
    np.random.seed(42)
    idx = np.random.choice(len(y), SAMPLE_SIZE, replace=False)
    X, y = X[idx], y[idx]
    print(f"After sampling: {X.shape}")

all_results = []

print(f"\nStarting experiment with GradientShap ({N_SEEDS} seeds, 100 samples)...\n")

for seed in range(N_SEEDS):
    print(f"\n{'=' * 60}")
    print(f"SEED {seed + 1} of {N_SEEDS}")
    print(f"{'=' * 60}")
    set_seed(seed)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    model = train_model(X_train, y_train, X_val, y_val, X.shape[1], seed)
    torch.save(model.state_dict(), f"Table_data_results/models/model_seed{seed}.pt")

    metrics = evaluate_and_plot(model, X_test, y_test, seed)
    print("Baseline:", {k: round(v, 4) for k, v in metrics.items()})

    X_sample = X_test[:min(50, len(X_test))]
    y_sample = y_test[:min(50, len(X_test))]

    # ← Updated methods
    for method in ["IntegratedGradients", "DeepLift", "GradientShap"]:
        print(f"\n  → {method}")
        attr = get_attributions(model, method, X_sample)
        np.save(f"Table_data_results/xai_outputs/{method}_seed{seed}.npy", attr)
        plot_feature_importance(attr, feature_names, method, seed)

        del_score, ins_score = deletion_insertion_score(model, X_sample, y_sample, attr)
        print(f"     Deletion: {del_score:.4f} | Insertion: {ins_score:.4f}")

        print("     Running ROAR...")
        roar_scores = compute_roar(X_train, y_train, X_val, y_val, X_test, y_test, attr, seed)

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

# ====================== FINAL RESULTS ======================
results_df = pd.DataFrame(all_results)
results_df.to_csv("Table_data_results/results/all_results_gradientshap_100.csv", index=False)

summary = results_df.groupby("Method").agg(["mean", "std"]).round(4)
summary.to_csv("Table_data_results/results/summary_gradientshap_100.csv")

print("\n" + "=" * 60)
print("FINAL SUMMARY (GradientShap | 100 samples | 5 seeds)")
print("=" * 60)
print(summary)

print("\nAll outputs saved successfully!")