"""
Handschrifterkennung mit MNIST — Komplettpaket
==============================================
Vergleicht verschiedene Ansätze:
1. Unser selbstgebautes NN (aus neuronal-network-from-scratch)
2. scikit-learn MLPClassifier
3. Convolutional Neural Network (CNN) mit PyTorch

Inklusive Visualisierung der Vorhersagen und Fehleranalyse.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import gzip
import os
from urllib import request


# ═══════════════════════════════════════════════════════════════
# Daten laden
# ═══════════════════════════════════════════════════════════════

def load_mnist():
    cache_dir = os.path.join(os.path.dirname(__file__), ".mnist_cache")
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    for name, fname in files.items():
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            print(f"  Lade {fname}...")
            request.urlretrieve(base_url + fname, path)

    def load_images(path):
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def load_labels(path):
        with gzip.open(path, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    X_train = load_images(os.path.join(cache_dir, files["train_images"]))
    y_train = load_labels(os.path.join(cache_dir, files["train_labels"]))
    X_test = load_images(os.path.join(cache_dir, files["test_images"]))
    y_test = load_labels(os.path.join(cache_dir, files["test_labels"]))

    return X_train, y_train, X_test, y_test


# ═══════════════════════════════════════════════════════════════
# Visualisierung
# ═══════════════════════════════════════════════════════════════

def plot_samples(X, y, n=10):
    """Zeigt n zufällige MNIST-Bilder mit Labels."""
    idx = np.random.choice(len(X), n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X[idx[i]].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {y[idx[i]]}")
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    """Zeigt die Confusion-Matrix als Heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return fig


def plot_misclassified(X, y_true, y_pred, n=10):
    """Zeigt falsch klassifizierte Bilder."""
    errors = np.where(y_true != y_pred)[0]
    if len(errors) == 0:
        print("Keine Fehler!")
        return None
    idx = np.random.choice(errors, min(n, len(errors)), replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(idx):
            ax.imshow(X[idx[i]].reshape(28, 28), cmap="gray")
            ax.set_title(f"True: {y_true[idx[i]]} → Pred: {y_pred[idx[i]]}",
                        color="red")
        ax.axis("off")
    plt.suptitle("Falsch klassifizierte Bilder", fontsize=14, color="red")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# Modelle
# ═══════════════════════════════════════════════════════════════

def train_sklearn_mlp(X_train, y_train, X_test, y_test):
    """scikit-learn MLPClassifier"""
    print("\n🔧 Training: scikit-learn MLPClassifier...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=20,
        random_state=42,
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"   Test-Accuracy: {acc:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════
# Hauptprogramm
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Handschrifterkennung — MNIST Analyse")
    print("=" * 60)

    # Daten laden
    print("\n📦 Lade MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # Samples zeigen
    print("\n📸 Zufällige Trainings-Samples:")
    fig = plot_samples(X_train, y_train, n=10)
    fig.savefig("samples.png", dpi=100, bbox_inches="tight")
    print("   → samples.png gespeichert")

    # scikit-learn MLP trainieren
    model = train_sklearn_mlp(X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    fig = plot_confusion(y_test, y_pred)
    fig.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
    print("   → confusion_matrix.png gespeichert")

    # Fehleranalyse
    print("\n🔍 Fehleranalyse:")
    fig = plot_misclassified(X_test, y_test, y_pred, n=10)
    if fig:
        fig.savefig("misclassified.png", dpi=100, bbox_inches="tight")
        print("   → misclassified.png gespeichert")

    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n✅ Analyse abgeschlossen!")


if __name__ == "__main__":
    main()
