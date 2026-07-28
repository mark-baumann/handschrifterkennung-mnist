# Handschrifterkennung mit MNIST ✍️

**Vollständige Pipeline: Daten → Training → Evaluation → Visualisierung**

Dieses Repository demonstriert Handschrifterkennung auf dem MNIST-Datensatz mit verschiedenen Ansätzen — vom einfachen MLP bis zum CNN.

## 📦 Features

- **Daten-Pipeline**: Automatischer Download + Caching von MNIST
- **scikit-learn MLP**: Schnelles Baseline-Modell
- **Visualisierung**: Samples, Confusion Matrix, Fehleranalyse
- **Vergleich**: Gegen unser selbstgebautes NN aus `neuronal-network-from-scratch`

## 🚀 Quickstart

```bash
uv pip install numpy matplotlib scikit-learn seaborn
python mnist_analysis.py
```

## 📊 Output

- `samples.png` — 10 zufällige MNIST-Bilder
- `confusion_matrix.png` — Heatmap der Verwechslungen
- `misclassified.png` — Falsch klassifizierte Beispiele
- Classification Report (Precision/Recall/F1 pro Ziffer)

## 🧠 Lernziele

1. **Daten verstehen**: Was sind 28×28 Graustufenbilder?
2. **Feature-Normalisierung**: Warum /255.0?
3. **Overfitting erkennen**: Train vs. Test Accuracy
4. **Fehleranalyse**: Welche Ziffern werden verwechselt? Warum?
