# ✍️ Handschrifterkennung mit MNIST

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mark-baumann/handschrifterkennung-mnist/blob/main/mnist_analyse.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Aktiv-brightgreen.svg)]()

**Handschrifterkennung** mit dem klassischen MNIST-Datensatz — direkt in **Google Colab**, ganz ohne App/GUI. Erkunde die 70.000 handgeschriebenen Ziffern, trainiere ein MLP-Modell (scikit-learn) und ein CNN (PyTorch), visualisiere Vorhersagen und analysiere Fehlklassifikationen.

## ✨ Features

- **🔍 Daten erkunden** — MNIST-Bilder durchstöbern, Klassenverteilung und Pixelstatistiken anzeigen
- **🤖 Modell trainieren** — MLPClassifier (scikit-learn) und CNN (PyTorch)
- **📊 Vorhersagen visualisieren** — Testbilder mit Modellvorhersage und Konfidenz anzeigen
- **🔬 Fehleranalyse** — Confusion Matrix, falsch klassifizierte Beispiele und deren tatsächliche vs. vorhergesagte Klasse
- **📈 Metriken** — Accuracy, Precision, Recall und F1-Score pro Ziffer
- **✅ Vollständige Testabdeckung** — Unit-Tests für Analyse- und Utility-Funktionen

## 🚀 In Google Colab ausführen

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mark-baumann/handschrifterkennung-mnist/blob/main/mnist_analyse.ipynb)

Einfach oben auf das **„Open in Colab"**-Abzeichen klicken oder direkt öffnen:

```text
https://colab.research.google.com/github/mark-baumann/handschrifterkennung-mnist/blob/main/mnist_analyse.ipynb
```

Das Notebook ist **Google-Colab-fähig** und vollständig eigenständig:

- Es erkennt automatisch, dass es in Colab läuft, und installiert fehlende Pakete (`numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`) selbst.
- Der MNIST-Datensatz wird beim ersten Lauf automatisch heruntergeladen und gecacht.
- Der PyTorch-CNN-Abschnitt wird übersprungen, falls `torch` nicht verfügbar ist.
- Keine Installation, keine App, keine GUI — alle Zellen von oben nach unten ausführen.

## 🖥️ Lokal (Notebook & Tests)

```bash
# Repository klonen
git clone https://github.com/mark-baumann/handschrifterkennung-mnist.git
cd handschrifterkennung-mnist

# Virtuelle Umgebung erstellen
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Abhängigkeiten installieren
uv pip install -e ".[dev]"
```

### Jupyter-Notebook

```bash
jupyter notebook mnist_analyse.ipynb
```

### Tests ausführen

```bash
pytest tests/ -v
```

## 🛠️ Tech-Stack

| Technologie | Einsatz |
|-------------|---------|
| **scikit-learn** | MLPClassifier, Metriken, Confusion Matrix |
| **PyTorch** | CNN-Training (optional, wird in Colab installiert) |
| **NumPy** | Datenverarbeitung und -transformation |
| **Matplotlib** | Visualisierung von Ziffern und Diagrammen |
| **Seaborn** | Heatmaps für Confusion Matrix |
| **Pandas** | Datenstrukturen und -analyse |
| **Pytest** | Test-Framework |

## 📁 Projektstruktur

```
handschrifterkennung-mnist/
├── mnist_analyse.ipynb      # Google-Colab-fähiges Notebook (Haupteinstiegspunkt)
├── pyproject.toml           # Projekt-Konfiguration
├── mnist_analysis.py        # Daten laden, Plots, Confusion Matrix
├── wandb_utils.py           # W&B-Integration
└── tests/
    ├── test_mnist_analysis.py
    └── test_wandb_utils.py
```

## 📖 Über den MNIST-Datensatz

MNIST (Modified National Institute of Standards and Technology) enthält **70.000 handgeschriebene Ziffern** (0–9) in 28×28 Pixel Graustufen:

- **60.000** Trainingsbilder
- **10.000** Testbilder
- 10 Klassen (Ziffern 0–9)
- Jedes Bild: 784 Features (28×28 Pixel)

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*MNIST ist das „Hello World" des Machine Learning — ideal, um Klassifikation, Modell-Evaluation und Fehleranalyse zu lernen.*