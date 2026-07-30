# ✍️ Handschrifterkennung mit MNIST

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Aktiv-brightgreen.svg)]()

Interaktive **Handschrifterkennung** mit dem klassischen MNIST-Datensatz. Erkunde die 70.000 handgeschriebenen Ziffern, trainiere ein MLP-Modell (Multi-Layer Perceptron), visualisiere Vorhersagen und analysiere Fehlklassifikationen — alles in einer übersichtlichen Streamlit-App.

## ✨ Features

- **🔍 Daten erkunden** — MNIST-Bilder durchstöbern, Klassenverteilung und Pixelstatistiken anzeigen
- **🤖 Modell trainieren** — MLPClassifier mit konfigurierbaren Hyperparametern (Hidden Layer, Learning Rate)
- **📊 Vorhersagen visualisieren** — Zufällige Testbilder mit Modellvorhersage und Konfidenz anzeigen
- **🔬 Fehleranalyse** — Confusion Matrix, falsch klassifizierte Beispiele und deren tatsächliche vs. vorhergesagte Klasse
- **📈 Metriken** — Accuracy, Precision, Recall und F1-Score pro Ziffer
- **✅ Vollständige Testabdeckung** — Unit-Tests für Analyse- und Utility-Funktionen

## 🚀 Installation

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

## 🎯 Nutzung

```bash
# Streamlit-App starten
streamlit run app.py
```

Die App öffnet sich im Browser unter `http://localhost:8501`. Durchlaufe die vier Tabs: Daten erkunden → Modell trainieren → Vorhersagen → Fehleranalyse.

## 🧪 Tests ausführen

```bash
pytest tests/ -v
```

## 🛠️ Tech-Stack

| Technologie | Einsatz |
|-------------|---------|
| **scikit-learn** | MLPClassifier, Metriken, Confusion Matrix |
| **NumPy** | Datenverarbeitung und -transformation |
| **Matplotlib** | Visualisierung von Ziffern und Diagrammen |
| **Seaborn** | Heatmaps für Confusion Matrix |
| **Pandas** | Datenstrukturen und -analyse |
| **Streamlit** | Interaktive Web-App |
| **Pytest** | Test-Framework |

## 📁 Projektstruktur

```
handschrifterkennung-mnist/
├── app.py                  # Streamlit-Hauptapp
├── pyproject.toml          # Projekt-Konfiguration
├── mnist_analysis.py       # Daten laden, Plots, Confusion Matrix
├── wandb_utils.py          # W&B-Integration
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
