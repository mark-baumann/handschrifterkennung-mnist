"""
Streamlit-App: Handschrifterkennung mit MNIST
=============================================
MNIST-Daten erkunden, Modell trainieren, Vorhersagen visualisieren, Fehleranalyse.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mnist_analysis import load_mnist, plot_samples, plot_confusion, plot_misclassified

st.set_page_config(page_title="Handschrifterkennung MNIST", layout="wide")
st.title("✍️ Handschrifterkennung mit MNIST")
st.markdown("### MNIST-Daten erkunden, trainieren, visualisieren & Fehler analysieren")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Daten erkunden", "🤖 Modell trainieren", "📊 Vorhersagen", "🔬 Fehleranalyse"
])

# ═══════════════════════════════════════════════════════════════
# Session State für Daten & Modell
# ═══════════════════════════════════════════════════════════════
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ═══════════════════════════════════════════════════════════════
# Tab 1: Daten erkunden
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 MNIST-Daten erkunden")

    if st.button("📦 MNIST-Daten laden", type="primary"):
        with st.spinner("Lade MNIST-Datensatz (ca. 11 MB)..."):
            X_train, y_train, X_test, y_test = load_mnist()
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.data_loaded = True
        st.success("✅ Daten geladen!")

    if st.session_state.data_loaded:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Trainingsbilder", f"{X_train.shape[0]:,}")
        col_b.metric("Testbilder", f"{X_test.shape[0]:,}")
        col_c.metric("Bildgröße", "28×28 = 784 Pixel")
        col_d.metric("Klassen", "0–9 (10 Ziffern)")

        st.subheader("Klassenverteilung")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        x_pos = np.arange(10)
        width = 0.35
        ax_dist.bar(x_pos - width/2, counts_train, width, label='Train', color='#4A90D9')
        ax_dist.bar(x_pos + width/2, counts_test, width, label='Test', color='#50C878')
        ax_dist.set_xlabel('Ziffer')
        ax_dist.set_ylabel('Anzahl')
        ax_dist.set_title('Klassenverteilung: Train vs. Test', fontweight='bold')
        ax_dist.set_xticks(x_pos)
        ax_dist.legend()
        st.pyplot(fig_dist)

        st.subheader("Zufällige Beispiele")
        n_samples = st.slider("Anzahl Beispiele", 5, 50, 20)
        fig_samples = plot_samples(X_train, y_train, n=n_samples)
        st.pyplot(fig_samples)

        st.subheader("Einzelne Ziffern im Detail")
        digit = st.selectbox("Ziffer auswählen", list(range(10)))
        digit_indices = np.where(y_train == digit)[0]
        n_show = min(25, len(digit_indices))
        show_idx = np.random.choice(digit_indices, n_show, replace=False)

        fig_grid, axes_grid = plt.subplots(5, 5, figsize=(8, 8))
        for i, ax in enumerate(axes_grid.flat):
            if i < n_show:
                ax.imshow(X_train[show_idx[i]].reshape(28, 28), cmap='gray')
            ax.axis('off')
        fig_grid.suptitle(f'Ziffer {digit} — {len(digit_indices):,} Beispiele im Trainingsset',
                          fontweight='bold')
        st.pyplot(fig_grid)

        st.subheader("Pixel-Intensitäts-Heatmap (Durchschnitt)")
        digit_avg = st.selectbox("Ziffer für Heatmap", list(range(10)), key="heatmap_digit")
        avg_img = X_train[y_train == digit_avg].mean(axis=0).reshape(28, 28)
        fig_heat, ax_heat = plt.subplots(figsize=(6, 6))
        im = ax_heat.imshow(avg_img, cmap='hot')
        ax_heat.set_title(f'Durchschnittsbild: Ziffer {digit_avg}', fontweight='bold')
        plt.colorbar(im, ax=ax_heat, label='Intensität')
        st.pyplot(fig_heat)

# ═══════════════════════════════════════════════════════════════
# Tab 2: Modell trainieren
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("🤖 Modell trainieren")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Bitte zuerst die Daten laden (Tab 1)!")
    else:
        st.subheader("Modell-Konfiguration")
        col1, col2, col3 = st.columns(3)
        with col1:
            hidden1 = st.number_input("Hidden Layer 1", 16, 512, 128, step=16)
        with col2:
            hidden2 = st.number_input("Hidden Layer 2", 0, 512, 64, step=16)
        with col3:
            max_iter = st.slider("Max Iterationen", 5, 100, 20)

        hidden_layers = (hidden1, hidden2) if hidden2 > 0 else (hidden1,)

        if st.button("🚀 Modell trainieren", type="primary"):
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Trainiere MLPClassifier...")
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=max_iter,
                random_state=42,
                verbose=False,
            )

            # Manuelles Training mit Progress
            model.partial_fit = False  # nicht partial_fit nutzen
            model.fit(X_train, y_train)

            progress_bar.progress(100)
            status_text.text("✅ Training abgeschlossen!")

            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            y_pred = model.predict(X_test)

            st.session_state.model = model
            st.session_state.y_pred = y_pred
            st.session_state.model_trained = True

            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Train-Accuracy", f"{train_acc:.2%}")
            col_r2.metric("Test-Accuracy", f"{test_acc:.2%}")

            st.info(f"""
            **Modell-Architektur:**
            - Input: 784 Neuronen (28×28 Pixel)
            - Hidden: {hidden_layers}
            - Output: 10 Neuronen (Ziffern 0–9)
            - Aktivierung: ReLU
            - Optimizer: Adam
            """)

# ═══════════════════════════════════════════════════════════════
# Tab 3: Vorhersagen
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("📊 Vorhersagen visualisieren")

    if not st.session_state.model_trained:
        st.warning("⚠️ Bitte zuerst das Modell trainieren (Tab 2)!")
    else:
        model = st.session_state.model
        y_pred = st.session_state.y_pred
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion(y_test, y_pred, title="Confusion Matrix — MNIST")
        st.pyplot(fig_cm)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = {
            'Ziffer': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'Support': [],
        }
        for label in range(10):
            key = str(label)
            report_df['Ziffer'].append(label)
            report_df['Precision'].append(f"{report[key]['precision']:.3f}")
            report_df['Recall'].append(f"{report[key]['recall']:.3f}")
            report_df['F1-Score'].append(f"{report[key]['f1-score']:.3f}")
            report_df['Support'].append(int(report[key]['support']))

        import pandas as pd
        st.dataframe(pd.DataFrame(report_df).set_index('Ziffer'), use_container_width=True)

        st.subheader("Zufällige Vorhersagen")
        n_show = st.slider("Anzahl Vorhersagen", 10, 100, 30, key="pred_slider")
        idx = np.random.choice(len(X_test), n_show, replace=False)

        cols = 10
        rows = (n_show + cols - 1) // cols
        fig_pred, axes_pred = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes_flat = axes_pred.flat if hasattr(axes_pred, 'flat') else [axes_pred]

        for i, ax in enumerate(axes_flat):
            if i < n_show:
                img_idx = idx[i]
                ax.imshow(X_test[img_idx].reshape(28, 28), cmap='gray')
                pred = y_pred[img_idx]
                true = y_test[img_idx]
                color = 'green' if pred == true else 'red'
                ax.set_title(f"{pred}", color=color, fontweight='bold', fontsize=8)
            ax.axis('off')

        plt.suptitle('Vorhersagen (grün = korrekt, rot = falsch)', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_pred)

# ═══════════════════════════════════════════════════════════════
# Tab 4: Fehleranalyse
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("🔬 Fehleranalyse")

    if not st.session_state.model_trained:
        st.warning("⚠️ Bitte zuerst das Modell trainieren (Tab 2)!")
    else:
        y_pred = st.session_state.y_pred
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        errors = np.where(y_test != y_pred)[0]
        n_errors = len(errors)
        error_rate = n_errors / len(y_test)

        col_e1, col_e2 = st.columns(2)
        col_e1.metric("Fehlklassifikationen", f"{n_errors:,} / {len(y_test):,}")
        col_e2.metric("Fehlerrate", f"{error_rate:.2%}")

        st.subheader("Falsch klassifizierte Bilder")
        n_mis = st.slider("Anzahl Fehler anzeigen", 5, 50, 20, key="mis_slider")
        fig_mis = plot_misclassified(X_test, y_test, y_pred, n=n_mis)
        if fig_mis:
            st.pyplot(fig_mis)

        st.subheader("Fehler pro Ziffer")
        error_by_digit = {}
        for d in range(10):
            mask = y_test == d
            n_total = np.sum(mask)
            n_wrong = np.sum(y_pred[mask] != d)
            error_by_digit[d] = (n_wrong, n_total, n_wrong / n_total if n_total > 0 else 0)

        fig_err, ax_err = plt.subplots(figsize=(10, 4))
        digits = list(range(10))
        rates = [error_by_digit[d][2] for d in digits]
        bars = ax_err.bar(digits, rates, color=['green' if r < 0.05 else 'orange' if r < 0.1 else 'red' for r in rates])
        ax_err.set_xlabel('Ziffer')
        ax_err.set_ylabel('Fehlerrate')
        ax_err.set_title('Fehlerrate pro Ziffer', fontweight='bold')
        ax_err.set_xticks(digits)
        for bar, rate in zip(bars, rates):
            ax_err.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{rate:.1%}', ha='center', fontsize=9)
        st.pyplot(fig_err)

        st.subheader("Häufigste Verwechslungen")
        error_pairs = {}
        for err_idx in errors:
            pair = (int(y_test[err_idx]), int(y_pred[err_idx]))
            error_pairs[pair] = error_pairs.get(pair, 0) + 1

        sorted_pairs = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        st.markdown("| Wahre Ziffer | Vorhergesagt | Anzahl |")
        st.markdown("|-------------|-------------|--------|")
        for (true, pred), count in sorted_pairs:
            st.markdown(f"| {true} | {pred} | {count} |")

st.sidebar.markdown("""
### 📚 Über diese App

Interaktive Exploration des **MNIST-Datensatzes** —
dem "Hello World" des Deep Learning.

**Funktionen:**
- 🔍 Datenverteilung & Beispiele erkunden
- 🤖 MLPClassifier trainieren & evaluieren
- 📊 Vorhersagen visualisieren
- 🔬 Fehler systematisch analysieren

**Code:** `mnist_analysis.py` im Repo
""")
