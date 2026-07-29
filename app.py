"""
Streamlit-App: Handschrifterkennung mit MNIST
==============================================
MNIST-Daten erkunden, Modell trainieren, Vorhersagen visualisieren, Fehleranalyse.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from mnist_analysis import load_mnist, plot_samples, plot_confusion, plot_misclassified

st.set_page_config(page_title="Handschrifterkennung MNIST", page_icon="✍️", layout="wide")
st.title("✍️ Handschrifterkennung mit MNIST")
st.markdown("MNIST-Daten erkunden, Modelle trainieren, Vorhersagen visualisieren & Fehler analysieren")

page = st.sidebar.radio(
    "Bereich wählen",
    ["Daten erkunden", "Modell trainieren", "Vorhersagen", "Fehleranalyse"]
)

# ═══════════════════════════════════════════════════════════════════════════
# DATEN ERKUNDEN
# ═══════════════════════════════════════════════════════════════════════════
if page == "Daten erkunden":
    st.header("📸 MNIST-Daten erkunden")

    @st.cache_data
    def load_data():
        return load_mnist()

    with st.spinner("Lade MNIST-Daten..."):
        X_train, y_train, X_test, y_test = load_data()

    st.success(f"✅ Daten geladen: {X_train.shape[0]:,} Trainingsbilder, {X_test.shape[0]:,} Testbilder")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trainingsdaten", f"{X_train.shape[0]:,}")
    with col2:
        st.metric("Testdaten", f"{X_test.shape[0]:,}")
    with col3:
        st.metric("Bildgröße", "28×28 = 784 Pixel")
    with col4:
        st.metric("Klassen", "10 (Ziffern 0-9)")

    st.subheader("Zufällige Trainingsbilder")
    n_samples = st.slider("Anzahl Bilder", 5, 30, 10)

    if st.button("Neue Bilder anzeigen"):
        idx = np.random.choice(len(X_train), n_samples, replace=False)
        n_cols = min(n_samples, 10)
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else axes.flat

        for i, ax in enumerate(axes_flat):
            if i < n_samples:
                ax.imshow(X_train[idx[i]].reshape(28, 28), cmap="gray")
                ax.set_title(f"Label: {y_train[idx[i]]}", fontsize=10)
            ax.axis("off")

        st.pyplot(fig)

    st.subheader("Klassenverteilung")
    fig, ax = plt.subplots(figsize=(10, 4))
    unique, counts = np.unique(y_train, return_counts=True)
    ax.bar(unique, counts, color="#4ECDC4", edgecolor="white")
    ax.set_xlabel("Ziffer")
    ax.set_ylabel("Anzahl")
    ax.set_title("Verteilung der Ziffern im Trainingsdatensatz")
    ax.set_xticks(unique)
    for i, c in enumerate(counts):
        ax.text(i, c + 50, str(c), ha="center", fontsize=9)
    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# MODELL TRAINIEREN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Modell trainieren":
    st.header("🤖 Modell trainieren")

    @st.cache_data
    def load_data():
        return load_mnist()

    with st.spinner("Lade MNIST-Daten..."):
        X_train, y_train, X_test, y_test = load_data()

    col1, col2 = st.columns(2)
    with col1:
        hidden_layers = st.text_input("Hidden-Layer (z.B. 128,64)", "128,64")
        activation = st.selectbox("Aktivierung", ["relu", "tanh", "logistic"])
    with col2:
        max_iter = st.slider("Max Iterationen", 5, 50, 20)
        random_state = st.number_input("Random State", 0, 100, 42)

    if st.button("Modell trainieren", type="primary"):
        layers = tuple(int(x.strip()) for x in hidden_layers.split(","))

        with st.spinner(f"Training MLPClassifier{layers}..."):
            model = MLPClassifier(
                hidden_layer_sizes=layers,
                activation=activation,
                solver="adam",
                max_iter=max_iter,
                random_state=random_state,
            )
            model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        y_pred = model.predict(X_test)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train-Accuracy", f"{train_acc:.4f}")
        with col2:
            st.metric("Test-Accuracy", f"{test_acc:.4f}")

        st.subheader("Confusion-Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=range(10), yticklabels=range(10))
        ax.set_xlabel("Vorhergesagt")
        ax.set_ylabel("Tatsächlich")
        ax.set_title("Confusion-Matrix")
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(report)

# ═══════════════════════════════════════════════════════════════════════════
# VORHERSAGEN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Vorhersagen":
    st.header("🔮 Vorhersagen visualisieren")

    @st.cache_data
    def load_data():
        return load_mnist()

    @st.cache_resource
    def train_model():
        X_train, y_train, X_test, y_test = load_data()
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    with st.spinner("Lade Daten & trainiere Modell..."):
        model, X_test, y_test = train_model()

    y_pred = model.predict(X_test)
    test_acc = model.score(X_test, y_test)
    st.metric("Test-Accuracy", f"{test_acc:.4f}")

    st.subheader("Zufällige Test-Vorhersagen")
    n_show = st.slider("Anzahl", 5, 20, 10)

    if st.button("Neue Vorhersagen"):
        idx = np.random.choice(len(X_test), n_show, replace=False)
        n_cols = min(n_show, 5)
        n_rows = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else axes.flat

        for i, ax in enumerate(axes_flat):
            if i < n_show:
                img_idx = idx[i]
                ax.imshow(X_test[img_idx].reshape(28, 28), cmap="gray")
                true_label = y_test[img_idx]
                pred_label = y_pred[img_idx]
                color = "green" if true_label == pred_label else "red"
                ax.set_title(f"True: {true_label} → Pred: {pred_label}", color=color, fontsize=10)
            ax.axis("off")

        fig.suptitle("Vorhersagen (grün = korrekt, rot = falsch)", fontsize=14)
        st.pyplot(fig)

    st.subheader("Wahrscheinlichkeiten pro Klasse")
    img_idx = st.number_input("Bild-Index (0-9999)", 0, 9999, 0)

    if st.button("Wahrscheinlichkeiten anzeigen"):
        probs = model.predict_proba(X_test[img_idx:img_idx+1])[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.imshow(X_test[img_idx].reshape(28, 28), cmap="gray")
        ax1.set_title(f"Bild {img_idx} (True: {y_test[img_idx]})")
        ax1.axis("off")

        colors = ["#4ECDC4" if i != np.argmax(probs) else "#FF6B6B" for i in range(10)]
        ax2.bar(range(10), probs, color=colors, edgecolor="white")
        ax2.set_xlabel("Ziffer")
        ax2.set_ylabel("Wahrscheinlichkeit")
        ax2.set_title("Vorhergesagte Wahrscheinlichkeiten")
        ax2.set_xticks(range(10))

        st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════════════
# FEHLERANALYSE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Fehleranalyse":
    st.header("🔍 Fehleranalyse")

    @st.cache_data
    def load_data():
        return load_mnist()

    @st.cache_resource
    def train_model():
        X_train, y_train, X_test, y_test = load_data()
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    with st.spinner("Lade Daten & trainiere Modell..."):
        model, X_test, y_test = train_model()

    y_pred = model.predict(X_test)
    errors = np.where(y_test != y_pred)[0]
    error_rate = len(errors) / len(y_test)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test-Samples", f"{len(y_test):,}")
    with col2:
        st.metric("Fehler", f"{len(errors):,}")
    with col3:
        st.metric("Fehlerrate", f"{error_rate:.2%}")

    st.subheader("Falsch klassifizierte Bilder")
    n_errors = st.slider("Anzahl Fehler anzeigen", 5, 30, 10)

    if st.button("Fehler anzeigen"):
        n_show = min(n_errors, len(errors))
        error_idx = np.random.choice(errors, n_show, replace=False)

        n_cols = min(n_show, 5)
        n_rows = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else axes.flat

        for i, ax in enumerate(axes_flat):
            if i < n_show:
                img_idx = error_idx[i]
                ax.imshow(X_test[img_idx].reshape(28, 28), cmap="gray")
                ax.set_title(f"True: {y_test[img_idx]} → Pred: {y_pred[img_idx]}",
                            color="red", fontsize=10)
            ax.axis("off")

        fig.suptitle("Falsch klassifizierte Bilder", fontsize=14, color="red")
        st.pyplot(fig)

    st.subheader("Häufigste Verwechslungen")
    if st.button("Verwechslungen analysieren"):
        cm = confusion_matrix(y_test, y_pred)
        # Diagonale auf 0 setzen (korrekte Klassifikationen ignorieren)
        np.fill_diagonal(cm, 0)

        # Top-10 Verwechslungen finden
        flat_indices = np.argsort(cm.flatten())[-10:][::-1]
        confusions = []
        for idx in flat_indices:
            true_class = idx // 10
            pred_class = idx % 10
            count = cm[true_class, pred_class]
            if count > 0:
                confusions.append((true_class, pred_class, count))

        if confusions:
            st.markdown("| Tatsächlich | Vorhergesagt | Anzahl |")
            st.markdown("|------------|-------------|--------|")
            for true_c, pred_c, count in confusions:
                st.markdown(f"| {true_c} | {pred_c} | {count} |")
        else:
            st.info("Keine Verwechslungen gefunden!")

st.sidebar.markdown("---")
st.sidebar.markdown("📚 **Handschrifterkennung MNIST**")
st.sidebar.markdown("[GitHub Repository](https://github.com/mark-baumann/handschrifterkennung-mnist)")
