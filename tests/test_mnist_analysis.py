"""
Tests für die MNIST-Handschrifterkennung.

Führt Tests für Datenladen, Visualisierung und Modelltraining aus.
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Kein GUI-Backend nötig

from mnist_analysis import (
    load_mnist,
    plot_samples,
    plot_confusion,
    plot_misclassified,
    train_sklearn_mlp,
)


# ── Session-scoped Fixture: Daten nur einmal laden ──────────────
@pytest.fixture(scope="session")
def mnist_data():
    """Lädt MNIST-Daten einmal für die gesamte Test-Session."""
    return load_mnist()


class TestDatenLaden:
    """Tests für die MNIST-Datenladefunktion."""

    def test_load_mnist_returns_correct_shapes(self, mnist_data):
        """Prüft, ob die geladenen Daten die erwarteten Dimensionen haben."""
        X_train, y_train, X_test, y_test = mnist_data

        assert X_train.shape == (60000, 784)
        assert y_train.shape == (60000,)
        assert X_test.shape == (10000, 784)
        assert y_test.shape == (10000,)

    def test_load_mnist_dtypes(self, mnist_data):
        """Prüft die korrekten Datentypen."""
        X_train, y_train, X_test, y_test = mnist_data

        assert X_train.dtype == np.float32
        assert y_train.dtype == np.uint8
        assert X_test.dtype == np.float32
        assert y_test.dtype == np.uint8

    def test_load_mnist_value_ranges(self, mnist_data):
        """Prüft, ob die Pixelwerte im Bereich [0, 1] liegen."""
        X_train, _, X_test, _ = mnist_data

        assert 0.0 <= X_train.min() <= X_train.max() <= 1.0
        assert 0.0 <= X_test.min() <= X_test.max() <= 1.0

    def test_load_mnist_labels_in_range(self, mnist_data):
        """Prüft, ob alle Labels im Bereich 0-9 liegen."""
        _, y_train, _, y_test = mnist_data

        assert set(np.unique(y_train)) == set(range(10))
        assert set(np.unique(y_test)) == set(range(10))

    def test_load_mnist_no_nan(self, mnist_data):
        """Prüft, ob keine NaN-Werte in den Daten sind."""
        X_train, _, X_test, _ = mnist_data

        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_test))


class TestVisualisierung:
    """Tests für die Visualisierungsfunktionen."""

    def test_plot_samples_returns_figure(self, mnist_data):
        """Prüft, ob plot_samples eine matplotlib-Figure zurückgibt."""
        X_train, y_train, _, _ = mnist_data
        fig = plot_samples(X_train, y_train, n=5)
        assert fig is not None
        assert len(fig.axes) == 5
        matplotlib.pyplot.close(fig)

    def test_plot_samples_default_n(self, mnist_data):
        """Prüft plot_samples mit Standard-n=10."""
        X_train, y_train, _, _ = mnist_data
        fig = plot_samples(X_train, y_train)
        assert len(fig.axes) == 10
        matplotlib.pyplot.close(fig)

    def test_plot_confusion_returns_figure(self, mnist_data):
        """Prüft, ob plot_confusion eine Figure zurückgibt."""
        _, _, _, y_test = mnist_data
        y_pred = np.random.randint(0, 10, len(y_test))
        fig = plot_confusion(y_test, y_pred)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_plot_misclassified_returns_figure(self, mnist_data):
        """Prüft, ob plot_misclassified Fehlerbilder anzeigt."""
        _, _, X_test, y_test = mnist_data
        # Erzeuge künstliche Fehler
        y_pred = y_test.copy()
        y_pred[:20] = (y_pred[:20] + 1) % 10  # Erzwinge Fehler
        fig = plot_misclassified(X_test, y_test, y_pred, n=6)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_plot_misclassified_no_errors(self, mnist_data):
        """Prüft, ob plot_misclassified None zurückgibt wenn keine Fehler."""
        _, _, X_test, y_test = mnist_data
        fig = plot_misclassified(X_test, y_test, y_test, n=10)
        assert fig is None


class TestModellTraining:
    """Tests für das MLP-Modelltraining (mit Subsets für Speichereffizienz)."""

    @pytest.fixture(autouse=True)
    def setup_data(self, mnist_data):
        """Extrahiert Subsets für speicherschonende Modelltests."""
        X_train, y_train, X_test, y_test = mnist_data
        self.X_train = X_train[:5000]
        self.y_train = y_train[:5000]
        self.X_test = X_test[:1000]
        self.y_test = y_test[:1000]

    def test_train_sklearn_mlp_accuracy(self):
        """Prüft, ob das MLP eine sinnvolle Genauigkeit erreicht."""
        model = train_sklearn_mlp(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        y_pred = model.predict(self.X_test)
        acc = np.mean(y_pred == self.y_test)

        # MLP sollte mindestens 80% auf dem Subset erreichen
        assert acc > 0.80, f"Accuracy {acc:.4f} zu niedrig"

    def test_train_sklearn_mlp_predictions_shape(self):
        """Prüft, ob predict die korrekte Shape zurückgibt."""
        model = train_sklearn_mlp(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        y_pred = model.predict(self.X_test)
        assert y_pred.shape == (1000,)

    def test_train_sklearn_mlp_predictions_in_range(self):
        """Prüft, ob alle Vorhersagen im Bereich 0-9 liegen."""
        model = train_sklearn_mlp(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        y_pred = model.predict(self.X_test)
        assert set(np.unique(y_pred)).issubset(set(range(10)))


class TestDatenKonsistenz:
    """Tests für die Konsistenz der geladenen Daten."""

    def test_train_test_no_overlap(self, mnist_data):
        """Prüft, dass Trainings- und Testdaten disjunkt sind (Stichprobe)."""
        X_train, _, X_test, _ = mnist_data

        # Vergleiche erste 1000 Samples (Hash-basiert)
        train_hashes = set(
            hash(X_train[i].tobytes()) for i in range(1000)
        )
        test_hashes = set(
            hash(X_test[i].tobytes()) for i in range(1000)
        )
        assert train_hashes.isdisjoint(test_hashes)
