"""
Tests für wandb_utils.py — W&B Experiment Tracking für MNIST-Handschrifterkennung.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wandb_utils import WANDB_AVAILABLE, WandBTracker


class TestWandBTracker:
    """Tests für den WandBTracker."""

    def test_initialization_offline(self):
        """Tracker sollte im Offline-Modus initialisieren."""
        tracker = WandBTracker(
            project="test-mnist",
            config={"model": "MLP", "hidden_layers": (100,)},
            tags=["test", "mnist"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        if WANDB_AVAILABLE:
            assert tracker.is_active
            assert tracker.run is not None
        else:
            assert not tracker.is_active
        tracker.finish()

    def test_log_model_result(self):
        """Modell-Ergebnisse sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-mnist", offline=True)
        if tracker.is_active:
            tracker.log_model_result(
                model_name="MLP",
                accuracy=0.95,
                train_time=12.3,
                params={"hidden_layers": "(100,)", "activation": "relu"},
            )
        tracker.finish()

    def test_log_confusion_matrix(self):
        """Confusion Matrix sollte ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-mnist", offline=True)
        if tracker.is_active:
            tracker.log_confusion_matrix(
                model_name="MLP",
                y_true=[0, 1, 2, 3, 4],
                y_pred=[0, 1, 2, 3, 4],
                class_names=[str(i) for i in range(10)],
            )
        tracker.finish()

    def test_log_misclassified(self):
        """Falsch klassifizierte Beispiele sollten geloggt werden."""
        tracker = WandBTracker(project="test-mnist", offline=True)
        if tracker.is_active:
            tracker.log_misclassified(
                model_name="MLP",
                indices=[5, 12, 42],
                true_labels=[3, 7, 1],
                pred_labels=[8, 2, 7],
            )
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() sollte den Run beenden und safe für doppelte Aufrufe sein."""
        tracker = WandBTracker(project="test-mnist", offline=True)
        tracker.finish()
        tracker.finish()

    def test_multiple_models(self):
        """Mehrere Modelle sollten korrekt getrackt werden."""
        tracker = WandBTracker(project="test-mnist", offline=True)
        if tracker.is_active:
            for model in ["MLP", "CNN", "SelfBuilt"]:
                tracker.log_model_result(
                    model_name=model,
                    accuracy=0.90 + hash(model) % 10 * 0.01,
                    train_time=5.0 + hash(model) % 10,
                )
        tracker.finish()
