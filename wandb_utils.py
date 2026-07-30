"""
W&B Experiment Tracking für MNIST-Handschrifterkennung
======================================================
Integriert Weights & Biases in die MNIST-Handschrifterkennung.
Loggt Modell-Ergebnisse, Confusion Matrices und Fehlklassifikationen.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="mnist", config={...})
    tracker.log_model_result("MLP", accuracy=0.95, train_time=12.3)
    tracker.finish()
"""

import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für MNIST-Handschrifterkennung.

    Features:
    - Modell-Ergebnisse tracken
    - Confusion Matrix visualisieren
    - Fehlklassifikationen loggen
    """

    def __init__(self, project: str = "mnist",
                 config: dict = None, tags: list = None,
                 group: str = None, job_type: str = "train",
                 notes: str = None, offline: bool = False):
        self.project = project
        self.run = None

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["mnist", "handwriting"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        import subprocess
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_model_result(self, model_name: str, accuracy: float,
                         train_time: float = None, params: dict = None):
        """Loggt Ergebnisse eines Modells."""
        metrics = {
            f"model/{model_name}/accuracy": accuracy,
        }
        if train_time is not None:
            metrics[f"model/{model_name}/train_time"] = train_time
        if params:
            for k, v in params.items():
                metrics[f"model/{model_name}/param_{k}"] = v
        self.log(metrics)

    def log_confusion_matrix(self, model_name: str, y_true: list,
                             y_pred: list, class_names: list = None):
        """Loggt eine Confusion Matrix."""
        if not self.run:
            return
        try:
            cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names or [str(i) for i in range(10)],
            )
            self.run.log({f"model/{model_name}/confusion_matrix": cm})
        except Exception:
            pass

    def log_misclassified(self, model_name: str, indices: list,
                          true_labels: list, pred_labels: list):
        """Loggt falsch klassifizierte Beispiele."""
        if not self.run:
            return
        table = wandb.Table(
            columns=["index", "true_label", "pred_label"]
        )
        for i, t, p in zip(indices, true_labels, pred_labels):
            table.add_data(i, t, p)
        self.run.log({f"model/{model_name}/misclassified": table})

    def finish(self):
        """Beendet den W&B-Run. Sicher bei mehrfachem Aufruf."""
        if self.run:
            self.run.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        return self.run is not None
