"""Update Data/model_training_performance.csv from training notebooks."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
from tensorflow.keras.callbacks import Callback

PERFORMANCE_COLUMNS = [
    "version",
    "record_type",
    "epoch",
    "train_accuracy",
    "train_loss",
    "val_accuracy",
    "val_loss",
    "train_samples",
    "val_samples",
    "batch_size",
    "epochs_planned",
    "epochs_completed",
    "best_epoch_val_accuracy",
    "best_val_accuracy",
    "best_val_loss_at_best_acc",
    "best_epoch_val_loss",
    "best_val_loss",
    "best_val_accuracy_at_best_loss",
    "early_stopping_restored_epoch",
    "eval_val_accuracy",
    "eval_val_loss",
    "val_roc_auc",
    "precision_negative",
    "recall_negative",
    "f1_negative",
    "precision_positive",
    "recall_positive",
    "f1_positive",
    "macro_f1",
    "weighted_f1",
    "train_val_acc_gap_at_best_val_acc",
    "model_architecture",
    "l2_reg",
    "augmentation",
    "optimizer",
    "lr_schedule",
    "initial_learning_rate",
    "dropout_conv",
    "dropout_dense",
    "checkpoint_monitor",
    "early_stopping_monitor",
    "early_stopping_patience",
    "notebook_file",
    "training_date",
    "notes",
]


def default_performance_csv_path(project_root: Optional[str | Path] = None) -> Path:
    root = Path(project_root or Path.cwd())
    return root / "Data" / "model_training_performance.csv"


def _empty_performance_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PERFORMANCE_COLUMNS)


def load_performance_log(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return _empty_performance_frame()
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    for column in PERFORMANCE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[PERFORMANCE_COLUMNS]


def save_performance_log(csv_path: Path | str, df: pd.DataFrame) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for column in PERFORMANCE_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out = out[PERFORMANCE_COLUMNS].fillna("")
    out.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


def clear_version_records(csv_path: Path | str, version: str) -> None:
    df = load_performance_log(csv_path)
    kept = df[df["version"] != version]
    save_performance_log(csv_path, kept)


def _epoch_row(
    version: str,
    epoch: int,
    train_accuracy: float,
    train_loss: float,
    val_accuracy: float,
    val_loss: float,
) -> dict[str, str]:
    return {
        "version": version,
        "record_type": "epoch",
        "epoch": str(epoch),
        "train_accuracy": f"{train_accuracy:.10g}",
        "train_loss": f"{train_loss:.10g}",
        "val_accuracy": f"{val_accuracy:.10g}",
        "val_loss": f"{val_loss:.10g}",
    }


def sync_epochs_from_history(
    history: Mapping[str, list[float]],
    *,
    version: str,
    csv_path: Optional[Path | str] = None,
) -> None:
    """Replace epoch rows for `version` using a Keras History object."""
    csv_path = Path(csv_path or default_performance_csv_path())
    df = load_performance_log(csv_path)
    df = df[~((df["version"] == version) & (df["record_type"] == "epoch"))]

    epoch_count = len(history.get("loss", []))
    epoch_rows = []
    for idx in range(epoch_count):
        epoch_rows.append(
            _epoch_row(
                version,
                idx + 1,
                float(history["accuracy"][idx]),
                float(history["loss"][idx]),
                float(history["val_accuracy"][idx]),
                float(history["val_loss"][idx]),
            )
        )

    if epoch_rows:
        df = pd.concat([df, pd.DataFrame(epoch_rows)], ignore_index=True)
    save_performance_log(csv_path, df)


def _best_epoch(values: list[float], mode: str) -> tuple[int, float]:
    if mode == "min":
        epoch_idx = min(range(len(values)), key=lambda i: values[i])
    else:
        epoch_idx = max(range(len(values)), key=lambda i: values[i])
    return epoch_idx + 1, float(values[epoch_idx])


def metrics_from_classification_report(report: Mapping[str, Any]) -> dict[str, str]:
    if "Negative" in report:
        negative_key, positive_key = "Negative", "Positive"
    else:
        negative_key, positive_key = "0", "1"

    negative = report[negative_key]
    positive = report[positive_key]
    return {
        "precision_negative": f"{negative['precision']:.2f}",
        "recall_negative": f"{negative['recall']:.2f}",
        "f1_negative": f"{negative['f1-score']:.2f}",
        "precision_positive": f"{positive['precision']:.2f}",
        "recall_positive": f"{positive['recall']:.2f}",
        "f1_positive": f"{positive['f1-score']:.2f}",
        "macro_f1": f"{report['macro avg']['f1-score']:.2f}",
        "weighted_f1": f"{report['weighted avg']['f1-score']:.2f}",
    }


def write_training_summary(
    *,
    version: str,
    history: Mapping[str, list[float]],
    csv_path: Optional[Path | str] = None,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    epochs_planned: Optional[int] = None,
    eval_val_accuracy: Optional[float] = None,
    eval_val_loss: Optional[float] = None,
    val_roc_auc: Optional[float] = None,
    classification_report_dict: Optional[Mapping[str, Any]] = None,
    early_stopping_restored_epoch: Optional[int] = None,
    model_architecture: str = "",
    l2_reg: Optional[float] = None,
    augmentation: str = "",
    optimizer: str = "",
    lr_schedule: str = "",
    initial_learning_rate: Optional[float] = None,
    dropout_conv: Optional[float] = None,
    dropout_dense: Optional[float] = None,
    checkpoint_monitor: str = "",
    early_stopping_monitor: str = "",
    early_stopping_patience: Optional[int] = None,
    notebook_file: str = "",
    training_date: Optional[str] = None,
    notes: str = "",
) -> None:
    csv_path = Path(csv_path or default_performance_csv_path())
    df = load_performance_log(csv_path)
    df = df[~((df["version"] == version) & (df["record_type"] == "summary"))]

    val_acc = [float(v) for v in history["val_accuracy"]]
    val_loss = [float(v) for v in history["val_loss"]]
    train_acc = [float(v) for v in history["accuracy"]]

    best_acc_epoch, best_val_accuracy = _best_epoch(val_acc, "max")
    best_loss_epoch, best_val_loss = _best_epoch(val_loss, "min")
    train_at_best_acc = train_acc[best_acc_epoch - 1]

    summary: dict[str, str] = {
        "version": version,
        "record_type": "summary",
        "epoch": "",
        "train_samples": "" if train_samples is None else str(train_samples),
        "val_samples": "" if val_samples is None else str(val_samples),
        "batch_size": "" if batch_size is None else str(batch_size),
        "epochs_planned": "" if epochs_planned is None else str(epochs_planned),
        "epochs_completed": str(len(val_acc)),
        "best_epoch_val_accuracy": str(best_acc_epoch),
        "best_val_accuracy": f"{best_val_accuracy:.6f}",
        "best_val_loss_at_best_acc": f"{val_loss[best_acc_epoch - 1]:.10g}",
        "best_epoch_val_loss": str(best_loss_epoch),
        "best_val_loss": f"{best_val_loss:.10g}",
        "best_val_accuracy_at_best_loss": f"{val_acc[best_loss_epoch - 1]:.10g}",
        "early_stopping_restored_epoch": ""
        if early_stopping_restored_epoch is None
        else str(early_stopping_restored_epoch),
        "eval_val_accuracy": ""
        if eval_val_accuracy is None
        else f"{eval_val_accuracy:.4f}",
        "eval_val_loss": "" if eval_val_loss is None else f"{eval_val_loss:.4f}",
        "val_roc_auc": "" if val_roc_auc is None else f"{val_roc_auc:.4f}",
        "train_val_acc_gap_at_best_val_acc": f"{train_at_best_acc - best_val_accuracy:.6f}",
        "model_architecture": model_architecture,
        "l2_reg": "" if l2_reg is None else f"{l2_reg:g}",
        "augmentation": augmentation,
        "optimizer": optimizer,
        "lr_schedule": lr_schedule,
        "initial_learning_rate": ""
        if initial_learning_rate is None
        else f"{initial_learning_rate:g}",
        "dropout_conv": "" if dropout_conv is None else str(dropout_conv),
        "dropout_dense": "" if dropout_dense is None else str(dropout_dense),
        "checkpoint_monitor": checkpoint_monitor,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_patience": ""
        if early_stopping_patience is None
        else str(early_stopping_patience),
        "notebook_file": notebook_file,
        "training_date": training_date or date.today().isoformat(),
        "notes": notes,
    }

    if classification_report_dict is not None:
        summary.update(metrics_from_classification_report(classification_report_dict))

    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    save_performance_log(csv_path, df)


class TrainingPerformanceCallback(Callback):
    """Write per-epoch rows to Data/model_training_performance.csv during fit()."""

    def __init__(
        self,
        version: str,
        csv_path: Optional[Path | str] = None,
        clear_existing: bool = True,
    ) -> None:
        super().__init__()
        self.version = version
        self.csv_path = Path(csv_path or default_performance_csv_path())
        self.clear_existing = clear_existing

    def on_train_begin(self, logs=None) -> None:
        if self.clear_existing:
            clear_version_records(self.csv_path, self.version)

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, float]] = None) -> None:
        logs = logs or {}
        df = load_performance_log(self.csv_path)
        df = df[
            ~(
                (df["version"] == self.version)
                & (df["record_type"] == "epoch")
                & (df["epoch"] == str(epoch + 1))
            )
        ]
        row = _epoch_row(
            self.version,
            epoch + 1,
            float(logs.get("accuracy", float("nan"))),
            float(logs.get("loss", float("nan"))),
            float(logs.get("val_accuracy", float("nan"))),
            float(logs.get("val_loss", float("nan"))),
        )
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_performance_log(self.csv_path, df)
