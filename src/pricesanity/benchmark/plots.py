"""Small plotting helpers used by reports and intentionally thin notebooks."""

from typing import Any

import numpy as np
import pandas as pd


def plot_learning_curve(values: pd.DataFrame, *, axis: Any = None) -> Any:
    """Plot macro-F1 against training sessions, or show an honest empty state."""

    import matplotlib.pyplot as plt

    axis = axis or plt.subplots()[1]
    required = {"training_session_count", "macro_f1", "model_name"}
    if values.empty:
        axis.text(0.5, 0.5, "No learning-curve artifacts yet.", ha="center", va="center")
        axis.set_axis_off()
        return axis
    if not required.issubset(values.columns):
        raise ValueError("Learning-curve values are missing required columns.")
    for model_name, model_values in values.groupby("model_name", sort=True):
        ordered = model_values.sort_values("training_session_count")
        axis.plot(
            ordered["training_session_count"], ordered["macro_f1"],
            marker="o", label=model_name,
        )
    axis.set_xlabel("Training sessions")
    axis.set_ylabel("Macro-F1")
    axis.legend()
    return axis


def plot_confusion_matrix(
    confusion_matrix: list[list[int]] | tuple[tuple[int, ...], ...],
    *,
    axis: Any = None,
) -> Any:
    """Render a labeled three-regime confusion matrix."""

    import matplotlib.pyplot as plt

    values = np.asarray(confusion_matrix, dtype=int)
    if values.shape != (3, 3):
        raise ValueError("Regime confusion matrices must have three rows and columns.")
    axis = axis or plt.subplots()[1]
    image = axis.imshow(values, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    labels = ("Bull", "Bear", "Range")
    axis.set_xticks(range(3), labels)
    axis.set_yticks(range(3), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Human")
    axis.figure.colorbar(image, ax=axis)
    return axis
