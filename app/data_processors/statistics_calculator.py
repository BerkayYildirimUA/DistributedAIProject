# Calculate and visualize statistics for the following:
#
# Computer vision module:
# - Objects detected by YOLO vs actual objects
# - Estimated distance to vehicle in front by radar vs actual distance to vehicle in front
#
# Reinforcement learning module:
# - Learnt driving speed vs speed limit
# - Learnt G-force vs comfortable G-force
# - Learnt following distance vs comfortable following distance


# metrics_plotter.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Sequence, Optional


class StatisticsCalculator:
    """
    Collects time series for multiple metrics and plots:
      - actual vs estimated vs frame_id
      - percentage error vs frame_id

    One row per metric, 2 columns:
      [estimated vs actual]  [percentage error]
    """

    def __init__(self) -> None:
        # name -> data dict
        self._metrics: Dict[str, Dict[str, np.ndarray]] = {}

    def add_metric(
        self,
        name: str,
        frame_ids: Sequence[int],
        actual: Sequence[float],
        estimated: Sequence[float],
    ) -> None:
        """
        Register a metric time series.

        Args:
            name: Metric name, e.g. "lead_distance", "objects_in_front".
            frame_ids: Sequence of frame indices (or timestamps).
            actual: Ground truth values.
            estimated: Estimated/predicted values.
        """
        frame_ids = np.asarray(frame_ids)
        actual = np.asarray(actual, dtype=float)
        estimated = np.asarray(estimated, dtype=float)

        if not (len(frame_ids) == len(actual) == len(estimated)):
            raise ValueError("frame_ids, actual, and estimated must have same length")

        self._metrics[name] = {
            "frame_ids": frame_ids,
            "actual": actual,
            "estimated": estimated,
        }

    @staticmethod
    def _percentage_error(actual: np.ndarray, estimated: np.ndarray) -> np.ndarray:
        """
        Compute percentage error = (estimated - actual) / actual * 100.

        For actual == 0, returns NaN to avoid division by zero.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            err = (estimated - actual) / actual * 100.0
            err[~np.isfinite(err)] = np.nan  # inf, -inf, nan -> nan
        return err

    def plot_all(
        self,
        output_path: str,
        *,
        suptitle: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """
        Create one big PNG/JPG with all metrics:

        Row i:
          [estimated vs actual]  [percentage error]

        Args:
            output_path: e.g. "metrics.png" or "metrics.jpg".
            suptitle: Optional figure title.
            dpi: Resolution of saved figure.
        """
        if not self._metrics:
            raise RuntimeError("No metrics added. Call add_metric() first.")

        n_metrics = len(self._metrics)
        # Height scales with number of metrics
        fig_height = 3.0 * n_metrics
        fig, axes = plt.subplots(
            nrows=n_metrics,
            ncols=2,
            figsize=(12, fig_height),
            squeeze=False,
        )

        for row_idx, (name, data) in enumerate(self._metrics.items()):
            frame_ids = data["frame_ids"]
            actual = data["actual"]
            estimated = data["estimated"]
            err_pct = self._percentage_error(actual, estimated)

            # Left: actual vs estimated
            ax_val = axes[row_idx, 0]
            ax_val.plot(frame_ids, actual, label="actual")
            ax_val.plot(frame_ids, estimated, linestyle="--", label="estimated")
            ax_val.set_xlabel("frame_id")
            ax_val.set_ylabel(name)
            ax_val.set_title(f"{name}: actual vs estimated")
            ax_val.legend(loc="best")

            # Right: percentage error
            ax_err = axes[row_idx, 1]
            ax_err.plot(frame_ids, err_pct)
            ax_err.set_xlabel("frame_id")
            ax_err.set_ylabel("error [%]")
            ax_err.set_title(f"{name}: % error (estimated vs actual)")

            # Optional: y=0 reference line
            ax_err.axhline(0.0, linestyle=":", linewidth=1)

        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            fig.tight_layout()

        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)








