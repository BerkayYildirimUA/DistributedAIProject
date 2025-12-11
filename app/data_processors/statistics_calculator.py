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

from typing import Any, Callable, Dict, Optional, Iterable
import matplotlib.pyplot as plt
import numpy as np


Record = Dict[str, Any]
MetricsReaderFn = Callable[[str], Iterable[Record]]



def _load_series_from_reader(
    reader_fn: MetricsReaderFn,
    path: str,
    value_key: str,
    frame_key: str = "frame_id",
) -> Dict[int, float]:
    series: Dict[int, float] = {}

    for rec in reader_fn(path):
        if frame_key not in rec or value_key not in rec:
            continue

        frame_id = rec[frame_key]
        val = rec[value_key]

        try:
            frame_id_int = int(frame_id)
        except (ValueError, TypeError):
            continue

        if val is None:
            # No data (e.g., no lead vehicle) -> represent as NaN
            series[frame_id_int] = float("nan")
            continue

        try:
            val_float = float(val)
        except (ValueError, TypeError):
            # cannot convert to float, skip this record
            continue

        series[frame_id_int] = val_float

    return series



class StatisticsCalculator:
    """
    Reads metrics from files via a reader function (e.g. metrics_logger.iter_metrics),
    aligns actual vs estimated by frame_id, and plots:

      - actual vs estimated vs frame_id
      - percentage error vs frame_id

    One row per metric, 2 columns:
      [estimated vs actual]  [percentage error]
    """

    def __init__(self, reader_fn: MetricsReaderFn) -> None:
        """
        Args:
            reader_fn: A function that takes a path and yields dict records.
                       Typically metrics_logger.iter_metrics.
        """
        self._reader_fn = reader_fn
        # name -> data dict
        self._metrics: Dict[str, Dict[str, np.ndarray]] = {}

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

    def add_metric_from_files(
        self,
        name: str,
        actual_file: str,
        estimated_file: str,
        *,
        actual_key: str,
        estimated_key: str,
        frame_key: str = "frame_id",
        actual_label=None,
        estimated_label=None,
        display_name=None,
        plot_title=None,
        error_plot_title=None
    ) -> None:
        """
        Register a metric stored in two files.

        Args:
            name:
                Metric name, e.g. "lead_distance_m".
            actual_file:
                Path to file with ground-truth values (JSONL / JSONL.gz).
            estimated_file:
                Path to file with estimated values (JSONL / JSONL.gz).
                Can be the same file if keys differ.
            actual_key:
                JSON key for the actual value, e.g. 'gt_lead_distance'.
            estimated_key:
                JSON key for estimated value, e.g. 'radar_lead_distance'.
            frame_key:
                JSON key for frame id, default 'frame_id'.
        """
        actual_series = _load_series_from_reader(
            self._reader_fn, actual_file, actual_key, frame_key
        )
        est_series = _load_series_from_reader(
            self._reader_fn, estimated_file, estimated_key, frame_key
        )

        if not actual_series:
            raise ValueError(
                f"No actual data found in {actual_file} for key '{actual_key}'"
            )
        if not est_series:
            raise ValueError(
                f"No estimated data found in {estimated_file} for key '{estimated_key}'"
            )

        # Align by common frame_ids
        common_frames = sorted(set(actual_series.keys()) & set(est_series.keys()))
        if not common_frames:
            raise ValueError(
                f"No common frame_ids between actual_file={actual_file} "
                f"and estimated_file={estimated_file}"
            )

        frame_ids = np.array(common_frames, dtype=int)
        actual = np.array([actual_series[f] for f in common_frames], dtype=float)
        estimated = np.array([est_series[f] for f in common_frames], dtype=float)

        self._metrics[name] = {
            "frame_ids": frame_ids,
            "actual": actual,
            "estimated": estimated,
            "display_name":display_name,
            "actual_label":actual_label,
            "estimated_label":estimated_label,
            "plot_title":plot_title,
            "error_plot_title":error_plot_title
        }

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
            raise RuntimeError(
                "No metrics added. Call add_metric_from_files() first."
            )

        n_metrics = len(self._metrics)
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

            # Absolute error statistics
            abs_err = np.abs(estimated - actual)
            mean_abs_err = np.nanmean(abs_err)
            max_abs_err = np.nanmax(abs_err)

            # Left: actual vs estimated
            ax_val = axes[row_idx, 0]
            ax_val.plot(
                frame_ids,
                actual,
                label="Actual" if data["actual_label"] is None else data["actual_label"]
                )
            ax_val.plot(
                frame_ids,
                estimated,
                linestyle="--",
                label="Estimated" if data["estimated_label"] is None else data["estimated_label"]
            )
            ax_val.set_xlabel("frame_id")
            ax_val.set_ylabel(name if data["display_name"] is None else data["display_name"])
            ax_val.set_title(f"{name}: actual vs estimated" if data["plot_title"] is None else data["plot_title"])
            ax_val.legend(loc="best")
            ax_val.text(
                0.01,
                0.02,
                f"Mean |error|: {mean_abs_err:.3f}, Max |error|: {max_abs_err:.3f}",
                transform=ax_val.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )

            # Right: percentage error
            ax_err = axes[row_idx, 1]
            ax_err.plot(frame_ids, err_pct)
            ax_err.set_xlabel("frame_id")
            ax_err.set_ylabel("error [%]")
            ax_err.set_title(f"{name}: % error (estimated vs actual)" if data["error_plot_title"] is None else data["error_plot_title"])
            ax_err.axhline(0.0, linestyle=":", linewidth=1)
            mean_err_pct = np.nanmean(err_pct)
            max_err_pct = np.nanmax(np.abs(err_pct))

            ax_err.text(
                0.01,
                0.02,
                f"Mean % error: {mean_err_pct:.2f}, Max |% error|: {max_err_pct:.2f}",
                transform=ax_err.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )

        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            fig.tight_layout()

        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)








