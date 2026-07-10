import math

import numpy as np


SCORE_METRIC_KEYS = [
    "score_mse",
    "score_nmse",
    "score_cos",
    "score_corr",
    "target_energy",
    "pred_energy",
]


def update_metric_sums(metric_sums, metric_counts, metrics, batch_size, keys=SCORE_METRIC_KEYS):
    if not metrics:
        return

    for key in keys:
        if key not in metrics:
            continue

        value = float(metrics[key])
        if not np.isfinite(value):
            continue

        metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
        metric_counts[key] = metric_counts.get(key, 0) + batch_size


def average_metric_sums(metric_sums, metric_counts):
    averaged = {}
    for key, total in metric_sums.items():
        count = metric_counts.get(key, 0)
        if count > 0:
            averaged[key] = total / count
    return averaged


def psnr_from_mse(mse, max_value=1.0):
    mse = max(float(mse), 1e-12)
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)
