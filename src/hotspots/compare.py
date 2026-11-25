"""Score clusterings against the planted hotspots."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from hotspots.cluster import _radians, dbscan_km, k_distance_km, knee


def noise_scores(truth: pd.Series, labels: np.ndarray) -> dict:
    """How well background tickets are kept out of hotspots (label -1 means 'not a hotspot')."""
    is_noise, called_noise = truth.eq("noise").to_numpy(), labels == -1
    return {"noise_recall": round(float((is_noise & called_noise).sum() / is_noise.sum()), 3),
            "hotspot_tickets_dropped": int((~is_noise & called_noise).sum())}


def run(df: pd.DataFrame, min_samples: int = 8) -> pd.DataFrame:
    eps = knee(k_distance_km(df, min_samples))
    cases = {
        f"dbscan eps={eps:.2f}km (knee)": dbscan_km(df, eps, min_samples),
        "dbscan eps=0.15km (too small)": dbscan_km(df, 0.15, min_samples),
        "dbscan eps=1.50km (too big)": dbscan_km(df, 1.5, min_samples),
    }
    for k in (4, 5):
        cases[f"kmeans k={k}"] = KMeans(k, n_init=10, random_state=0).fit_predict(_radians(df))
    rows = []
    for name, lab in cases.items():
        rows.append({"method": name, "clusters": len(set(lab) - {-1}),
                     "ari": round(adjusted_rand_score(df.truth, lab), 3), **noise_scores(df.truth, lab)})
    return pd.DataFrame(rows)
