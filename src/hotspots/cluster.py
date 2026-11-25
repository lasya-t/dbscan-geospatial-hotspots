"""DBSCAN on lat/lon with the haversine metric, plus tools to pick eps and summarise hotspots."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

EARTH_RADIUS_KM = 6371.0088


def _radians(df: pd.DataFrame) -> np.ndarray:
    return np.radians(df[["lat", "lon"]].to_numpy())


def k_distance_km(df: pd.DataFrame, k: int) -> np.ndarray:
    """Sorted distance (km) from each point to its k-th nearest neighbour, for the elbow plot."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine", algorithm="ball_tree").fit(_radians(df))
    dist, _ = nn.kneighbors(_radians(df))
    return np.sort(dist[:, -1] * EARTH_RADIUS_KM)


def knee(values: np.ndarray) -> float:
    """Point of the sorted curve farthest from the straight line joining its ends."""
    y = np.asarray(values, dtype=float)
    x = np.linspace(0, 1, len(y))
    yn = (y - y[0]) / (y[-1] - y[0])
    return float(y[np.argmax(x - yn)])


def dbscan_km(df: pd.DataFrame, eps_km: float, min_samples: int) -> np.ndarray:
    model = DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples,
                   metric="haversine", algorithm="ball_tree")
    return model.fit_predict(_radians(df))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def summarise(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in sorted(set(labels) - {-1}):
        pts = df[labels == c]
        lat, lon = pts.lat.mean(), pts.lon.mean()
        d = haversine_km(lat, lon, pts.lat.to_numpy(), pts.lon.to_numpy())
        rows.append({"hotspot": c, "tickets": len(pts), "lat": round(lat, 4), "lon": round(lon, 4),
                     "p90_radius_km": round(float(np.quantile(d, 0.9)), 2)})
    return pd.DataFrame(rows).sort_values("tickets", ascending=False).reset_index(drop=True)
