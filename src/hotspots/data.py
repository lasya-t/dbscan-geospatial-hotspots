"""Synthetic service-outage tickets around Philadelphia.

Real tickets would come with a lat/lon from the service address. Here we plant hotspots of different
shapes and sizes on top of background noise so we know the right answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CENTER = (39.9526, -75.1652)
KM_PER_DEG_LAT = 111.0


def _km_to_deg(lat: float, dx_km: np.ndarray, dy_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return dy_km / KM_PER_DEG_LAT, dx_km / (KM_PER_DEG_LAT * np.cos(np.radians(lat)))


def make_tickets(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []

    def blob(name, lat, lon, n, sd_km):
        dlat, dlon = _km_to_deg(lat, rng.normal(0, sd_km, n), rng.normal(0, sd_km, n))
        parts.append(pd.DataFrame({"lat": lat + dlat, "lon": lon + dlon, "truth": name}))

    blob("center_city", 39.9526, -75.1652, 220, 0.35)
    blob("north_philly", 39.9900, -75.1550, 120, 0.25)
    blob("west_small", 39.9600, -75.2300, 45, 0.15)
    # a damaged cable along a road: long and thin, which k-means handles badly
    t = rng.uniform(0, 1, 140)
    lat0, lon0, lat1, lon1 = 39.915, -75.20, 39.925, -75.12
    dlat, dlon = _km_to_deg(39.92, rng.normal(0, 0.08, 140), rng.normal(0, 0.08, 140))
    parts.append(pd.DataFrame({"lat": lat0 + t * (lat1 - lat0) + dlat,
                               "lon": lon0 + t * (lon1 - lon0) + dlon, "truth": "cable_corridor"}))
    # background tickets scattered over ~25 x 25 km
    n = 260
    dlat, dlon = _km_to_deg(CENTER[0], rng.uniform(-12, 12, n), rng.uniform(-12, 12, n))
    parts.append(pd.DataFrame({"lat": CENTER[0] + dlat, "lon": CENTER[1] + dlon, "truth": "noise"}))
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "ticket_id", [f"T{i:05d}" for i in range(len(df))])
    return df
