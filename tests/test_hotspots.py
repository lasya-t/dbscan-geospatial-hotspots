import numpy as np
import pandas as pd

from hotspots.cluster import dbscan_km, haversine_km, k_distance_km, knee, summarise
from hotspots.compare import run
from hotspots.data import make_tickets


def test_haversine_known_distance():
    # Philadelphia City Hall to New York City Hall is about 130 km
    assert 125 < haversine_km(39.9526, -75.1652, 40.7128, -74.0060) < 135


def test_knee_of_hockey_stick():
    y = np.r_[np.linspace(0, 1, 90), np.linspace(1.5, 10, 10)]
    assert 0.9 <= knee(y) <= 1.5


def test_eps_is_in_kilometres():
    # two points 300 m apart: together at eps=0.5 km, apart at eps=0.2 km
    df = pd.DataFrame({"lat": [39.95, 39.95 + 0.3 / 111.0], "lon": [-75.16, -75.16]})
    assert list(dbscan_km(df, 0.5, 2)) == [0, 0]
    assert list(dbscan_km(df, 0.2, 2)) == [-1, -1]


def test_finds_the_four_planted_hotspots():
    df = make_tickets()
    labels = dbscan_km(df, knee(k_distance_km(df, 8)), 8)
    assert len(set(labels) - {-1}) == 4
    s = summarise(df, labels)
    assert s.tickets.sum() == (labels != -1).sum()


def test_dbscan_beats_kmeans_and_bad_eps():
    res = run(make_tickets()).set_index("method")
    best = res.ari.idxmax()
    assert best.startswith("dbscan") and "knee" in best
    assert res.loc[best, "noise_recall"] > 0.9
