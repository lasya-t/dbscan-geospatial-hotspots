"""python -m hotspots [--min-samples 8] [--out docs]"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hotspots.cluster import dbscan_km, k_distance_km, knee, summarise  # noqa: E402
from hotspots.compare import run  # noqa: E402
from hotspots.data import make_tickets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    df = make_tickets()
    kd = k_distance_km(df, args.min_samples)
    eps = knee(kd)
    labels = dbscan_km(df, eps, args.min_samples)
    print(f"tickets={len(df)}  min_samples={args.min_samples}  eps from knee={eps:.3f} km")
    print("\nhotspots\n" + summarise(df, labels).to_string(index=False))
    print("\ncomparison\n" + run(df, args.min_samples).to_string(index=False))

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.plot(kd)
    a.axhline(eps, ls="--", c="grey")
    a.set(title=f"{args.min_samples}-NN distance (sorted)", xlabel="tickets", ylabel="km")
    noise = labels == -1
    b.scatter(df.lon[noise], df.lat[noise], s=4, c="lightgrey", label="not a hotspot")
    b.scatter(df.lon[~noise], df.lat[~noise], s=6, c=labels[~noise], cmap="tab10")
    b.set(title="DBSCAN hotspots", xlabel="lon", ylabel="lat")
    b.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "hotspots.png", dpi=110)
    print(f"\nwrote {out / 'hotspots.png'}")


if __name__ == "__main__":
    main()
