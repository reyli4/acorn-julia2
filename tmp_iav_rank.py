import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/home/fs01/jl2966/acorn-julia2/acorn-julia/runs/low_RE_mod_elec_iter0/inputs")
YEARS = set(range(1985, 1991))


def aggregate_selected_columns(path: Path):
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_idx = 0
        keep_idx = []
        keep_ts = []
        for i, col in enumerate(header):
            if i == id_idx:
                continue
            try:
                year = int(col[:4])
            except ValueError:
                continue
            if year in YEARS:
                keep_idx.append(i)
                keep_ts.append(col)

        sums = np.zeros(len(keep_idx), dtype=float)
        for row in reader:
            for j, idx in enumerate(keep_idx):
                val = row[idx]
                if val:
                    sums[j] += float(val)

    idx = pd.to_datetime(pd.Index(keep_ts), utc=True)
    return pd.Series(sums, index=idx)


def build_results():
    wind = aggregate_selected_columns(BASE / "wind_historical_1980_2019.csv")
    solar = aggregate_selected_columns(BASE / "solar_upv_historical_1980_2019.csv") + aggregate_selected_columns(
        BASE / "solar_dpv_historical_1980_2019.csv"
    )
    small_h = aggregate_selected_columns(BASE / "small_hydro_historical.csv")
    large_h = aggregate_selected_columns(BASE / "large_hydro_historical.csv")
    load = aggregate_selected_columns(BASE / "load_historical_1980_2019.csv")

    monthly = pd.DataFrame(
        {
            "wind": wind.resample("MS").sum(),
            "solar": solar.resample("MS").sum(),
            "small_hydro": small_h.resample("MS").sum(),
            "large_hydro": large_h.resample("MS").mean(),
            "load": load.resample("MS").sum(),
        }
    )
    monthly["vre"] = monthly["wind"] + monthly["solar"]
    monthly["renew_plus_small"] = monthly["vre"] + monthly["small_hydro"]
    monthly["renew_plus_hydro"] = monthly["renew_plus_small"] + monthly["large_hydro"].fillna(0)

    rows = []
    for metric in ["wind", "solar", "vre", "renew_plus_small", "renew_plus_hydro", "load"]:
        pivot = monthly.pivot_table(
            index=monthly.index.month, columns=monthly.index.year, values=metric, aggfunc="sum"
        )
        clim = pivot.mean(axis=1)
        clim_share = clim / clim.sum()
        annual_mean = pivot.sum(axis=0).mean()
        for year in sorted(YEARS):
            vec = pivot[year]
            share = vec / vec.sum()
            rows.append(
                {
                    "metric": metric,
                    "year": year,
                    "annual_total": vec.sum(),
                    "annual_dev_pct": 100 * (vec.sum() - annual_mean) / annual_mean,
                    "seasonal_profile_rms_share": float(np.sqrt(((share - clim_share) ** 2).mean())),
                    "monthly_abs_rms": float(np.sqrt(((vec - clim) ** 2).mean())),
                    "seasonal_range": float(vec.max() - vec.min()),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = build_results()
    for metric in ["wind", "solar", "vre", "renew_plus_small", "renew_plus_hydro", "load"]:
        sub = res[res.metric == metric].sort_values("seasonal_profile_rms_share", ascending=False)
        print(f"\nMETRIC {metric}")
        print(
            sub[
                [
                    "year",
                    "annual_total",
                    "annual_dev_pct",
                    "seasonal_profile_rms_share",
                    "seasonal_range",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:,.4f}")
        )
