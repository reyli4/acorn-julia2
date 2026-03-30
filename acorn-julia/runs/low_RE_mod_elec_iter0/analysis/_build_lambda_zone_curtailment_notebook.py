from pathlib import Path
import json


OUT = Path(
    "acorn-julia/runs/low_RE_mod_elec_iter0/analysis/"
    "seasonal_stage1_lambda_storage_zone_curtailment_reduction.ipynb"
)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """# Stage 1 - lambda comparison of curtailment reduction in seasonal-storage zones

This notebook compares how different seasonal-storage penalty values change **wind and solar curtailment in the zones where seasonal storage is placed**. It uses a baseline run with no seasonal-storage dispatch as the reference and reports both absolute curtailment and reduction relative to baseline.
"""
    ),
    code(
        """from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Baseline run used to compute reduction values.
BASELINE_SPEC = ("baseline", ["stagel_3month_base", "stagel_3month_baseline"])

# Seasonal runs to compare. Edit this list if you want to add/remove lambda cases.
RUN_SPECS = [
    ("lambda2", ["3month_lamba_2_seasonal", "stagel_3month_lamba_2_seasonal"]),
    ("lambda3", ["3month_lamba_3.47_seasonal", "3month_lamba_3_seasonal", "stagel_3month_lamba_3_seasonal"]),
    ("lambda3.25", ["3month_lamba_3.43_seasonal"]),
    ("lambda3.3", ["3month_lamba_3.45_seasonal", "3month_lamba_3.3_seasonal"]),
    ("lambda3.4", ["3month_lamba_3.4_seasonal", "stagel_3month_lamba_3.4_seasonal"]),
    ("lambda3.5", ["3month_lamba_3.5_seasonal", "stagel_3month_lamba_3.5_seasonal"]),
    ("lambda3.75", ["3month_lamba_3.75_seasonal", "stagel_3month_lamba_3.75_seasonal"]),
    ("lambda4", ["3month_lamba_4_seasonal", "stagel_3month_lamba_4_seasonal"]),
    ("lambda5", ["3month_lamba_5_seasonal", "stagel_3month_lamba_5_seasonal"]),
]

# Use a single year for the lambda sensitivity screen. Set to None to auto-intersect available years.
YEARS = [1985]

# Manually override storage zones if needed, e.g. ["A"]. Leave as None to infer from seasonal storage outputs.
STORAGE_ZONES = None

output_root = Path("acorn-julia/runs/low_RE_mod_elec_iter0/outputs/historical_1980_2019")
if not output_root.exists():
    output_root = Path.cwd().parent / "outputs" / "historical_1980_2019"


def select_existing(candidates, root):
    return next((name for name in candidates if (root / name).exists()), None)


baseline_label, baseline_candidates = BASELINE_SPEC
BASELINE_RUN = select_existing(baseline_candidates, output_root)
if BASELINE_RUN is None:
    raise FileNotFoundError(f"Could not find a baseline run from: {baseline_candidates}")

RUNS = [BASELINE_RUN]
LABELS = {BASELINE_RUN: baseline_label}
MISSING_RUNS = []

for label, candidates in RUN_SPECS:
    selected = select_existing(candidates, output_root)
    if selected is None:
        MISSING_RUNS.append((label, candidates))
        continue
    RUNS.append(selected)
    LABELS[selected] = label

print("Baseline run:", BASELINE_RUN)
print("Comparison runs:", RUNS[1:])
print("Labels:", LABELS)
print("Missing requested scenarios:", MISSING_RUNS)
print("Years:", YEARS)
print("output_root:", output_root)
"""
    ),
    code(
        """# --- Helpers ----------------------------------------------------------------

def _strip_tz(idx):
    if hasattr(idx, "tz") and idx.tz is not None:
        return idx.tz_convert(None)
    return idx


def _empty_tidy(value_name: str, extra_cols=None) -> pd.DataFrame:
    extra_cols = extra_cols or []
    data = {c: pd.Series(dtype=float) for c in extra_cols + [value_name]}
    data["timestamp"] = pd.to_datetime([])
    return pd.DataFrame(data)


def tidy_storage_df(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    meta_cols = [c for c in ["bus_id", "asset_type", "zone", "is_seasonal"] if c in df.columns]
    value_cols = [c for c in df.columns if c not in meta_cols]
    tidy = df.melt(id_vars=meta_cols, value_vars=value_cols, var_name="timestamp", value_name=value_name)
    tidy["timestamp"] = pd.to_datetime(tidy["timestamp"], errors="coerce")
    tidy = tidy.dropna(subset=["timestamp"])
    tidy[value_name] = pd.to_numeric(tidy[value_name], errors="coerce").fillna(0.0)
    return tidy


def tidy_storage_path(path: Path, value_name: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_tidy(value_name)
    return tidy_storage_df(pd.read_csv(path), value_name)


def tidy_bus_df(path: Path, value_name: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_tidy(value_name, extra_cols=["bus_id"])
    df = pd.read_csv(path)
    meta_cols = [c for c in ["bus_id", "zone"] if c in df.columns]
    value_cols = [c for c in df.columns if c not in meta_cols]
    tidy = df.melt(id_vars=meta_cols, value_vars=value_cols, var_name="timestamp", value_name=value_name)
    tidy["timestamp"] = pd.to_datetime(tidy["timestamp"], errors="coerce")
    tidy = tidy.dropna(subset=["timestamp"])
    tidy[value_name] = pd.to_numeric(tidy[value_name], errors="coerce").fillna(0.0)
    return tidy


def total_ts(df: pd.DataFrame, value_name: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df.groupby("timestamp")[value_name].sum()


def load_seasonal_soc(run_dir: Path, year: int) -> pd.Series:
    p = run_dir / f"storage_state_seasonal_{year}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    if df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in df.columns if c != "zone"]
    df = df[cols]
    if not df.empty and df.iloc[0, 0] == "bus_id":
        df = df.iloc[1:]
    value_cols = [c for c in df.columns if c != "bus_id"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    soc = df[value_cols].sum(axis=0)
    soc.index = pd.to_datetime(soc.index, errors="coerce")
    soc = soc.dropna()
    if hasattr(soc.index, "tz") and soc.index.tz is not None:
        soc.index = soc.index.tz_convert(None)
    return soc


def safe_sum(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).sum()


def detect_years(run_dir: Path):
    years = set()
    patterns = ["charge_base_*.csv", "load_shedding_*.csv", "charge_seasonal_*.csv"]
    for pat in patterns:
        for p in run_dir.glob(pat):
            try:
                years.add(int(p.stem.split("_")[-1]))
            except ValueError:
                pass
    return sorted(years)


def load_run_year(run_dir: Path, year: int) -> dict:
    return {
        "charge_base": tidy_storage_path(run_dir / f"charge_base_{year}.csv", "charge"),
        "discharge_base": tidy_storage_path(run_dir / f"discharge_base_{year}.csv", "discharge"),
        "charge_seasonal": tidy_storage_path(run_dir / f"charge_seasonal_{year}.csv", "charge"),
        "discharge_seasonal": tidy_storage_path(run_dir / f"discharge_seasonal_{year}.csv", "discharge"),
        "load_shed": tidy_bus_df(run_dir / f"load_shedding_{year}.csv", "load_shedding"),
        "wind_curt": tidy_bus_df(run_dir / f"wind_curtailment_{year}.csv", "wind_curtailment"),
        "solar_curt": tidy_bus_df(run_dir / f"solar_curtailment_{year}.csv", "solar_curtailment"),
        "soc_seasonal": load_seasonal_soc(run_dir, year),
    }


def concat_tidy(dfs, value_name: str, extra_cols=None) -> pd.DataFrame:
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return _empty_tidy(value_name, extra_cols=extra_cols)
    return pd.concat(dfs, ignore_index=True)


def concat_series(series_list) -> pd.Series:
    series_list = [s for s in series_list if s is not None and len(s) > 0]
    if not series_list:
        return pd.Series(dtype=float)
    return pd.concat(series_list).sort_index()


def infer_storage_zones(run_data: dict, seasonal_only: bool = True):
    zones = set()
    for run, data in run_data.items():
        if LABELS.get(run) == "baseline":
            continue
        seasonal_dfs = (data.get("charge_seasonal"), data.get("discharge_seasonal"))
        base_dfs = (data.get("charge_base"), data.get("discharge_base"))
        candidate_dfs = seasonal_dfs if seasonal_only else seasonal_dfs + base_dfs
        for df in candidate_dfs:
            if df is not None and not df.empty and "zone" in df.columns:
                zones |= set(df["zone"].dropna().unique())
    if not zones:
        for run, data in run_data.items():
            if LABELS.get(run) == "baseline":
                continue
            for df in (data.get("charge_base"), data.get("discharge_base")):
                if df is not None and not df.empty and "zone" in df.columns:
                    zones |= set(df["zone"].dropna().unique())
    return sorted(zones)


def zone_curtailment_ts(df: pd.DataFrame, value_col: str, zones) -> pd.Series:
    if df is None or df.empty or "zone" not in df.columns:
        return pd.Series(dtype=float)
    out = df[df["zone"].isin(zones)].groupby("timestamp")[value_col].sum()
    if len(out) > 0:
        out.index = _strip_tz(out.index)
    return out.sort_index()
"""
    ),
    code(
        """# --- Load all runs / years -------------------------------------------------
run_years = {}
for run in RUNS:
    run_dir = output_root / run
    run_years[run] = detect_years(run_dir)

if YEARS is None:
    if run_years:
        years = set(run_years[RUNS[0]])
        for r in RUNS[1:]:
            years &= set(run_years.get(r, []))
        YEARS = sorted(years)
    else:
        YEARS = []
else:
    YEARS = list(YEARS)

print("Detected years per run:", run_years)
print("Using YEARS:", YEARS)

run_data_by_year = {}
run_data = {}

for run in RUNS:
    run_dir = output_root / run
    yearly = {}
    for year in YEARS:
        yearly[year] = load_run_year(run_dir, year)
    run_data_by_year[run] = yearly

    run_data[run] = {
        "charge_base": concat_tidy([yearly[y]["charge_base"] for y in YEARS], "charge"),
        "discharge_base": concat_tidy([yearly[y]["discharge_base"] for y in YEARS], "discharge"),
        "charge_seasonal": concat_tidy([yearly[y]["charge_seasonal"] for y in YEARS], "charge"),
        "discharge_seasonal": concat_tidy([yearly[y]["discharge_seasonal"] for y in YEARS], "discharge"),
        "load_shed": concat_tidy([yearly[y]["load_shed"] for y in YEARS], "load_shedding", extra_cols=["bus_id"]),
        "wind_curt": concat_tidy([yearly[y]["wind_curt"] for y in YEARS], "wind_curtailment", extra_cols=["bus_id"]),
        "solar_curt": concat_tidy([yearly[y]["solar_curt"] for y in YEARS], "solar_curtailment", extra_cols=["bus_id"]),
        "soc_seasonal": concat_series([yearly[y]["soc_seasonal"] for y in YEARS]),
    }

zones = STORAGE_ZONES if STORAGE_ZONES is not None else infer_storage_zones(run_data)
print("Using storage zones:", zones)
if not zones:
    raise RuntimeError("No storage zones detected. Set STORAGE_ZONES manually.")
"""
    ),
    md("## A) Annual curtailment in storage zones and reduction vs baseline"),
    code(
        """rows = []
wind_daily = {}
solar_daily = {}
wind_monthly = {}
solar_monthly = {}

for run, data in run_data.items():
    label = LABELS.get(run, run)
    wind_ts = zone_curtailment_ts(data.get("wind_curt"), "wind_curtailment", zones)
    solar_ts = zone_curtailment_ts(data.get("solar_curt"), "solar_curtailment", zones)

    wind_daily[label] = wind_ts.resample("D").sum() if not wind_ts.empty else wind_ts
    solar_daily[label] = solar_ts.resample("D").sum() if not solar_ts.empty else solar_ts
    wind_monthly[label] = wind_ts.resample("ME").sum() if not wind_ts.empty else wind_ts
    solar_monthly[label] = solar_ts.resample("ME").sum() if not solar_ts.empty else solar_ts

    rows.append({
        "run": label,
        "wind_curt_MWh_in_storage_zones": float(safe_sum(wind_ts)),
        "solar_curt_MWh_in_storage_zones": float(safe_sum(solar_ts)),
    })

zone_curt_df = pd.DataFrame(rows)
zone_curt_df["total_curt_MWh_in_storage_zones"] = (
    zone_curt_df["wind_curt_MWh_in_storage_zones"] + zone_curt_df["solar_curt_MWh_in_storage_zones"]
)

baseline_row = zone_curt_df.loc[zone_curt_df["run"] == "baseline"]
if baseline_row.empty:
    raise RuntimeError("Baseline row not found in curtailment summary.")
base_wind = float(baseline_row["wind_curt_MWh_in_storage_zones"].iloc[0])
base_solar = float(baseline_row["solar_curt_MWh_in_storage_zones"].iloc[0])
base_total = float(baseline_row["total_curt_MWh_in_storage_zones"].iloc[0])

zone_curt_df["wind_reduction_MWh_vs_baseline"] = base_wind - zone_curt_df["wind_curt_MWh_in_storage_zones"]
zone_curt_df["solar_reduction_MWh_vs_baseline"] = base_solar - zone_curt_df["solar_curt_MWh_in_storage_zones"]
zone_curt_df["total_reduction_MWh_vs_baseline"] = base_total - zone_curt_df["total_curt_MWh_in_storage_zones"]
zone_curt_df["wind_reduction_pct_vs_baseline"] = np.where(
    base_wind != 0, 100 * zone_curt_df["wind_reduction_MWh_vs_baseline"] / base_wind, np.nan
)
zone_curt_df["solar_reduction_pct_vs_baseline"] = np.where(
    base_solar != 0, 100 * zone_curt_df["solar_reduction_MWh_vs_baseline"] / base_solar, np.nan
)
zone_curt_df["total_reduction_pct_vs_baseline"] = np.where(
    base_total != 0, 100 * zone_curt_df["total_reduction_MWh_vs_baseline"] / base_total, np.nan
)

order = ["baseline"] + [label for label, _ in RUN_SPECS if label in zone_curt_df["run"].tolist()]
zone_curt_df["run"] = pd.Categorical(zone_curt_df["run"], categories=order, ordered=True)
zone_curt_df = zone_curt_df.sort_values("run").reset_index(drop=True)

for c in zone_curt_df.columns[1:]:
    zone_curt_df[c] = zone_curt_df[c].round(2)

print(f"Storage zones used for comparison: {zones}")
print(f"Baseline run: {BASELINE_RUN}")
display(zone_curt_df)
"""
    ),
    code(
        """plot_df = zone_curt_df[zone_curt_df["run"] != "baseline"].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
plot_df.plot.bar(x="run", y="wind_reduction_MWh_vs_baseline", legend=False, ax=axes[0], color="#4c78a8")
axes[0].set_title("Wind curtailment reduction\\nin storage zones")
axes[0].set_ylabel("MWh vs baseline")

plot_df.plot.bar(x="run", y="solar_reduction_MWh_vs_baseline", legend=False, ax=axes[1], color="#f58518")
axes[1].set_title("Solar curtailment reduction\\nin storage zones")
axes[1].set_ylabel("MWh vs baseline")

plot_df.plot.bar(x="run", y="total_reduction_MWh_vs_baseline", legend=False, ax=axes[2], color="#54a24b")
axes[2].set_title("Total curtailment reduction\\nin storage zones")
axes[2].set_ylabel("MWh vs baseline")

for ax in axes:
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    md("## B) Daily curtailment profiles in storage zones"),
    code(
        """wind_daily_df = pd.DataFrame(wind_daily).sort_index()
solar_daily_df = pd.DataFrame(solar_daily).sort_index()

plot_wind = wind_daily_df.dropna(how="all", axis=1)
if not plot_wind.empty:
    ax = plot_wind.plot(figsize=(12, 4), title=f"Daily wind curtailment in storage zones ({min(YEARS)}-{max(YEARS)})")
    ax.set_ylabel("MWh/day")
    plt.tight_layout()
    plt.show()

plot_solar = solar_daily_df.dropna(how="all", axis=1)
if not plot_solar.empty:
    ax = plot_solar.plot(figsize=(12, 4), title=f"Daily solar curtailment in storage zones ({min(YEARS)}-{max(YEARS)})")
    ax.set_ylabel("MWh/day")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        """baseline_wind_daily = wind_daily_df.get("baseline", pd.Series(dtype=float))
baseline_solar_daily = solar_daily_df.get("baseline", pd.Series(dtype=float))

wind_reduction_daily = pd.DataFrame(index=wind_daily_df.index)
solar_reduction_daily = pd.DataFrame(index=solar_daily_df.index)

for col in wind_daily_df.columns:
    if col == "baseline":
        continue
    wind_reduction_daily[col] = baseline_wind_daily.reindex(wind_daily_df.index).fillna(0) - wind_daily_df[col].fillna(0)

for col in solar_daily_df.columns:
    if col == "baseline":
        continue
    solar_reduction_daily[col] = baseline_solar_daily.reindex(solar_daily_df.index).fillna(0) - solar_daily_df[col].fillna(0)

if not wind_reduction_daily.empty:
    ax = wind_reduction_daily.plot(figsize=(12, 4), title="Daily wind-curtailment reduction vs baseline in storage zones")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MWh/day")
    plt.tight_layout()
    plt.show()

if not solar_reduction_daily.empty:
    ax = solar_reduction_daily.plot(figsize=(12, 4), title="Daily solar-curtailment reduction vs baseline in storage zones")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MWh/day")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## C) Monthly reduction pattern by lambda"),
    code(
        """wind_monthly_df = pd.DataFrame(wind_monthly).sort_index()
solar_monthly_df = pd.DataFrame(solar_monthly).sort_index()

wind_monthly_reduction = pd.DataFrame(index=wind_monthly_df.index)
solar_monthly_reduction = pd.DataFrame(index=solar_monthly_df.index)

for col in wind_monthly_df.columns:
    if col == "baseline":
        continue
    wind_monthly_reduction[col] = wind_monthly_df["baseline"].reindex(wind_monthly_df.index).fillna(0) - wind_monthly_df[col].fillna(0)

for col in solar_monthly_df.columns:
    if col == "baseline":
        continue
    solar_monthly_reduction[col] = solar_monthly_df["baseline"].reindex(solar_monthly_df.index).fillna(0) - solar_monthly_df[col].fillna(0)

if not wind_monthly_reduction.empty:
    heat = wind_monthly_reduction.copy().T
    heat.columns = [ts.strftime("%Y-%m") for ts in heat.columns]
    plt.figure(figsize=(12, 4))
    sns.heatmap(heat, cmap="RdYlGn", center=0)
    plt.title("Monthly wind-curtailment reduction vs baseline in storage zones")
    plt.xlabel("Month")
    plt.ylabel("Lambda run")
    plt.tight_layout()
    plt.show()

if not solar_monthly_reduction.empty:
    heat = solar_monthly_reduction.copy().T
    heat.columns = [ts.strftime("%Y-%m") for ts in heat.columns]
    plt.figure(figsize=(12, 4))
    sns.heatmap(heat, cmap="RdYlGn", center=0)
    plt.title("Monthly solar-curtailment reduction vs baseline in storage zones")
    plt.xlabel("Month")
    plt.ylabel("Lambda run")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## D) Link zone-curtailment reduction to seasonal-storage activity"),
    code(
        """activity_rows = []
for run, data in run_data.items():
    label = LABELS.get(run, run)
    seas_ch = total_ts(data["charge_seasonal"], "charge")
    seas_dis = total_ts(data["discharge_seasonal"], "discharge")
    active_index = seas_ch.index.union(seas_dis.index)
    active_days = 0
    if len(active_index):
        charge_active = seas_ch.reindex(active_index, fill_value=0).fillna(0) > 0
        discharge_active = seas_dis.reindex(active_index, fill_value=0).fillna(0) > 0
        active_days = int((charge_active | discharge_active).sum())

    activity_rows.append({
        "run": label,
        "seasonal_charge_MWh": float(safe_sum(seas_ch)),
        "seasonal_discharge_MWh": float(safe_sum(seas_dis)),
        "seasonal_active_days": active_days,
    })

activity_df = pd.DataFrame(activity_rows)
merged = zone_curt_df.merge(activity_df, on="run", how="left")
for c in merged.columns[1:]:
    if merged[c].dtype.kind in "fc":
        merged[c] = merged[c].round(2)

display(merged)
"""
    ),
    code(
        """plot_df = merged[merged["run"] != "baseline"].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(plot_df["seasonal_charge_MWh"], plot_df["total_reduction_MWh_vs_baseline"], s=80)
for _, row in plot_df.iterrows():
    axes[0].annotate(row["run"], (row["seasonal_charge_MWh"], row["total_reduction_MWh_vs_baseline"]), fontsize=9)
axes[0].set_xlabel("Seasonal charge (MWh)")
axes[0].set_ylabel("Total curtailment reduction in storage zones (MWh)")
axes[0].set_title("More seasonal charging vs zone-curtailment reduction")

axes[1].scatter(plot_df["seasonal_discharge_MWh"], plot_df["total_reduction_MWh_vs_baseline"], s=80, color="#e45756")
for _, row in plot_df.iterrows():
    axes[1].annotate(row["run"], (row["seasonal_discharge_MWh"], row["total_reduction_MWh_vs_baseline"]), fontsize=9)
axes[1].set_xlabel("Seasonal discharge (MWh)")
axes[1].set_ylabel("Total curtailment reduction in storage zones (MWh)")
axes[1].set_title("More seasonal discharge vs zone-curtailment reduction")

plt.tight_layout()
plt.show()
"""
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT}")
