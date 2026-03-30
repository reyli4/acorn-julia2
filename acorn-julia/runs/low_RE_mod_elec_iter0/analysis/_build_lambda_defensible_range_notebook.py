import json
from pathlib import Path


OUT_PATH = Path(
    "/home/fs01/jl2966/acorn-julia2/acorn-julia/runs/low_RE_mod_elec_iter0/analysis/"
    "seasonal_stage1_lambda_defensible_range.ipynb"
)


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.append(
    md_cell(
        """# Turning seasonal-storage sensitivity tests into a defensible lambda range

This notebook organizes the evidence chain for lambda selection using three layers:

1. one-year sweep (`1985`) to identify phase changes rather than averaging across incompatible regimes
2. explicit operating tests for "what kind of technology does this look like?"
3. wrapped multi-year validation to check cross-year carryover and inventory sustainability

This is intended to support a supervisor meeting or methods write-up. The goal is not to choose lambda from a simple average. The goal is to identify a stable operating band that:

- avoids battery-like overuse
- still charges and discharges seasonally
- survives wrapped multi-year carryover
- retains system value in the years that actually need seasonal storage
"""
    )
)

cells.append(
    code_cell(
        """from pathlib import Path
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

OUTPUT_ROOT = Path("acorn-julia/runs/low_RE_mod_elec_iter0/outputs/historical_1980_2019")
if not OUTPUT_ROOT.exists():
    OUTPUT_ROOT = Path.cwd().parent / "outputs" / "historical_1980_2019"

# --- 1985 single-year sweep -------------------------------------------------
SWEEP_RUNS = {
    "3month_lamba_2_seasonal": "lambda2",
    "3month_lamba_3_seasonal": "lambda3",
    "3month_lamba_3.25_seasonal": "lambda3.25",
    "3month_lamba_3.3_seasonal": "lambda3.3",
    "3month_lamba_3.4_seasonal": "lambda3.4",
    "3month_lamba_3.43_seasonal": "lambda3.43",
    "3month_lamba_3.45_seasonal": "lambda3.45",
    "3month_lamba_3.47_seasonal": "lambda3.47",
    "3month_lamba_3.5_seasonal": "lambda3.5",
    "3month_lamba_3.75_seasonal": "lambda3.75",
    "3month_lamba_4_seasonal": "lambda4",
    "3month_lamba_5_seasonal": "lambda5",
}
SWEEP_YEAR = 1985

# --- Wrapped validation runs ------------------------------------------------
WRAPPED_COMMON_RUNS = {
    "3month_lambda_3.5_wrapped_1985_1988": "lambda3.5",
    "3month_lambda_3.55_wrapped_1985_1988": "lambda3.55",
    "stagel_3month_wrapped_1985_1988": "lambda5",
}
WRAPPED_EXTENDED_RUN = {
    "3month_lambda_3.5_wrapped_1985_1990": "lambda3.5_1985_1990",
}

# --- Metrics / thresholds ---------------------------------------------------
# Edit these thresholds if you want a stricter or looser technology screen.
SCARCITY_MONTHS = [1, 6, 7, 8, 12]
SURPLUS_MONTHS = [3, 4, 5, 6]
MAX_EQ_CYCLES_FOR_CANDIDATE = 2.0
MAX_ACTIVE_DISCHARGE_DAYS_FOR_CANDIDATE = 120
MIN_SCARCITY_DISCHARGE_SHARE = 0.45
MAX_SIMULT_HOURS_FOR_CANDIDATE = 24
MAX_NEGATIVE_SOC_DRIFT_MWH = 5e5

print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("Sweep runs:", SWEEP_RUNS)
print("Wrapped common runs:", WRAPPED_COMMON_RUNS)
print("Wrapped extended run:", WRAPPED_EXTENDED_RUN)
"""
    )
)

cells.append(
    code_cell(
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
    tidy = df.melt(
        id_vars=meta_cols,
        value_vars=value_cols,
        var_name="timestamp",
        value_name=value_name,
    )
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
    tidy = df.melt(
        id_vars=meta_cols,
        value_vars=value_cols,
        var_name="timestamp",
        value_name=value_name,
    )
    tidy["timestamp"] = pd.to_datetime(tidy["timestamp"], errors="coerce")
    tidy = tidy.dropna(subset=["timestamp"])
    tidy[value_name] = pd.to_numeric(tidy[value_name], errors="coerce").fillna(0.0)
    return tidy


def total_ts(df: pd.DataFrame, value_name: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df.groupby("timestamp")[value_name].sum()
    s.index = _strip_tz(s.index)
    return s


def load_seasonal_soc_path(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    if df.empty:
        return pd.Series(dtype=float)
    cols = [c for c in df.columns if c not in ("zone",)]
    df = df[cols]
    if not df.empty and df.iloc[0, 0] == "bus_id":
        df = df.iloc[1:]
    value_cols = [c for c in df.columns if c != "bus_id"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    soc = df[value_cols].sum(axis=0)
    soc.index = pd.to_datetime(soc.index, errors="coerce")
    soc = soc.dropna()
    soc.index = _strip_tz(soc.index)
    return soc


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "keep_for_analysis" in df.columns:
        keep = pd.to_numeric(df["keep_for_analysis"], errors="coerce").fillna(0).astype(int)
        df = df[keep == 1].copy()
    df["seq_idx"] = pd.to_numeric(df["seq_idx"], errors="coerce").astype(int)
    df["sim_year"] = pd.to_numeric(df["sim_year"], errors="coerce").astype(int)
    return df.sort_values("seq_idx").reset_index(drop=True)


def daily_sum(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    s = series.copy()
    s.index = _strip_tz(s.index)
    return s.resample("D").sum()


def daily_mean(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    s = series.copy()
    s.index = _strip_tz(s.index)
    return s.resample("D").mean()


def monthly_sum(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    s = series.copy()
    s.index = _strip_tz(s.index)
    return s.resample("ME").sum()


def safe_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def load_single_year_run(run_name: str, year: int) -> dict:
    run_dir = OUTPUT_ROOT / run_name
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "charge_base": tidy_storage_path(run_dir / f"charge_base_{year}.csv", "charge"),
        "discharge_base": tidy_storage_path(run_dir / f"discharge_base_{year}.csv", "discharge"),
        "charge_seasonal": tidy_storage_path(run_dir / f"charge_seasonal_{year}.csv", "charge"),
        "discharge_seasonal": tidy_storage_path(run_dir / f"discharge_seasonal_{year}.csv", "discharge"),
        "load_shed": tidy_bus_df(run_dir / f"load_shedding_{year}.csv", "load_shedding"),
        "wind_curt": tidy_bus_df(run_dir / f"wind_curtailment_{year}.csv", "wind_curtailment"),
        "solar_curt": tidy_bus_df(run_dir / f"solar_curtailment_{year}.csv", "solar_curtailment"),
        "soc_seasonal": load_seasonal_soc_path(run_dir / f"storage_state_seasonal_{year}.csv"),
    }


def load_wrapped_run(run_name: str, label: str) -> dict:
    run_dir = OUTPUT_ROOT / run_name
    manifest = load_manifest(run_dir / "_sequence_manifest.csv")
    yearly = {}
    for _, row in manifest.iterrows():
        year = int(row["sim_year"])
        token = row["token"]
        yearly[year] = {
            "charge_base": tidy_storage_path(run_dir / f"charge_base_{token}.csv", "charge"),
            "discharge_base": tidy_storage_path(run_dir / f"discharge_base_{token}.csv", "discharge"),
            "charge_seasonal": tidy_storage_path(run_dir / f"charge_seasonal_{token}.csv", "charge"),
            "discharge_seasonal": tidy_storage_path(run_dir / f"discharge_seasonal_{token}.csv", "discharge"),
            "load_shed": tidy_bus_df(run_dir / f"load_shedding_{token}.csv", "load_shedding"),
            "wind_curt": tidy_bus_df(run_dir / f"wind_curtailment_{token}.csv", "wind_curtailment"),
            "solar_curt": tidy_bus_df(run_dir / f"solar_curtailment_{token}.csv", "solar_curtailment"),
            "soc_seasonal": load_seasonal_soc_path(run_dir / f"storage_state_seasonal_{token}.csv"),
        }
    return {
        "run_name": run_name,
        "label": label,
        "run_dir": run_dir,
        "manifest": manifest,
        "yearly": yearly,
    }


def summarize_run(label: str, data: dict) -> dict:
    seas_ch = total_ts(data["charge_seasonal"], "charge")
    seas_dis = total_ts(data["discharge_seasonal"], "discharge")
    base_dis = total_ts(data["discharge_base"], "discharge")
    base_ch = total_ts(data["charge_base"], "charge")
    load_shed = total_ts(data["load_shed"], "load_shedding")
    wind = total_ts(data["wind_curt"], "wind_curtailment")
    solar = total_ts(data["solar_curt"], "solar_curtailment")
    soc = data["soc_seasonal"]
    ch_d = daily_sum(seas_ch)
    dis_d = daily_sum(seas_dis)
    aligned = pd.concat([seas_ch.rename("charge"), seas_dis.rename("discharge")], axis=1).fillna(0.0)
    monthly_ch = monthly_sum(seas_ch)
    monthly_dis = monthly_sum(seas_dis)
    scarcity_discharge = monthly_dis[monthly_dis.index.month.isin(SCARCITY_MONTHS)].sum() if len(monthly_dis) else 0.0
    surplus_charge = monthly_ch[monthly_ch.index.month.isin(SURPLUS_MONTHS)].sum() if len(monthly_ch) else 0.0
    soc_span = float(soc.max() - soc.min()) if len(soc) else np.nan
    eq_cycles = safe_sum(seas_dis) / soc_span if pd.notna(soc_span) and soc_span > 0 else np.nan
    return {
        "run": label,
        "base_charge_MWh": safe_sum(base_ch),
        "base_discharge_MWh": safe_sum(base_dis),
        "seasonal_charge_MWh": safe_sum(seas_ch),
        "seasonal_discharge_MWh": safe_sum(seas_dis),
        "load_shed_MWh": safe_sum(load_shed),
        "wind_curt_MWh": safe_sum(wind),
        "solar_curt_MWh": safe_sum(solar),
        "soc_start_MWh": float(soc.iloc[0]) if len(soc) else np.nan,
        "soc_end_MWh": float(soc.iloc[-1]) if len(soc) else np.nan,
        "soc_max_MWh": float(soc.max()) if len(soc) else np.nan,
        "soc_min_MWh": float(soc.min()) if len(soc) else np.nan,
        "soc_span_MWh": soc_span,
        "eq_cycles_per_year": eq_cycles,
        "discharge_to_charge_ratio": safe_sum(seas_dis) / safe_sum(seas_ch) if safe_sum(seas_ch) > 0 else np.nan,
        "active_charge_days": int((ch_d > 1e-6).sum()) if len(ch_d) else 0,
        "active_discharge_days": int((dis_d > 1e-6).sum()) if len(dis_d) else 0,
        "simultaneous_hours": int(((aligned["charge"] > 1e-6) & (aligned["discharge"] > 1e-6)).sum()) if not aligned.empty else 0,
        "scarcity_discharge_share": scarcity_discharge / safe_sum(seas_dis) if safe_sum(seas_dis) > 0 else np.nan,
        "surplus_charge_share": surplus_charge / safe_sum(seas_ch) if safe_sum(seas_ch) > 0 else np.nan,
        "soc_drift_MWh": float(soc.iloc[-1] - soc.iloc[0]) if len(soc) else np.nan,
    }


def classify_sweep_row(row: pd.Series) -> str:
    if pd.notna(row["eq_cycles_per_year"]) and row["eq_cycles_per_year"] > MAX_EQ_CYCLES_FOR_CANDIDATE:
        return "overused / battery-like"
    if row["active_discharge_days"] > MAX_ACTIVE_DISCHARGE_DAYS_FOR_CANDIDATE:
        return "overused / battery-like"
    if row["simultaneous_hours"] > MAX_SIMULT_HOURS_FOR_CANDIDATE:
        return "overused / battery-like"
    if row["seasonal_charge_MWh"] <= 0 and row["seasonal_discharge_MWh"] > 0:
        return "inventory drawdown / too expensive"
    if pd.notna(row["scarcity_discharge_share"]) and row["scarcity_discharge_share"] < MIN_SCARCITY_DISCHARGE_SHARE:
        return "weak seasonal alignment"
    return "candidate seasonal regime"
"""
    )
)

cells.append(
    md_cell(
        """## A) One-year sweep: find phase changes, not averages

The point of the `1985` sweep is to detect regime changes. If lambda changes cause a jump from one corner solution to another, averaging all lambdas together is not meaningful.
"""
    )
)

cells.append(
    code_cell(
        """# Load and summarize the 1985 one-year sweep
sweep_data = {label: load_single_year_run(run_name, SWEEP_YEAR) for run_name, label in SWEEP_RUNS.items()}
sweep_rows = [summarize_run(label, data) for label, data in sweep_data.items()]
sweep_df = pd.DataFrame(sweep_rows)
sweep_df["lambda"] = sweep_df["run"].str.replace("lambda", "", regex=False).astype(float)
sweep_df["regime"] = sweep_df.apply(classify_sweep_row, axis=1)
sweep_df = sweep_df.sort_values("lambda").reset_index(drop=True)

display_cols = [
    "run", "lambda",
    "seasonal_charge_MWh", "seasonal_discharge_MWh", "load_shed_MWh",
    "eq_cycles_per_year", "discharge_to_charge_ratio",
    "active_charge_days", "active_discharge_days",
    "scarcity_discharge_share", "surplus_charge_share",
    "soc_drift_MWh", "regime",
]

for c in [c for c in display_cols if c in sweep_df.columns and c not in ["run", "regime"]]:
    sweep_df[c] = sweep_df[c].round(3)

print("1985 seasonal penalty sweep summary:")
display(sweep_df[display_cols])
"""
    )
)

cells.append(
    code_cell(
        """# Regime plots across lambda
plot_metrics = [
    ("seasonal_charge_MWh", "Seasonal charge"),
    ("seasonal_discharge_MWh", "Seasonal discharge"),
    ("load_shed_MWh", "Load shedding"),
    ("eq_cycles_per_year", "Equivalent cycles per year"),
    ("active_discharge_days", "Active discharge days"),
]

fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(12, 16), sharex=True)
for ax, (metric, title) in zip(axes, plot_metrics):
    ax.plot(sweep_df["lambda"], sweep_df[metric], marker="o")
    ax.axvspan(3.4, 3.5, color="tab:red", alpha=0.08, label="suspected phase-change region")
    ax.set_ylabel(metric)
    ax.set_title(title)
axes[-1].set_xlabel("Seasonal penalty multiplier lambda")
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code_cell(
        """# Monthly discharge/charge structure for the sweep candidates near the transition
FOCUS_LAMBDAS = ["lambda3.4", "lambda3.43", "lambda3.45", "lambda3.47", "lambda3.5", "lambda5"]

monthly_rows = []
for label in FOCUS_LAMBDAS:
    if label not in sweep_data:
        continue
    data = sweep_data[label]
    seas_ch = monthly_sum(total_ts(data["charge_seasonal"], "charge"))
    seas_dis = monthly_sum(total_ts(data["discharge_seasonal"], "discharge"))
    ls = monthly_sum(total_ts(data["load_shed"], "load_shedding"))
    for ts in sorted(set(seas_ch.index).union(seas_dis.index).union(ls.index)):
        monthly_rows.append({
            "run": label,
            "month": ts.month,
            "month_name": calendar.month_abbr[ts.month],
            "seasonal_charge_MWh": float(seas_ch.get(ts, 0.0)),
            "seasonal_discharge_MWh": float(seas_dis.get(ts, 0.0)),
            "load_shed_MWh": float(ls.get(ts, 0.0)),
        })

monthly_focus_df = pd.DataFrame(monthly_rows)
if not monthly_focus_df.empty:
    for metric in ["seasonal_charge_MWh", "seasonal_discharge_MWh", "load_shed_MWh"]:
        pivot = monthly_focus_df.pivot(index="run", columns="month_name", values=metric)
        order = [calendar.month_abbr[m] for m in range(1, 13)]
        pivot = pivot[[m for m in order if m in pivot.columns]]
        plt.figure(figsize=(14, 4))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title(f"{metric} in 1985 near the lambda transition")
        plt.tight_layout()
        plt.show()
"""
    )
)

cells.append(
    md_cell(
        """## B) Turn "what technology does this look like?" into hard operating tests

This section converts the supervisor question into explicit screening criteria:

- equivalent cycles per year
- concentration of discharge in scarcity months
- frequency of active discharge days
- simultaneous charge/discharge hours
- SOC drift

These are not perfect technology models. They are screening metrics that make lambda selection defendable.
"""
    )
)

cells.append(
    code_cell(
        """# Scorecard against the explicit operating tests
scorecard = sweep_df[[
    "run", "lambda", "eq_cycles_per_year", "active_discharge_days",
    "simultaneous_hours", "scarcity_discharge_share", "soc_drift_MWh", "regime"
]].copy()

scorecard["eq_cycles_ok"] = scorecard["eq_cycles_per_year"] <= MAX_EQ_CYCLES_FOR_CANDIDATE
scorecard["active_days_ok"] = scorecard["active_discharge_days"] <= MAX_ACTIVE_DISCHARGE_DAYS_FOR_CANDIDATE
scorecard["simult_ok"] = scorecard["simultaneous_hours"] <= MAX_SIMULT_HOURS_FOR_CANDIDATE
scorecard["scarcity_alignment_ok"] = scorecard["scarcity_discharge_share"] >= MIN_SCARCITY_DISCHARGE_SHARE
scorecard["inventory_drift_ok"] = scorecard["soc_drift_MWh"].abs() <= MAX_NEGATIVE_SOC_DRIFT_MWH
scorecard["n_tests_passed"] = scorecard[[
    "eq_cycles_ok", "active_days_ok", "simult_ok",
    "scarcity_alignment_ok", "inventory_drift_ok"
]].sum(axis=1)

display(scorecard.sort_values("lambda"))
"""
    )
)

cells.append(
    md_cell(
        """## C) Wrapped multi-year validation: stable band vs over-penalized band

The wrapped notebooks are the real test for cross-year carryover:

- `lambda3.5`
- `lambda3.55`
- `lambda5`

The question here is not just "does load shedding go down?" The question is whether the technology still forms a sustainable seasonal inventory once free boundary energy is removed.
"""
    )
)

cells.append(
    code_cell(
        """# Load wrapped common-year runs
wrapped_common = {label: load_wrapped_run(run_name, label) for run_name, label in WRAPPED_COMMON_RUNS.items()}
common_year_sets = [set(bundle["manifest"]["sim_year"].tolist()) for bundle in wrapped_common.values()]
COMMON_YEARS = sorted(set.intersection(*common_year_sets))

print("Common retained years:", COMMON_YEARS)
for label, bundle in wrapped_common.items():
    print("\\n", label)
    display(bundle["manifest"])
"""
    )
)

cells.append(
    code_cell(
        """# Common-year wrapped annual totals and activity
wrapped_rows = []
for label, bundle in wrapped_common.items():
    for year in COMMON_YEARS:
        row = summarize_run(label, bundle["yearly"][year])
        row["year"] = year
        wrapped_rows.append(row)

wrapped_df = pd.DataFrame(wrapped_rows)
wrapped_df = wrapped_df.sort_values(["year", "run"]).reset_index(drop=True)
for c in [c for c in wrapped_df.columns if c not in ["run"]]:
    wrapped_df[c] = wrapped_df[c].round(3)

display_cols = [
    "run", "year",
    "seasonal_charge_MWh", "seasonal_discharge_MWh", "load_shed_MWh",
    "eq_cycles_per_year", "active_discharge_days",
    "scarcity_discharge_share", "soc_start_MWh", "soc_end_MWh", "soc_drift_MWh",
]
display(wrapped_df[display_cols])

wrapped_stats = wrapped_df.groupby("run")[[
    "seasonal_charge_MWh", "seasonal_discharge_MWh", "load_shed_MWh",
    "eq_cycles_per_year", "scarcity_discharge_share", "soc_drift_MWh"
]].agg(["median", "min", "max"]).round(3)
display(wrapped_stats)
"""
    )
)

cells.append(
    code_cell(
        """# Percent changes against lambda3.5 in the wrapped common years
base = wrapped_df[wrapped_df["run"] == "lambda3.5"].set_index("year")
compare_metrics = ["seasonal_charge_MWh", "seasonal_discharge_MWh", "load_shed_MWh"]
rows = []
for run_label in sorted(wrapped_df["run"].unique()):
    if run_label == "lambda3.5":
        continue
    comp = wrapped_df[wrapped_df["run"] == run_label].set_index("year")
    for year in COMMON_YEARS:
        rec = {"run": run_label, "year": year}
        for metric in compare_metrics:
            b = base.loc[year, metric]
            v = comp.loc[year, metric]
            rec[f"{metric}_pct_vs_lambda3.5"] = np.nan if b == 0 else 100.0 * (v - b) / b
        rows.append(rec)

wrapped_pct = pd.DataFrame(rows)
for c in [c for c in wrapped_pct.columns if c not in ["run", "year"]]:
    wrapped_pct[c] = wrapped_pct[c].round(2)
display(wrapped_pct.sort_values(["year", "run"]))
"""
    )
)

cells.append(
    code_cell(
        """# Visual check: annual charge/discharge/load shed by wrapped run
metrics = [
    ("seasonal_charge_MWh", "Seasonal charge"),
    ("seasonal_discharge_MWh", "Seasonal discharge"),
    ("load_shed_MWh", "Load shedding"),
]

fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 12), sharex=True)
for ax, (metric, title) in zip(axes, metrics):
    sns.barplot(data=wrapped_df, x="year", y=metric, hue="run", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("MWh")
    ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code_cell(
        """# SOC and seasonal-only daily traces for the wrapped runs
for year in COMMON_YEARS:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for label, bundle in wrapped_common.items():
        data = bundle["yearly"][year]
        soc = daily_mean(data["soc_seasonal"])
        seas_ch = daily_sum(total_ts(data["charge_seasonal"], "charge"))
        seas_dis = daily_sum(total_ts(data["discharge_seasonal"], "discharge"))
        ls = daily_sum(total_ts(data["load_shed"], "load_shedding"))

        if not soc.empty:
            axes[0].plot(soc.index, soc.values, label=label)
        if not seas_ch.empty:
            axes[1].plot(seas_ch.index, seas_ch.values, label=f"{label} charge")
        if not seas_dis.empty:
            axes[1].plot(seas_dis.index, -seas_dis.values, linestyle="--", label=f"{label} discharge")
        if not ls.empty:
            axes[2].plot(ls.index, ls.values, label=label)

    axes[0].set_title(f"Daily mean seasonal SOC ({year})")
    axes[0].set_ylabel("MWh")
    axes[1].set_title(f"Seasonal-only daily charge/discharge ({year})")
    axes[1].set_ylabel("MWh/day")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title(f"Daily load shedding ({year})")
    axes[2].set_ylabel("MWh/day")
    axes[2].set_xlabel("Date")
    for ax in axes:
        ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.show()
"""
    )
)

cells.append(
    md_cell(
        """## D) Extended wrapped run: which years create the most seasonal-storage value?

This is where the analysis shifts from "pick lambda" to "explain when seasonal storage matters."

The value signal should be concentrated in stress years, not spread evenly across all years.
"""
    )
)

cells.append(
    code_cell(
        """# Load the extended wrapped run and rank years
extended_label = list(WRAPPED_EXTENDED_RUN.values())[0]
extended_bundle = load_wrapped_run(list(WRAPPED_EXTENDED_RUN.keys())[0], extended_label)
EXTENDED_YEARS = extended_bundle["manifest"]["sim_year"].tolist()

extended_rows = []
for year in EXTENDED_YEARS:
    row = summarize_run(extended_label, extended_bundle["yearly"][year])
    row["year"] = year
    extended_rows.append(row)

extended_df = pd.DataFrame(extended_rows).sort_values("year").reset_index(drop=True)
for c in [c for c in extended_df.columns if c not in ["run"]]:
    extended_df[c] = extended_df[c].round(3)

display(extended_df[[
    "year", "seasonal_charge_MWh", "seasonal_discharge_MWh", "load_shed_MWh",
    "wind_curt_MWh", "solar_curt_MWh", "eq_cycles_per_year", "soc_drift_MWh"
]])

rank_df = extended_df[[
    "year", "seasonal_discharge_MWh", "load_shed_MWh",
    "wind_curt_MWh", "solar_curt_MWh"
]].copy()
rank_df["discharge_rank"] = rank_df["seasonal_discharge_MWh"].rank(ascending=False, method="dense").astype(int)
rank_df["load_shed_rank"] = rank_df["load_shed_MWh"].rank(ascending=False, method="dense").astype(int)
rank_df = rank_df.sort_values(["discharge_rank", "year"]).reset_index(drop=True)
display(rank_df)
"""
    )
)

cells.append(
    code_cell(
        """# Monthly year-characteristics: when does seasonal storage charge and discharge?
monthly_rows = []
for year in EXTENDED_YEARS:
    data = extended_bundle["yearly"][year]
    seas_ch = monthly_sum(total_ts(data["charge_seasonal"], "charge"))
    seas_dis = monthly_sum(total_ts(data["discharge_seasonal"], "discharge"))
    ls = monthly_sum(total_ts(data["load_shed"], "load_shedding"))
    for ts in sorted(set(seas_ch.index).union(seas_dis.index).union(ls.index)):
        monthly_rows.append({
            "year": year,
            "month": ts.month,
            "month_name": calendar.month_abbr[ts.month],
            "seasonal_charge_MWh": float(seas_ch.get(ts, 0.0)),
            "seasonal_discharge_MWh": float(seas_dis.get(ts, 0.0)),
            "load_shed_MWh": float(ls.get(ts, 0.0)),
        })

monthly_ext_df = pd.DataFrame(monthly_rows)
display(monthly_ext_df.sort_values(["year", "month"]))

for metric in ["seasonal_charge_MWh", "seasonal_discharge_MWh", "load_shed_MWh"]:
    pivot = monthly_ext_df.pivot(index="year", columns="month_name", values=metric)
    order = [calendar.month_abbr[m] for m in range(1, 13)]
    pivot = pivot[[m for m in order if m in pivot.columns]]
    plt.figure(figsize=(14, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title(f"{metric} by year and month")
    plt.tight_layout()
    plt.show()
"""
    )
)

cells.append(
    code_cell(
        """# Daily traces for the highest-value years in the extended run
top_years = extended_df.sort_values(["seasonal_discharge_MWh", "load_shed_MWh"], ascending=[False, False])["year"].head(4).tolist()
print("Top years by seasonal discharge:", top_years)

for year in top_years:
    data = extended_bundle["yearly"][year]
    soc = daily_mean(data["soc_seasonal"])
    seas_ch = daily_sum(total_ts(data["charge_seasonal"], "charge"))
    seas_dis = daily_sum(total_ts(data["discharge_seasonal"], "discharge"))
    ls = daily_sum(total_ts(data["load_shed"], "load_shedding"))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(soc.index, soc.values, color="tab:blue")
    axes[0].set_title(f"Daily mean seasonal SOC ({year})")
    axes[0].set_ylabel("MWh")

    axes[1].plot(seas_ch.index, seas_ch.values, label="seasonal charge", color="tab:green")
    axes[1].plot(seas_dis.index, -seas_dis.values, label="seasonal discharge", color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title(f"Seasonal-only daily charge/discharge ({year})")
    axes[1].set_ylabel("MWh/day")
    axes[1].legend(loc="upper right")

    axes[2].plot(ls.index, ls.values, color="tab:red")
    axes[2].set_title(f"Daily load shedding ({year})")
    axes[2].set_ylabel("MWh/day")
    axes[2].set_xlabel("Date")
    plt.tight_layout()
    plt.show()
"""
    )
)

cells.append(
    md_cell(
        """## E) A defendable lambda-selection workflow

Use the notebook outputs in this sequence:

1. **One-year sweep:** identify the phase-change region and reject clearly battery-like or clearly over-penalized lambdas.
2. **Operating scorecard:** state explicit tests for "seasonal enough" behavior.
3. **Wrapped common-years validation:** show that the candidate band remains stable once free boundary energy is reduced.
4. **Extended wrapped run:** show which years and seasons actually create value.

The intended interpretation is:

- low lambda: too much throughput, too many active days, seasonal storage starts behaving like short-duration storage
- candidate band: stable wrapped behavior, plausible charge/discharge timing, no sustained inventory collapse
- high lambda: charging is suppressed, SOC drifts downward, and load shedding worsens
"""
    )
)

cells.append(
    code_cell(
        """# Optional synthesis table for a meeting slide or methods section
summary_rows = []

# Candidate sweep band from explicit tests
candidate_sweep = scorecard.loc[
    scorecard["regime"] == "candidate seasonal regime",
    ["run", "lambda", "n_tests_passed"]
].sort_values(["n_tests_passed", "lambda"], ascending=[False, True])

for _, row in candidate_sweep.iterrows():
    summary_rows.append({
        "stage": "1985 sweep",
        "candidate": row["run"],
        "evidence": f"passes {int(row['n_tests_passed'])} screen tests",
    })

# Wrapped interpretation
for run_label in ["lambda3.5", "lambda3.55", "lambda5"]:
    sub = wrapped_df[wrapped_df["run"] == run_label]
    summary_rows.append({
        "stage": "wrapped 1985-1988",
        "candidate": run_label,
        "evidence": (
            f"median seasonal charge={sub['seasonal_charge_MWh'].median():.1f} MWh, "
            f"median load shed={sub['load_shed_MWh'].median():.1f} MWh, "
            f"median SOC drift={sub['soc_drift_MWh'].median():.1f} MWh"
        ),
    })

summary_table = pd.DataFrame(summary_rows)
display(summary_table)
"""
    )
)


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

OUT_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT_PATH}")
