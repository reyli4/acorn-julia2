from pathlib import Path
import json


OUT = Path(
    "acorn-julia/runs/low_RE_mod_elec_iter0/analysis/"
    "seasonal_stage1_hourly_dispatch_with_seasonal.ipynb"
)


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """# Hourly dispatch view with seasonal-storage generation and charging

This notebook creates a figure similar to the example you shared: a two-panel hourly dispatch view for a selected window, with seasonal-storage **discharge shown as added generation above the stack** and seasonal-storage **charging shown below zero**.

The default comparison is:
- base / no seasonal-use reference: `stagel_3month_base`
- seasonal case: `3month_lamba_3.5_seasonal`

Change the parameter cell if you want a different run, year, or week.
"""
    ),
    code(
        """from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

RUN_BASE = "stagel_3month_base"
RUN_SCENARIO = "3month_lamba_3.5_seasonal"
YEAR = 1985
START = "1985-08-01 00:00:00"
HOURS = 168
SHOW_BASE_BATTERY = False

RUN_DIR = Path("acorn-julia/runs/low_RE_mod_elec_iter0")
INPUT_ROOT = RUN_DIR / "inputs"
OUTPUT_ROOT = RUN_DIR / "outputs" / "historical_1980_2019"

print("Base run:", RUN_BASE)
print("Scenario run:", RUN_SCENARIO)
print("Year:", YEAR)
print("Window start:", START)
print("Hours:", HOURS)
"""
    ),
    code(
        """# --- Helpers ----------------------------------------------------------------

def _strip_tz(idx):
    if hasattr(idx, "tz") and idx.tz is not None:
        return idx.tz_convert(None)
    return idx


def _time_cols(columns, skip=2):
    cols = list(columns)[skip:]
    idx = pd.to_datetime(cols, errors="coerce")
    mask = ~pd.isna(idx)
    return cols, idx, mask


def read_wide_output(path, skip=2):
    df = pd.read_csv(path)
    cols, idx, mask = _time_cols(df.columns, skip=skip)
    value_cols = [c for c, keep in zip(cols, mask) if keep]
    idx = idx[mask]
    values = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values.columns = _strip_tz(idx)
    meta = df.iloc[:, :skip].copy()
    return meta, values


def total_from_wide_output(path, skip=2):
    _, values = read_wide_output(path, skip=skip)
    return values.sum(axis=0)


def read_input_profile(path, year):
    df = pd.read_csv(path)
    time_cols = [c for c in df.columns if c != "bus_id"]
    idx = pd.to_datetime(time_cols, errors="coerce")
    mask = (~pd.isna(idx)) & (idx.year == year)
    use_cols = [c for c, keep in zip(time_cols, mask) if keep]
    idx = _strip_tz(idx[mask])
    vals = df[use_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    vals.columns = idx
    return vals


def build_gen_group_slices(input_root):
    n_nuclear = len(pd.read_csv(input_root / "genprop_nuclear_matched.csv"))
    n_ng = len(pd.read_csv(input_root / "genprop_ng_matched.csv"))
    n_hydro = len(pd.read_csv(input_root / "genprop_hydro.csv"))
    n_wind = len(pd.read_csv(input_root / "wind_historical_1980_2019.csv", usecols=["bus_id"]))
    n_solar = len(pd.read_csv(input_root / "solar_upv_historical_1980_2019.csv", usecols=["bus_id"]))

    starts = {}
    s = 0
    for name, n in [
        ("nuclear", n_nuclear),
        ("ng", n_ng),
        ("hydro", n_hydro),
        ("wind", n_wind),
        ("solar", n_solar),
    ]:
        starts[name] = slice(s, s + n)
        s += n
    return starts, s


def load_dispatch_components(run_name, year, input_root, output_root):
    run_dir = output_root / run_name
    gen_meta, gen_values = read_wide_output(run_dir / f"gen_{year}.csv", skip=2)
    slices, expected = build_gen_group_slices(input_root)
    if len(gen_values) != expected:
        raise ValueError(
            f"Generator row count mismatch for {run_name} {year}: "
            f"expected {expected}, found {len(gen_values)}"
        )

    components = {
        "nuclear": gen_values.iloc[slices["nuclear"]].sum(axis=0),
        "ng": gen_values.iloc[slices["ng"]].sum(axis=0),
        "hydro": gen_values.iloc[slices["hydro"]].sum(axis=0),
        "wind": gen_values.iloc[slices["wind"]].sum(axis=0),
        "solar": gen_values.iloc[slices["solar"]].sum(axis=0),
    }
    used = set(range(expected))
    other_rows = [i for i in range(len(gen_values)) if i not in used]
    components["other"] = gen_values.iloc[other_rows].sum(axis=0) if other_rows else pd.Series(0.0, index=gen_values.columns)

    components["seasonal_discharge"] = total_from_wide_output(run_dir / f"discharge_seasonal_{year}.csv", skip=2)
    components["seasonal_charge"] = total_from_wide_output(run_dir / f"charge_seasonal_{year}.csv", skip=2)
    components["base_discharge"] = total_from_wide_output(run_dir / f"discharge_base_{year}.csv", skip=2)
    components["base_charge"] = total_from_wide_output(run_dir / f"charge_base_{year}.csv", skip=2)
    components["load_shed"] = total_from_wide_output(run_dir / f"load_shedding_{year}.csv", skip=2)

    load_vals = read_input_profile(input_root / "load_historical_1980_2019.csv", year)
    components["load"] = load_vals.sum(axis=0)

    return pd.DataFrame(components).sort_index()


def window_df(df, start, hours):
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=hours - 1)
    out = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if out.empty:
        raise ValueError(f"No data found between {start} and {end}")
    return out


def summarize_window(df, label):
    return {
        "case": label,
        "nuclear_MWh": df["nuclear"].sum(),
        "gas_MWh": df["ng"].sum(),
        "hydro_MWh": df["hydro"].sum(),
        "wind_MWh": df["wind"].sum(),
        "solar_MWh": df["solar"].sum(),
        "seasonal_discharge_MWh": df["seasonal_discharge"].sum(),
        "seasonal_charge_MWh": df["seasonal_charge"].sum(),
        "base_discharge_MWh": df["base_discharge"].sum(),
        "base_charge_MWh": df["base_charge"].sum(),
        "load_shed_MWh": df["load_shed"].sum(),
    }
"""
    ),
    code(
        """# --- Load selected window ---------------------------------------------------
base_full = load_dispatch_components(RUN_BASE, YEAR, INPUT_ROOT, OUTPUT_ROOT)
scenario_full = load_dispatch_components(RUN_SCENARIO, YEAR, INPUT_ROOT, OUTPUT_ROOT)

base_win = window_df(base_full, START, HOURS)
scenario_win = window_df(scenario_full, START, HOURS)

summary_df = pd.DataFrame([
    summarize_window(base_win, RUN_BASE),
    summarize_window(scenario_win, RUN_SCENARIO),
])
for c in summary_df.columns[1:]:
    summary_df[c] = summary_df[c].round(2)
display(summary_df)
"""
    ),
    md("## Hourly stacked dispatch with seasonal-storage discharge above the stack and charging below zero"),
    code(
        """def plot_dispatch_panel(ax, df, title, show_base_battery=False):
    x = np.arange(len(df))
    labels = []
    arrays = []
    colors = []

    stack_order = [
        ("nuclear", "Nuclear", "#9ecae1"),
        ("ng", "Gas / thermal", "#fdd0a2"),
        ("hydro", "Large hydro", "#74c476"),
        ("wind", "Wind", "#9e9ac8"),
        ("solar", "Solar", "#fdd835"),
    ]
    if df["other"].abs().sum() > 1e-6:
        stack_order.append(("other", "Other", "#bdbdbd"))

    for col, label, color in stack_order:
        labels.append(label)
        arrays.append(df[col].values)
        colors.append(color)

    ax.stackplot(x, arrays, labels=labels, colors=colors, alpha=0.95)
    positive_top = np.sum(arrays, axis=0)

    if df["seasonal_discharge"].sum() > 0:
        ax.fill_between(
            x,
            positive_top,
            positive_top + df["seasonal_discharge"].values,
            color="#08306b",
            alpha=0.85,
            label="Seasonal discharge",
            step="mid",
        )

    if show_base_battery and df["base_discharge"].sum() > 0:
        ax.fill_between(
            x,
            positive_top + df["seasonal_discharge"].values,
            positive_top + df["seasonal_discharge"].values + df["base_discharge"].values,
            color="#2171b5",
            alpha=0.7,
            label="Base battery discharge",
            step="mid",
        )

    if df["seasonal_charge"].sum() > 0:
        ax.fill_between(
            x,
            0,
            -df["seasonal_charge"].values,
            color="#6baed6",
            alpha=0.9,
            label="Seasonal charging",
            step="mid",
        )

    if show_base_battery and df["base_charge"].sum() > 0:
        ax.fill_between(
            x,
            -df["seasonal_charge"].values,
            -(df["seasonal_charge"].values + df["base_charge"].values),
            color="#9ecae1",
            alpha=0.8,
            label="Base battery charging",
            step="mid",
        )

    ax.plot(x, df["load"].values, color="crimson", linewidth=1.5, label="Load")

    if df["load_shed"].sum() > 0:
        ax.plot(x, (df["load"] - df["load_shed"]).values, color="black", linestyle="--", linewidth=1.0, label="Served load")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Power output / MW")
    ax.set_title(title)


fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
plot_dispatch_panel(axes[0], base_win, f"(a) {RUN_BASE} — hourly dispatch", show_base_battery=SHOW_BASE_BATTERY)
plot_dispatch_panel(axes[1], scenario_win, f"(b) {RUN_SCENARIO} — hourly dispatch with seasonal storage", show_base_battery=SHOW_BASE_BATTERY)

tick_step = max(1, len(base_win) // 14)
tick_positions = np.arange(0, len(base_win), tick_step)
tick_labels = [base_win.index[i].strftime("%m-%d\\n%H:%M") for i in tick_positions]
axes[1].set_xticks(tick_positions)
axes[1].set_xticklabels(tick_labels)
axes[1].set_xlabel("Hour in selected window")

handles, labels = axes[1].get_legend_handles_labels()
uniq = dict(zip(labels, handles))
fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.98))
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
"""
    ),
    md("## Seasonal-storage contribution only"),
    code(
        """fig, ax = plt.subplots(figsize=(16, 4))
x = np.arange(len(scenario_win))
ax.fill_between(x, 0, scenario_win["seasonal_discharge"].values, color="#08306b", alpha=0.9, label="Seasonal discharge")
ax.fill_between(x, 0, -scenario_win["seasonal_charge"].values, color="#6baed6", alpha=0.9, label="Seasonal charging")
ax.plot(x, (scenario_win["seasonal_discharge"] - scenario_win["seasonal_charge"]).values, color="black", linewidth=1.2, label="Net seasonal output")
ax.axhline(0, color="black", linewidth=0.8)
tick_step = max(1, len(scenario_win) // 14)
tick_positions = np.arange(0, len(scenario_win), tick_step)
tick_labels = [scenario_win.index[i].strftime("%m-%d\\n%H:%M") for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_ylabel("MW")
ax.set_xlabel("Hour in selected window")
ax.set_title(f"Seasonal-storage hourly generation / charging — {RUN_SCENARIO}")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
"""
    ),
    md("## Difference versus base case in the same window"),
    code(
        """delta = scenario_win - base_win
delta_cols = [
    "nuclear", "ng", "hydro", "wind", "solar",
    "seasonal_discharge", "seasonal_charge", "base_discharge", "base_charge", "load_shed"
]
delta_summary = delta[delta_cols].sum().rename("MWh_change_vs_base").to_frame()
delta_summary["MWh_change_vs_base"] = delta_summary["MWh_change_vs_base"].round(2)
display(delta_summary)

fig, ax = plt.subplots(figsize=(14, 4))
plot_cols = ["ng", "hydro", "wind", "solar", "seasonal_discharge", "seasonal_charge", "load_shed"]
delta[plot_cols].plot(ax=ax)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("MW difference")
ax.set_xlabel("Hour in selected window")
ax.set_title(f"Scenario minus base — hourly change ({RUN_SCENARIO} - {RUN_BASE})")
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
print("Wrote", OUT)
