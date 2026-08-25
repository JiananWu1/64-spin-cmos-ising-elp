import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter, FuncFormatter
from matplotlib.lines import Line2D
from pathlib import Path


# ============================================================
# Repository-relative data and output paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_RESULTS_DIR = REPO_ROOT / "results" / "simulation" / "random_qubo"
MEASUREMENT_RESULTS_DIR = REPO_ROOT / "results" / "measurement" / "random_qubo"

FIGURE_6A_FILE = SIMULATION_RESULTS_DIR / "figure_6a.csv"
FIGURE_6B_FILE = SIMULATION_RESULTS_DIR / "figure_6b.csv"
FIGURE_6C_FILE = SIMULATION_RESULTS_DIR / "figure_6c.csv"
FIGURE_6D_FILE = MEASUREMENT_RESULTS_DIR / "figure_6d.csv"
FIGURE_6E_FILE = MEASUREMENT_RESULTS_DIR / "figure_6e.csv"
FIGURE_6FG_FILE = MEASUREMENT_RESULTS_DIR / "figure_6fg.csv"
FIGURE_6H_FILE = MEASUREMENT_RESULTS_DIR / "figure_6h.csv"
OUTPUT_FILE = REPO_ROOT / "figures" / "figure_6" / "figure_6.pdf"

for input_path in (
    FIGURE_6A_FILE,
    FIGURE_6B_FILE,
    FIGURE_6C_FILE,
    FIGURE_6D_FILE,
    FIGURE_6E_FILE,
    FIGURE_6FG_FILE,
    FIGURE_6H_FILE,
):
    if not input_path.is_file():
        raise FileNotFoundError(f"Required input file not found: {input_path}")


# ============================================================
# Figure dimensions and typography
# ============================================================
MM_TO_IN = 1.0 / 25.4

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 244.0 * 0.80

FIGURE_WIDTH = FIGURE_WIDTH_MM * MM_TO_IN
FIGURE_HEIGHT = FIGURE_HEIGHT_MM * MM_TO_IN

FONT_SCALE = FIGURE_WIDTH / 18.3

PANEL_LABEL_FONTSIZE = 28 * FONT_SCALE
LABEL_FONTSIZE = 24 * FONT_SCALE
TICK_FONTSIZE = 20 * FONT_SCALE
LEGEND_FONTSIZE = 15.5 * FONT_SCALE

WSPACE = 0.34
HSPACE = 0.28

PANEL_LABEL_X = -0.24
PANEL_LABEL_Y = 1.00

CURVE_LINEWIDTH = 1.05
GRAY_BAND_ALPHA = 0.36


# ============================================================
# Matplotlib export and font settings
# ============================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

plt.rcParams["axes.linewidth"] = 1.0


# ============================================================
# Color palettes
# ============================================================
method_colors = {
    "Gradient descent only": "#1f77b4",
    "SIS-ELP": "#d62728",
    "PI-ELP": "#2ca02c",
    "w/o PI-ELP/SIS-ELP": "#1f77b4",
    "w/o PI-ELP": "#d62728",
}

N_colors = {
    16: "#d62728",
    32: "#ff7f0e",
    48: "#1f77b4",
    64: "#2ca02c",
}


# ============================================================
# Utility functions
# ============================================================
def clean_numeric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def k_formatter(x, pos):
    return f"{x / 1000:.1f}k"


def style_legend(leg):
    if leg is None:
        return

    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


def style_axes(ax, log=False):
    if log:
        ax.grid(True, which="major", linewidth=0.22, alpha=0.26)
        ax.grid(True, which="minor", linewidth=0.14, alpha=0.18)
    else:
        ax.grid(True, linewidth=0.22, alpha=0.26)

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        width=0.7,
        length=2.5,
        pad=1.5,
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def add_panel_label(ax, label):
    ax.text(
        PANEL_LABEL_X,
        PANEL_LABEL_Y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        clip_on=False,
    )


def set_label(ax, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            labelpad=2.0,
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            labelpad=2.0,
        )


def fmt_plain(x, ndigits=1):
    if pd.isna(x):
        return "nan"

    return f"{x:,.{ndigits}f}"


def draw_stat_marker(ax, x, y, color, marker):
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=28,
        facecolors=color,
        edgecolors="black",
        linewidths=0.55,
        zorder=5,
    )


def add_stats_legend(ax, rows, unit_text, loc="upper left", ndigits=1):
    handles = []
    labels = []

    for row in rows:
        handles.append(Line2D([0], [0], color=row["color"], linewidth=1.2))
        labels.append(row["label"])

    for row in rows:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markersize=5.6,
                markerfacecolor=row["color"],
                markeredgecolor="black",
                markeredgewidth=0.5,
                color=row["color"],
            )
        )
        labels.append(f"mean = {fmt_plain(row['mean'], ndigits)} {unit_text}")

    for row in rows:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=4.8,
                markerfacecolor=row["color"],
                markeredgecolor="black",
                markeredgewidth=0.5,
                color=row["color"],
            )
        )
        labels.append(f"median = {fmt_plain(row['median'], ndigits)} {unit_text}")

    leg = ax.legend(
        handles,
        labels,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        loc=loc,
        handlelength=1.25,
        labelspacing=0.22,
        borderpad=0.28,
        handletextpad=0.52,
    )

    style_legend(leg)

    return leg


def grouped_bar_hist(
    ax,
    data_dict,
    color_dict,
    bin_width=100,
    percent=True,
    shade_first_bin=False,
):
    all_vals = pd.concat(
        [clean_numeric(v) for v in data_dict.values()]
    ).dropna()

    bins = np.arange(
        np.floor(all_vals.min() / bin_width) * bin_width,
        np.ceil(all_vals.max() / bin_width) * bin_width + bin_width,
        bin_width,
    )

    centers = 0.5 * (bins[:-1] + bins[1:])

    n = len(data_dict)
    bar_width = bin_width * 0.82 / n
    offsets = (np.arange(n) - (n - 1) / 2.0) * bar_width

    if shade_first_bin and len(bins) >= 2:
        ax.axvspan(
            bins[0],
            bins[1],
            color="gray",
            alpha=GRAY_BAND_ALPHA,
            zorder=0,
        )

    for i, (label, values) in enumerate(data_dict.items()):
        vals = clean_numeric(values).dropna().values
        counts, _ = np.histogram(vals, bins=bins)

        if percent:
            heights = counts / counts.sum() * 100.0
        else:
            heights = counts

        ax.bar(
            centers + offsets[i],
            heights,
            width=bar_width,
            align="center",
            color=color_dict[label],
            edgecolor="black",
            linewidth=0.35,
            label=label,
            alpha=0.82,
            zorder=2,
        )

    return bins, centers


def half_dot_shift_data(ax, marker_size_s, x_min, x_max, fig_dpi):
    bbox = ax.get_window_extent()
    ax_width_px = bbox.width
    x_range = x_max - x_min

    half_width_points = np.sqrt(marker_size_s) / 2.0
    half_width_px = half_width_points * fig_dpi / 72.0

    return half_width_px * x_range / ax_width_px


# ============================================================
# Load input data
# ============================================================
traj = pd.read_csv(FIGURE_6A_FILE).replace([np.inf, -np.inf], np.nan)
method_hist = pd.read_csv(FIGURE_6B_FILE).replace([np.inf, -np.inf], np.nan)
eta_hist = pd.read_csv(FIGURE_6C_FILE).replace([np.inf, -np.inf], np.nan)
figure_6d = pd.read_csv(FIGURE_6D_FILE).replace([np.inf, -np.inf], np.nan)
figure_6e = pd.read_csv(FIGURE_6E_FILE).replace([np.inf, -np.inf], np.nan)
figure_6fg = pd.read_csv(FIGURE_6FG_FILE).replace([np.inf, -np.inf], np.nan)
figure_6h = pd.read_csv(FIGURE_6H_FILE).replace([np.inf, -np.inf], np.nan)


def require_columns(frame, columns, label):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


require_columns(
    traj,
    ["time_s", "gradient_descent_only_H", "sis_elp_H", "pi_elp_H"],
    "trajectory CSV",
)
require_columns(
    method_hist,
    ["gradient_descent_only_H", "sis_elp_H", "pi_elp_H"],
    "method-comparison histogram CSV",
)
require_columns(
    eta_hist,
    ["H_eta_0", "H_eta_0p1", "H_eta_0p2", "H_eta_0p3", "H_eta_0p4", "H_eta_0p5"],
    "eta-sweep CSV",
)
# Panel-specific measurement columns.
require_columns(
    figure_6d,
    [
        "problem", "problem_index", "N", "density", "shuffle", "N_runs",
        "raw_H_nopert", "raw_H_pi_elp", "raw_H_sis_elp",
    ],
    "figure_6d.csv",
)
require_columns(
    figure_6e,
    [
        "problem", "N", "density", "shuffle", "N_runs", "raw_dH",
        "raw_SR99_pert", "raw_SR99_nopert",
        "post_SR99_pert", "post_SR99_nopert",
    ],
    "figure_6e.csv",
)
require_columns(
    figure_6fg,
    [
        "problem", "N", "density",
        "mean_raw_SR99_nopert", "mean_raw_SR99_pert",
        "raw_TTS99_nopert_us", "raw_TTS99_pert_us",
        "raw_ETS99_nopert_nJ", "raw_ETS99_pert_nJ",
    ],
    "figure_6fg.csv",
)
require_columns(
    figure_6h,
    [
        "problem", "N", "density",
        "mean_raw_SR99_pert", "mean_post_SR99_pert", "post_SR99_pert_gain",
    ],
    "figure_6h.csv",
)

# Panel (e) averages the shuffle-level PI-ELP Hamiltonian difference per problem.
pi_avg = (
    figure_6e.groupby(["problem", "N", "density"], as_index=False)["raw_dH"]
    .mean()
)
pi_avg["_problem_index"] = pi_avg["problem"].str.extract(r"graph(\d+)", expand=False).astype(int)

# ============================================================
# Figure layout
# ============================================================
fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

gs = gridspec.GridSpec(
    4,
    2,
    figure=fig,
    left=0.12,
    right=0.992,
    top=0.988,
    bottom=0.08,
    wspace=WSPACE,
    hspace=HSPACE,
)

axes = [
    fig.add_subplot(gs[i, j])
    for i in range(4)
    for j in range(2)
]

fig.canvas.draw()


# ============================================================
# (a) Energy trajectories
# ============================================================
ax = axes[0]

time_us = traj["time_s"].values * 1e6

trajectory_cols = {
    "Gradient descent only": "gradient_descent_only_H",
    "SIS-ELP": "sis_elp_H",
    "PI-ELP": "pi_elp_H",
}

for label, col in trajectory_cols.items():
    ax.plot(
        time_us,
        traj[col],
        linewidth=CURVE_LINEWIDTH,
        label=label,
        color=method_colors[label],
    )

set_label(ax, r"Time ($\mu$s)", "Hamiltonian")

leg = ax.legend(
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    loc="upper right",
    handlelength=1.15,
    borderpad=0.25,
)

style_legend(leg)
style_axes(ax)


# ============================================================
# (b) Method-comparison histogram
# ============================================================
ax = axes[1]

method_hist_cols = {
    "Gradient descent only": method_hist["gradient_descent_only_H"],
    "SIS-ELP": method_hist["sis_elp_H"],
    "PI-ELP": method_hist["pi_elp_H"],
}

grouped_bar_hist(
    ax,
    method_hist_cols,
    method_colors,
    bin_width=100,
    percent=True,
    shade_first_bin=True,
)

set_label(ax, "Hamiltonian", "Probability")

ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))

leg = ax.legend(
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    loc="upper right",
    handlelength=1.0,
    borderpad=0.25,
)

style_legend(leg)
style_axes(ax)


# ============================================================
# (c) Eta-sweep histogram
# ============================================================
ax = axes[2]

eta_cols = {
    r"$\eta=0$": eta_hist["H_eta_0"],
    r"$\eta=0.1$": eta_hist["H_eta_0p1"],
    r"$\eta=0.2$": eta_hist["H_eta_0p2"],
    r"$\eta=0.3$": eta_hist["H_eta_0p3"],
    r"$\eta=0.4$": eta_hist["H_eta_0p4"],
    r"$\eta=0.5$": eta_hist["H_eta_0p5"],
}

eta_colors = {
    r"$\eta=0$": "#d62728",
    r"$\eta=0.1$": "#ff7f0e",
    r"$\eta=0.2$": "#bcbd22",
    r"$\eta=0.3$": "#1f77b4",
    r"$\eta=0.4$": "#9467bd",
    r"$\eta=0.5$": "#2ca02c",
}

grouped_bar_hist(
    ax,
    eta_cols,
    eta_colors,
    bin_width=100,
    percent=True,
    shade_first_bin=True,
)

set_label(ax, "Hamiltonian", "Probability")

ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))

leg = ax.legend(
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    loc="upper right",
    ncol=2,
    handlelength=0.95,
    columnspacing=0.75,
    borderpad=0.25,
)

style_legend(leg)
style_axes(ax)


# ============================================================
# (d) Shuffle-level comparison
# ============================================================
ax = axes[3]

panel_d = figure_6d.copy()

marker_s_d = 7.5

x_min_d, x_max_d = -0.5, 19.5
ax.set_xlim(x_min_d, x_max_d)

shift_d = half_dot_shift_data(
    ax,
    marker_s_d,
    x_min_d,
    x_max_d,
    fig.dpi,
)

scatter_specs_d = [
    ("w/o PI-ELP/SIS-ELP", "raw_H_nopert", -shift_d, "o"),
    ("PI-ELP", "raw_H_pi_elp", 0.0, "s"),
    ("SIS-ELP", "raw_H_sis_elp", shift_d, "^"),
]

for label, col, offset, marker in scatter_specs_d:
    values = clean_numeric(panel_d[col])
    mask = values.notna()
    ax.scatter(
        panel_d.loc[mask, "problem_index"] + offset,
        values[mask],
        s=marker_s_d,
        marker=marker,
        alpha=0.86,
        linewidths=0.16,
        edgecolors="black",
        color=method_colors[label],
        label=label,
    )

set_label(ax, "Problem index", "Mean Hamiltonian")

ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))
ax.set_xticks(np.arange(0, 21, 5))

leg = ax.legend(
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    loc="lower right",
    handlelength=0.9,
    borderpad=0.25,
)

style_legend(leg)
style_axes(ax)


# ============================================================
# (e) Mean Hamiltonian difference
# ============================================================
ax = axes[4]

sizes = sorted(pi_avg["N"].dropna().unique())

marker_s_e = 9.0

x_min_e, x_max_e = 0.05, 0.95
ax.set_xlim(x_min_e, x_max_e)

shift_e = half_dot_shift_data(
    ax,
    marker_s_e,
    x_min_e,
    x_max_e,
    fig.dpi,
)

e_offsets = [
    (-1.5 + i) * shift_e
    for i in range(len(sizes))
]

for N, offset in zip(sizes, e_offsets):
    sub = pi_avg[pi_avg["N"] == N].copy()
    sub = sub.sort_values(["density", "_problem_index"], kind="stable")

    ax.scatter(
        sub["density"] + offset,
        sub["raw_dH"],
        s=marker_s_e,
        alpha=0.86,
        color=N_colors.get(N, None),
        edgecolors="black",
        linewidths=0.16,
        label=f"N={N}",
    )

set_label(ax, "Problem density", "Mean Hamiltonian\ndifference")

ax.set_xticks(np.arange(0.1, 1.0, 0.2))

leg = ax.legend(
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    loc="lower left",
    ncol=2,
    handlelength=0.9,
    columnspacing=0.65,
    borderpad=0.25,
)

style_legend(leg)
style_axes(ax)


# ============================================================
# (f) Time-to-solution distribution for N=64
# ============================================================
ax = axes[5]

series_info_tts = [
    {
        "plot_vals": figure_6fg["raw_TTS99_nopert_us"] / 1e6,   # plot in seconds
        "legend_vals": figure_6fg["raw_TTS99_nopert_us"] / 1000.0,  # display legend statistics in ms
        "label": "w/o PI-ELP",
        "color": method_colors["w/o PI-ELP"],
    },
    {
        "plot_vals": figure_6fg["raw_TTS99_pert_us"] / 1e6,   # plot in seconds
        "legend_vals": figure_6fg["raw_TTS99_pert_us"] / 1000.0,  # display legend statistics in ms
        "label": "PI-ELP",
        "color": method_colors["PI-ELP"],
    },
]

rows = []

for s in series_info_tts:
    x = (
        pd.Series(s["plot_vals"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    x = x[x > 0].sort_values().reset_index(drop=True)

    y = np.arange(1, len(x) + 1)

    ax.step(
        x,
        y,
        where="post",
        linewidth=CURVE_LINEWIDTH,
        color=s["color"],
        label=s["label"],
    )

    mean_plot = float(x.mean())
    median_plot = float(x.median())

    mean_y = int((x <= mean_plot).sum())
    median_y = int((x <= median_plot).sum())

    draw_stat_marker(ax, mean_plot, mean_y, s["color"], "*")
    draw_stat_marker(ax, median_plot, median_y, s["color"], "o")

    legend_vals = (
        pd.Series(s["legend_vals"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    legend_vals = legend_vals[legend_vals > 0]

    rows.append(
        {
            "label": s["label"],
            "color": s["color"],
            "mean": float(legend_vals.mean()),
            "median": float(legend_vals.median()),
        }
    )

ax.set_xscale("log")

set_label(
    ax,
    "Time-to-solution (s)",
    "Number of\nsolved problems",
)

add_stats_legend(ax, rows, "ms", loc="upper left", ndigits=2)

style_axes(ax, log=True)


# ============================================================
# (g) Energy-to-solution distribution for N=64
# ============================================================
ax = axes[6]

series_info_ets = [
    {
        "plot_vals": figure_6fg["raw_ETS99_nopert_nJ"] / 1e9,
        "legend_vals": figure_6fg["raw_ETS99_nopert_nJ"] / 1000.0,
        "label": "w/o PI-ELP",
        "color": method_colors["w/o PI-ELP"],
    },
    {
        "plot_vals": figure_6fg["raw_ETS99_pert_nJ"] / 1e9,
        "legend_vals": figure_6fg["raw_ETS99_pert_nJ"] / 1000.0,
        "label": "PI-ELP",
        "color": method_colors["PI-ELP"],
    },
]

rows = []

for s in series_info_ets:
    x = (
        pd.Series(s["plot_vals"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    x = x[x > 0].sort_values().reset_index(drop=True)

    y = np.arange(1, len(x) + 1)

    ax.step(
        x,
        y,
        where="post",
        linewidth=CURVE_LINEWIDTH,
        color=s["color"],
        label=s["label"],
    )

    mean_plot = float(x.mean())
    median_plot = float(x.median())

    mean_y = int((x <= mean_plot).sum())
    median_y = int((x <= median_plot).sum())

    draw_stat_marker(ax, mean_plot, mean_y, s["color"], "*")
    draw_stat_marker(ax, median_plot, median_y, s["color"], "o")

    legend_vals = (
        pd.Series(s["legend_vals"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    legend_vals = legend_vals[legend_vals > 0]

    rows.append(
        {
            "label": s["label"],
            "color": s["color"],
            "mean": float(legend_vals.mean()),
            "median": float(legend_vals.median()),
        }
    )

ax.set_xscale("log")

set_label(
    ax,
    "Energy-to-solution (J)",
    "Number of\nsolved problems",
)

add_stats_legend(ax, rows, "µJ", loc="upper left")

style_axes(ax, log=True)


# ============================================================
# (h) Success-rate gain
# ============================================================
ax = axes[7]

marker_s_h = 9.2

x_min_h, x_max_h = 0.05, 0.95
ax.set_xlim(x_min_h, x_max_h)

shift_h = half_dot_shift_data(
    ax,
    marker_s_h,
    x_min_h,
    x_max_h,
    fig.dpi,
)

h_offsets = [
    (-1.5 + i) * shift_h
    for i in range(len(sizes))
]

for N, offset in zip(sizes, h_offsets):
    sub = figure_6h[figure_6h["N"] == N].copy()
    sub["_problem_index"] = sub["problem"].str.extract(r"graph(\d+)", expand=False).astype(int)
    sub = sub.sort_values(["density", "_problem_index"], kind="stable")

    y_pi = clean_numeric(sub["post_SR99_pert_gain"])
    mask_pi = y_pi > 0

    ax.scatter(
        sub.loc[mask_pi, "density"] + offset,
        y_pi[mask_pi],
        s=marker_s_h,
        marker="s",
        alpha=0.86,
        color=N_colors.get(N, None),
        edgecolors="black",
        linewidths=0.16,
        label=f"N={N}",
    )

ax.set_yscale("log")

set_label(ax, "Problem density", r"Mean $SR_{\mathrm{post}}/SR_{\mathrm{raw}}$")

ax.set_xticks(np.arange(0.1, 1.0, 0.2))

handles, labels = ax.get_legend_handles_labels()

uniq = {}

for h, l in zip(handles, labels):
    if l not in uniq:
        uniq[l] = h

leg = ax.legend(
    list(uniq.values()),
    list(uniq.keys()),
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    loc="upper left",
    ncol=2,
    handlelength=0.8,
    columnspacing=0.7,
    borderpad=0.25,
)

style_legend(leg)
style_axes(ax, log=True)


# ============================================================
# Panel labels
# ============================================================
for ax, label in zip(axes, list("abcdefgh")):
    add_panel_label(ax, label)


# ============================================================
# Save figure outputs
# ============================================================
fig.savefig(OUTPUT_FILE, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")
plt.close(fig)