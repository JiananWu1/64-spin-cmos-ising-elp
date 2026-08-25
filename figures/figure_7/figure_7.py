from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# Figure configuration
# ============================================================
R_VALUE = 1.0
C_VALUE = 1.0
NUM_TOTAL_BITS = 64
NUM_PROBLEM_BITS = 49
BOARD_N = 7
PANEL_RATIO = 1.4

# Figure size and fonts
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
})
MM_TO_IN = 1.0 / 25.4
FIGURE_WIDTH = 183.0 * MM_TO_IN
FIGURE_HEIGHT = 183.0 * 20.0 / 24.0 * MM_TO_IN
FIGSIZE_COMBINED = (FIGURE_WIDTH, FIGURE_HEIGHT)
FONT_SCALE = FIGURE_WIDTH / 18.3
PANEL_LABEL_FONTSIZE = 28 * FONT_SCALE
LABEL_FONTSIZE = 24 * FONT_SCALE
TICK_FONTSIZE = 20 * FONT_SCALE
LEGEND_FONTSIZE = 15.5 * FONT_SCALE

# Figure layout
TOP_ROW_HEIGHT = 1.05
BOTTOM_ROW_HEIGHT = 1.85
D_AREA_WIDTH = 3.10
E_AREA_WIDTH = 1.28
C_PANEL_WIDTH_RATIO = 1.1183  # makes panel (c) match panel (e)
MAIN_HSPACE = 0.20
TOP_WSPACE = 0.58
BOTTOM_WSPACE = 0.26
BOARD_HSPACE = 0.22
BOARD_WSPACE = 0.10
BOARD_TOP_OFFSET = 0.035
PANEL_LABEL_X_OFFSET = 24  # points; larger moves labels farther left
PANEL_LABEL_Y_OFFSET = 4  # points; larger moves labels higher

# Plot style
CURVE_LINEWIDTH = 1.05
BAR_EDGE_LINEWIDTH = 0.35
BAR_ALPHA = 0.82
MARKER_SIZE = 9.0
MARKER_EDGE_LINEWIDTH = 0.16
SHOW_COLORBAR_LABEL = False
COLORBAR_LABEL = "quantized weight"
E_Y_MIN, E_Y_MAX = 0, 40
VECTOR_HEATMAP = True
INVERT_COUPLING_FOR_DISPLAY = True
HEATMAP_VMIN_DISPLAY, HEATMAP_VMAX_DISPLAY = -15, 0

# Board style
BOARD_LIGHT = "#ffffff"
BOARD_DARK = "#e0e0e0"
BOARD_EDGE_COLOR = "black"
BOARD_GRID_LW = 0.65 * FONT_SCALE
BOARD_EDGE_LW = 1.2 * FONT_SCALE
QUEEN_SYMBOL = "♛"
QUEEN_FONTFAMILY = "DejaVu Sans"
QUEEN_FONTSIZE = 27 * FONT_SCALE
QUEEN_COLOR = "#2E86DE"
BOARD_TICK_FONTSIZE = 11 * FONT_SCALE
VIOLATION_RED = "#d62728"
VIOLATION_ROWCOL_ALPHA = 0.06
VIOLATION_CELL_ALPHA = 0.20
VIOLATION_EDGE_LW = 1.8 * FONT_SCALE


# ============================================================
# Repository-relative data and output paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHT_FOLDER = REPO_ROOT / "data" / "problems" / "seven_queens"
MEASUREMENT_RESULTS_DIR = REPO_ROOT / "results" / "measurement" / "seven_queens"
FIGURE_7BC_FILE = MEASUREMENT_RESULTS_DIR / "figure_7bc.csv"
FIGURE_7D_FILE = MEASUREMENT_RESULTS_DIR / "figure_7d.csv"
FIGURE_7E_FILE = MEASUREMENT_RESULTS_DIR / "figure_7e.csv"
OUTPUT_FILE = REPO_ROOT / "figures" / "figure_7" / "figure_7.pdf"


def validate_paths() -> None:
    if not WEIGHT_FOLDER.is_dir():
        raise NotADirectoryError(f"Required weight directory not found: {WEIGHT_FOLDER}")

    for path in (FIGURE_7BC_FILE, FIGURE_7D_FILE, FIGURE_7E_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")


# ============================================================
# Shared helpers
# ============================================================
def format_value(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.10f}".rstrip("0").rstrip(".")


def parameter_labels(value: float) -> list[str]:
    compact, decimal = format_value(value), f"{float(value):.1f}"
    return list(dict.fromkeys([compact, compact.replace(".", "p"), decimal, decimal.replace(".", "p")]))


def bits_to_board(bits: str) -> np.ndarray:
    return np.fromiter((int(bit) for bit in bits), dtype=np.int8).reshape(BOARD_N, BOARD_N)


def set_label(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, fontweight="bold", labelpad=2.0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight="bold", labelpad=2.0)


def set_probability_axis(ax) -> None:
    set_label(ax, ylabel="Probability")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def style_axes(ax, grid: bool = True) -> None:
    if grid:
        ax.grid(True, axis="both", linewidth=0.22, alpha=0.26)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=0.7, length=2.5, pad=1.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.set_axisbelow(True)


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_alpha(1.0)
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_linewidth(0.6)


def style_colorbar(colorbar) -> None:
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE, width=0.7, length=2.5, pad=1.5)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    colorbar.outline.set_linewidth(1.0)


def add_aligned_panel_labels(fig, panel_rows) -> None:
    fig.canvas.draw()
    width, height = fig.get_size_inches()
    dx = (PANEL_LABEL_X_OFFSET / 72.0) / width
    dy = (PANEL_LABEL_Y_OFFSET / 72.0) / height

    for row in panel_rows:
        items, tops = [], []
        for axes, label in row:
            axes = axes if isinstance(axes, (list, tuple)) else [axes]
            boxes = [axis.get_position() for axis in axes]
            items.append((min(box.x0 for box in boxes), label))
            tops.append(max(box.y1 for box in boxes))
        y = max(tops) + dy
        for x, label in items:
            fig.text(x - dx, y, f"({label})", ha="right", va="top",
                     fontsize=PANEL_LABEL_FONTSIZE, fontweight="bold")


# ============================================================
# Panel-specific CSV loading
# ============================================================
def parse_pert(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid pert value: {value!r}")


def extract_distribution(row: dict[str, str], fields: list[str], prefix: str) -> dict[int, float]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_rate$")
    result = {}
    for field in fields:
        match = pattern.match(field)
        if match:
            value = row.get(field, "")
            result[int(match.group(1))] = float(value) if value not in {"", None} else 0.0
    return result


def to_percent_array(distribution: dict[int, float], maximum: int) -> np.ndarray:
    return 100.0 * np.array([distribution.get(i, 0.0) for i in range(maximum + 1)])


def load_figure_7bc(path: Path) -> tuple[dict, dict]:
    required = {"D_over_RC", "pert", "valid_solution_rate"}
    converted = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        missing = required - set(fields)
        if missing:
            raise ValueError(f"figure_7bc.csv is missing columns: {sorted(missing)}")
        for raw in reader:
            converted.append({
                "ratio": float(raw["D_over_RC"]),
                "pert": parse_pert(raw["pert"]),
                "success": 100.0 * float(raw.get("valid_solution_rate") or 0.0),
                "total": extract_distribution(raw, fields, "Total_constraint_violations_"),
                "row": extract_distribution(raw, fields, "row_constraint_violations_"),
                "col": extract_distribution(raw, fields, "col_constraint_violations_"),
                "diag": extract_distribution(raw, fields, "diag_constraint_violations_"),
            })

    pert_rows = [row for row in converted if row["pert"]]
    nopert_rows = [row for row in converted if not row["pert"]]
    if len(pert_rows) != 1 or len(nopert_rows) != 1:
        raise ValueError("figure_7bc.csv must contain one PI-ELP row and one no-perturbation row.")
    if not np.isclose(pert_rows[0]["ratio"], PANEL_RATIO) or not np.isclose(nopert_rows[0]["ratio"], PANEL_RATIO):
        raise ValueError(f"figure_7bc.csv must contain D/R(C)={PANEL_RATIO:.1f} data.")
    return pert_rows[0], nopert_rows[0]


def load_figure_7d(path: Path) -> dict:
    required = {"example", "spin_bits"}
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"figure_7d.csv is missing columns: {sorted(missing)}")
        for row in reader:
            bits = str(row["spin_bits"]).strip()
            if len(bits) != NUM_PROBLEM_BITS or any(bit not in "01" for bit in bits):
                raise ValueError(f"Invalid 49-bit board in figure_7d.csv: {row['example']}")
            rows[row["example"]] = bits_to_board(bits)

    success_names = ["solution_1", "solution_2", "solution_3"]
    failed_names = ["row_conflict", "column_conflict", "diagonal_conflict"]
    missing_examples = [name for name in success_names + failed_names if name not in rows]
    if missing_examples:
        raise ValueError(f"figure_7d.csv is missing examples: {missing_examples}")

    return {
        "success": [rows[name] for name in success_names],
        "failed": {
            "row": {"board": rows["row_conflict"]},
            "column": {"board": rows["column_conflict"]},
            "diagonal": {"board": rows["diagonal_conflict"]},
        },
    }


def load_figure_7e(path: Path) -> tuple[list[dict], list[dict]]:
    required = {"D_over_RC", "valid_solution_rate_nopert", "valid_solution_rate_pert"}
    pert_rows, nopert_rows = [], []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"figure_7e.csv is missing columns: {sorted(missing)}")
        for raw in reader:
            ratio = float(raw["D_over_RC"])
            nopert_rows.append({"ratio": ratio, "success": 100.0 * float(raw["valid_solution_rate_nopert"])})
            pert_rows.append({"ratio": ratio, "success": 100.0 * float(raw["valid_solution_rate_pert"])})
    pert_rows.sort(key=lambda row: row["ratio"])
    nopert_rows.sort(key=lambda row: row["ratio"])
    return pert_rows, nopert_rows


def curve_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([row["ratio"] for row in rows], dtype=float)
    y = np.array([row["success"] for row in rows], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


# ============================================================
# Panel (d): board visualization
# ============================================================
def violation_details(board: np.ndarray) -> dict[str, np.ndarray]:
    bad_rows = board.sum(axis=1) != 1
    bad_cols = board.sum(axis=0) != 1
    queen_mask = np.zeros_like(board, dtype=bool)
    diag_mask = np.zeros_like(board, dtype=bool)

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if board[r, c] and (bad_rows[r] or bad_cols[c]):
                queen_mask[r, c] = True

    diagonal_sets = []
    for diagonal in range(-(BOARD_N - 1), BOARD_N):
        diagonal_sets.append([(r, r - diagonal) for r in range(BOARD_N)
                              if 0 <= r - diagonal < BOARD_N])
    for diagonal in range(2 * BOARD_N - 1):
        diagonal_sets.append([(r, diagonal - r) for r in range(BOARD_N)
                              if 0 <= diagonal - r < BOARD_N])

    for cells in diagonal_sets:
        queens = [(r, c) for r, c in cells if board[r, c]]
        if len(queens) > 1:
            for r, c in cells:
                diag_mask[r, c] = True
            for r, c in queens:
                queen_mask[r, c] = True

    return {"bad_rows": bad_rows, "bad_cols": bad_cols,
            "diag_mask": diag_mask, "queen_mask": queen_mask}


def plot_board(ax, board: np.ndarray, mark_violations: bool) -> None:
    ax.set(xlim=(-0.5, BOARD_N - 0.5), ylim=(BOARD_N - 0.5, -0.5), aspect="equal")

    def rectangle(x, y, width, height, **kwargs):
        ax.add_patch(plt.Rectangle((x, y), width, height, **kwargs))

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            rectangle(c - 0.5, r - 0.5, 1, 1,
                      facecolor=BOARD_LIGHT if (r + c) % 2 == 0 else BOARD_DARK,
                      edgecolor="none", zorder=0)

    queen_mask = np.zeros_like(board, dtype=bool)
    if mark_violations:
        details = violation_details(board)
        queen_mask = details["queen_mask"]
        for r in range(BOARD_N):
            if details["bad_rows"][r]:
                rectangle(-0.5, r - 0.5, BOARD_N, 1, facecolor=VIOLATION_RED,
                          edgecolor="none", alpha=VIOLATION_ROWCOL_ALPHA, zorder=1)
        for c in range(BOARD_N):
            if details["bad_cols"][c]:
                rectangle(c - 0.5, -0.5, 1, BOARD_N, facecolor=VIOLATION_RED,
                          edgecolor="none", alpha=VIOLATION_ROWCOL_ALPHA, zorder=1)
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if details["diag_mask"][r, c]:
                    rectangle(c - 0.5, r - 0.5, 1, 1, facecolor=VIOLATION_RED,
                              edgecolor="none", alpha=0.12, zorder=1.5)
                if queen_mask[r, c]:
                    rectangle(c - 0.5, r - 0.5, 1, 1, facecolor=VIOLATION_RED,
                              edgecolor=VIOLATION_RED, alpha=VIOLATION_CELL_ALPHA,
                              linewidth=VIOLATION_EDGE_LW, zorder=2)

    for coordinate in np.arange(-0.5, BOARD_N + 0.5, 1):
        ax.axvline(coordinate, color=BOARD_EDGE_COLOR, linewidth=BOARD_GRID_LW, zorder=3)
        ax.axhline(coordinate, color=BOARD_EDGE_COLOR, linewidth=BOARD_GRID_LW, zorder=3)
    rectangle(-0.5, -0.5, BOARD_N, BOARD_N, facecolor="none",
              edgecolor=BOARD_EDGE_COLOR, linewidth=BOARD_EDGE_LW, zorder=4)

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if board[r, c]:
                ax.text(c, r, QUEEN_SYMBOL, ha="center", va="center",
                        fontsize=QUEEN_FONTSIZE, fontweight="bold",
                        fontfamily=QUEEN_FONTFAMILY,
                        color=VIOLATION_RED if queen_mask[r, c] else QUEEN_COLOR, zorder=5)

    ax.set_xticks(np.arange(BOARD_N), [str(i) for i in range(1, BOARD_N + 1)])
    ax.set_yticks(np.arange(BOARD_N), [str(i) for i in range(BOARD_N, 0, -1)])
    ax.tick_params(axis="both", which="both", length=0, labelsize=BOARD_TICK_FONTSIZE, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
# Panel (a): weights
# ============================================================
def find_weight_file(folder: Path, ratio: float) -> Path | None:
    candidates = []
    suffixes = (".txt", "_weights.txt", "_qubo.txt")

    for r in parameter_labels(R_VALUE):
        for c in parameter_labels(C_VALUE):
            for d in parameter_labels(ratio):
                stems = [f"7queens_R{r}_C{c}_D{d}", f"7queen_R{r}_C{c}_D{d}",
                         f"7queens_total64_R{r}_C{c}_D{d}", f"7queen_total64_R{r}_C{c}_D{d}"]
                candidates += [folder / f"{stem}{suffix}" for stem in stems for suffix in suffixes]

    # Original A/B/C filename compatibility
    for a in parameter_labels(R_VALUE):
        for b in parameter_labels(C_VALUE):
            for c in parameter_labels(ratio):
                stems = [f"7queens_A{a}_B{b}_C{c}", f"7queen_A{a}_B{b}_C{c}",
                         f"7queens_total64_A{a}_B{b}_C{c}", f"7queen_total64_A{a}_B{b}_C{c}"]
                candidates += [folder / f"{stem}{suffix}" for stem in stems for suffix in suffixes]

    return next((path for path in candidates if path.is_file()), None)


def load_weight_matrix(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                values = [float(value) for value in line.replace(",", " ").split()]
            except ValueError:
                continue
            if values:
                rows.append(values)

    if not rows:
        raise ValueError(f"No numeric data in weight file: {path}")

    matrix_rows = rows[1:] if len(rows) >= NUM_TOTAL_BITS + 1 and len(rows[0]) <= 2 else rows
    if matrix_rows and len({len(row) for row in matrix_rows}) == 1:
        shape = (len(matrix_rows), len(matrix_rows[0]))
        if shape in {(NUM_TOTAL_BITS, NUM_TOTAL_BITS), (NUM_PROBLEM_BITS, NUM_PROBLEM_BITS)}:
            raw = np.array(matrix_rows, dtype=float)
            matrix = np.zeros((NUM_TOTAL_BITS, NUM_TOTAL_BITS))
            matrix[:NUM_PROBLEM_BITS, :NUM_PROBLEM_BITS] = raw[:NUM_PROBLEM_BITS, :NUM_PROBLEM_BITS]
            if shape == (NUM_TOTAL_BITS, NUM_TOTAL_BITS):
                lower = raw[NUM_PROBLEM_BITS:, :NUM_PROBLEM_BITS]
                upper = raw[:NUM_PROBLEM_BITS, NUM_PROBLEM_BITS:]
                matrix[NUM_PROBLEM_BITS:, :NUM_PROBLEM_BITS] = lower if np.count_nonzero(lower) else upper.T
            return matrix

    triples = [(int(row[0]), int(row[1]), float(row[2])) for row in rows if len(row) >= 3]
    if not triples:
        raise ValueError(f"Unsupported weight-file format: {path}")
    indices = [value for i, j, _ in triples for value in (i, j)]
    offset = 1 if min(indices) >= 1 and max(indices) <= NUM_TOTAL_BITS else 0
    matrix = np.zeros((NUM_TOTAL_BITS, NUM_TOTAL_BITS))

    for i, j, weight in triples:
        i, j = i - offset, j - offset
        if not (0 <= i < NUM_TOTAL_BITS and 0 <= j < NUM_TOTAL_BITS):
            continue
        if i < NUM_PROBLEM_BITS and j < NUM_PROBLEM_BITS:
            matrix[i, j] = matrix[j, i] = weight
        elif i >= NUM_PROBLEM_BITS > j:
            matrix[i, j] = weight
        elif j >= NUM_PROBLEM_BITS > i:
            matrix[j, i] = weight
    return matrix


# ============================================================
# Figure generation
# ============================================================
def plot_summary(best: dict, gradient_only: dict, examples: dict,
                 pert_rows: list[dict], nopert_rows: list[dict],
                 weight_folder: Path, output: Path) -> None:
    ratio = PANEL_RATIO
    weight_file = find_weight_file(weight_folder, ratio)
    if weight_file:
        weights = load_weight_matrix(weight_file)
    else:
        print(f"Warning: no weight file found for D/R(C)={format_value(ratio)}; panel (a) uses zeros.")
        weights = np.zeros((NUM_TOTAL_BITS, NUM_TOTAL_BITS))

    fig = plt.figure(figsize=FIGSIZE_COMBINED)
    main = gridspec.GridSpec(2, 1, figure=fig, left=0.080, right=0.985, top=0.930,
                             bottom=0.100, height_ratios=[TOP_ROW_HEIGHT, BOTTOM_ROW_HEIGHT],
                             hspace=MAIN_HSPACE)
    top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=main[0, 0],
                                           width_ratios=[1.0, 1.0, C_PANEL_WIDTH_RATIO],
                                           wspace=TOP_WSPACE)
    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main[1, 0],
                                              width_ratios=[D_AREA_WIDTH, E_AREA_WIDTH],
                                              wspace=BOTTOM_WSPACE)
    boards = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=bottom[0, 0],
                                              height_ratios=[BOARD_TOP_OFFSET, 1.0, 1.0],
                                              hspace=BOARD_HSPACE, wspace=BOARD_WSPACE)

    # (a)
    ax_a = fig.add_subplot(top[0, 0])
    shown = -weights if INVERT_COUPLING_FOR_DISPLAY else weights
    if VECTOR_HEATMAP:
        edges = np.arange(NUM_TOTAL_BITS + 1) - 0.5
        image = ax_a.pcolormesh(edges, edges, shown, vmin=HEATMAP_VMIN_DISPLAY,
                                vmax=HEATMAP_VMAX_DISPLAY, cmap="viridis",
                                shading="flat", rasterized=False)
        ax_a.set(aspect="equal", xlim=(-0.5, NUM_TOTAL_BITS - 0.5),
                 ylim=(NUM_TOTAL_BITS - 0.5, -0.5))
    else:
        image = ax_a.imshow(shown, aspect="equal", vmin=HEATMAP_VMIN_DISPLAY,
                            vmax=HEATMAP_VMAX_DISPLAY, cmap="viridis",
                            interpolation="nearest", rasterized=False)
    set_label(ax_a, "Spin index", "Spin index")
    ax_a.axhline(NUM_PROBLEM_BITS - 0.5, color="white", linewidth=CURVE_LINEWIDTH)
    ax_a.axvline(NUM_PROBLEM_BITS - 0.5, color="white", linewidth=CURVE_LINEWIDTH)
    style_axes(ax_a, grid=False)
    cax = make_axes_locatable(ax_a).append_axes("right", size="4%", pad=0.05)
    colorbar = fig.colorbar(image, cax=cax)
    if SHOW_COLORBAR_LABEL:
        colorbar.set_label(COLORBAR_LABEL, fontsize=LABEL_FONTSIZE,
                           fontweight="bold", labelpad=8)
    colorbar.set_ticks([-15, -10, -5, 0])
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    style_colorbar(colorbar)

    # (b)
    ax_b = fig.add_subplot(top[0, 1])
    maximum = max(set(best["total"]) | set(gradient_only["total"]))
    x = np.arange(maximum + 1)
    width = 0.38
    ax_b.bar(x - width / 2, to_percent_array(best["total"], maximum), width=width,
             color="#2ca02c", edgecolor="black", linewidth=BAR_EDGE_LINEWIDTH,
             alpha=BAR_ALPHA, label="PI-ELP")
    ax_b.bar(x + width / 2, to_percent_array(gradient_only["total"], maximum), width=width,
             color="#d62728", edgecolor="black", linewidth=BAR_EDGE_LINEWIDTH,
             alpha=BAR_ALPHA, label="w/o PI-ELP")
    set_label(ax_b, "Total constraint violations")
    set_probability_axis(ax_b)
    ax_b.set_xticks(np.arange(0, maximum + 1, 2) if maximum > 8 else x)
    style_axes(ax_b)
    legend = ax_b.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=True,
                         handlelength=1.0, borderpad=0.28, labelspacing=0.22,
                         handletextpad=0.52)
    style_legend(legend)

    # (c)
    ax_c = fig.add_subplot(top[0, 2])
    distributions = [best["row"], best["col"], best["diag"],
                     gradient_only["row"], gradient_only["col"], gradient_only["diag"]]
    maximum = max(set().union(*(set(d) for d in distributions)))
    x = np.arange(maximum + 1)
    width = 0.12
    specs = [
        (-2.5, best["row"], "#2ca02c", None),
        (-1.5, best["col"], "#1f77b4", None),
        (-0.5, best["diag"], "#ff7f0e", None),
        (0.5, gradient_only["row"], "#2ca02c", "////"),
        (1.5, gradient_only["col"], "#1f77b4", "////"),
        (2.5, gradient_only["diag"], "#ff7f0e", "////"),
    ]
    for offset, distribution, color, hatch in specs:
        ax_c.bar(x + offset * width, to_percent_array(distribution, maximum), width=width,
                 color=color, edgecolor="black", linewidth=BAR_EDGE_LINEWIDTH,
                 alpha=1.0 if hatch else BAR_ALPHA, hatch=hatch)
    set_label(ax_c, "Violation count")
    set_probability_axis(ax_c)
    ax_c.set_xticks(x)
    style_axes(ax_c)
    labels = ["Row, PI-ELP", "Col, PI-ELP", "Diag, PI-ELP",
              "Row, w/o PI-ELP", "Col, w/o PI-ELP", "Diag, w/o PI-ELP"]
    handles = [Patch(facecolor=color, edgecolor="black", linewidth=BAR_EDGE_LINEWIDTH,
                     hatch=hatch, label=label)
               for (_, _, color, hatch), label in zip(specs, labels)]
    legend = ax_c.legend(handles=handles, fontsize=LEGEND_FONTSIZE, loc="upper right",
                         frameon=True, handlelength=1.2, borderpad=0.28,
                         labelspacing=0.22, handletextpad=0.52)
    style_legend(legend)

    # (d)
    background = fig.add_subplot(bottom[0, 0], frameon=False)
    background.set_xticks([])
    background.set_yticks([])
    background.patch.set_alpha(0.0)
    board_axes = []
    for index in range(3):
        axis = fig.add_subplot(boards[1, index])
        board_axes.append(axis)
        if index < len(examples["success"]):
            plot_board(axis, examples["success"][index], False)
            axis.set_title(f"Solution {index + 1}", fontsize=LEGEND_FONTSIZE,
                           fontweight="bold", pad=3)
        else:
            axis.axis("off")

    for index, (name, title) in enumerate([
        ("row", "Row conflict"), ("column", "Column conflict"),
        ("diagonal", "Diagonal conflict")
    ]):
        axis = fig.add_subplot(boards[2, index])
        board_axes.append(axis)
        item = examples["failed"][name]
        if item:
            plot_board(axis, item["board"], True)
        else:
            axis.axis("off")
        axis.set_title(title, fontsize=LEGEND_FONTSIZE, fontweight="bold", pad=3)

    # (e)
    ax_e = fig.add_subplot(bottom[0, 1])
    x_pert, y_pert = curve_arrays(pert_rows)
    x_base, y_base = curve_arrays(nopert_rows)
    pert_map = {round(float(x), 6): float(y) for x, y in zip(x_pert, y_pert)}
    base_map = {round(float(x), 6): float(y) for x, y in zip(x_base, y_base)}
    common = sorted(set(pert_map) & set(base_map))
    x = np.array(common)
    y_p = np.array([pert_map[value] for value in common])
    y_b = np.array([base_map[value] for value in common])
    ax_e.vlines(x, y_b, y_p, color="black", linewidth=CURVE_LINEWIDTH, alpha=0.9, zorder=1)
    ax_e.scatter(x, y_b, marker="o", s=MARKER_SIZE, color="#d62728",
                 edgecolor="black", linewidth=MARKER_EDGE_LINEWIDTH,
                 label="w/o PI-ELP", zorder=3)
    ax_e.scatter(x, y_p, marker="o", s=MARKER_SIZE, color="#2ca02c",
                 edgecolor="black", linewidth=MARKER_EDGE_LINEWIDTH,
                 label="PI-ELP", zorder=4)
    set_label(ax_e, "Diagonal-to-row/column\npenalty ratio", "Valid-solution rate")
    ax_e.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax_e.set_xticks(np.arange(0.5, 2.0 + 1e-9, 0.5))
    ax_e.set_ylim(E_Y_MIN, E_Y_MAX)
    style_axes(ax_e, grid=False)
    ax_e.grid(axis="both", linewidth=0.22, alpha=0.26, zorder=0)
    legend = ax_e.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", frameon=True,
                         handlelength=1.0, borderpad=0.28, labelspacing=0.22,
                         handletextpad=0.52)
    style_legend(legend)

    add_aligned_panel_labels(fig, [[(ax_a, "a"), (ax_b, "b"), (ax_c, "c")],
                                   [(board_axes, "d"), (ax_e, "e")]])
    fig.savefig(output, bbox_inches="tight", format="pdf", dpi=600)
    print(f"Saved {output}")
    plt.close(fig)


def main() -> None:
    validate_paths()
    best, gradient_only = load_figure_7bc(FIGURE_7BC_FILE)
    examples = load_figure_7d(FIGURE_7D_FILE)
    pert_rows, nopert_rows = load_figure_7e(FIGURE_7E_FILE)
    plot_summary(
        best,
        gradient_only,
        examples,
        pert_rows,
        nopert_rows,
        WEIGHT_FOLDER,
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()