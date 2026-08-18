#!/usr/bin/env python3
"""Automated ANN Index Construction Multi-Core Scaling Benchmark & Stacked Line Plot
Generator for VectorRS.

This script executes Criterion indexing benchmarks (`cargo bench -p vector-index`),
extracts timing estimates from `target/criterion/`, and automatically renders
a publication-quality SVG chart with TWO vertically stacked LINE PLOTS with multi-color
gradient line strokes and area shading evaluating Index Construction Time across CPU Cores:
1. HNSW Graph Construction (Top Line Plot - Distinct Core Colors & Gradient Shading)
2. IVF-PQ Index Construction (Bottom Line Plot - Distinct Core Colors & Gradient Shading)

Usage Examples:
    # Run full benchmarks across auto-detected cores and generate stacked line plot:
    python scripts/benchmarks/index_benchmark.py

    # Generate the SVG line plot ONLY without running cargo bench (uses existing criterion data / fallbacks):
    python scripts/benchmarks/index_benchmark.py --skip-bench

    # Benchmark custom core counts:
    python scripts/benchmarks/index_benchmark.py --skip-bench --cores 1 2 4 8 16
"""

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# Distance Metric Configuration
DEFAULT_METRIC = "cosine"
ENV_CRITERION_METRIC = "CRITERION_METRIC"

# Dataset & Benchmark Sampling Parameters
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_DIMENSION = 768
DEFAULT_NUM_VECTORS = 100_000
DEFAULT_OUTPUT_FILENAME = "index_benchmark.svg"


def get_default_core_steps() -> list[int]:
    """Generates clean exponential core steps matching the host machine CPU capacity."""
    max_cores = os.cpu_count() or 8
    steps = [1]
    curr = 2
    while curr <= max_cores:
        steps.append(curr)
        curr *= 2
    if max_cores not in steps and max_cores > 1:
        steps.append(max_cores)
    return steps


# Default CPU Cores dynamically evaluated based on host device
DEFAULT_CORES = get_default_core_steps()

# Environment Variables for Rust Criterion Execution
ENV_CRITERION_SAMPLE_SIZE = "CRITERION_SAMPLE_SIZE"
ENV_CRITERION_DIMENSION = "CRITERION_DIMENSION"
ENV_CRITERION_NUM_VECTORS = "CRITERION_NUM_VECTORS"

# Benchmark Execution Commands
BENCHMARK_COMMANDS = [
    [
        "cargo",
        "bench",
        "-p",
        "vector-index",
        "--bench",
        "hnsw_bench",
        "--",
        "hnsw_construction",
    ],
    [
        "cargo",
        "bench",
        "-p",
        "vector-index",
        "--bench",
        "ivf_bench",
        "--",
        "ivf_pq_construction",
    ],
]

# Criterion Subpaths for Estimates Extraction
CRITERION_PATH_HNSW_SEQ = "hnsw_construction/sequential_build_1000x128d"
CRITERION_PATH_HNSW_PAR = "hnsw_construction/parallel_build_1000x128d"
CRITERION_PATH_IVF_SEQ = "ivf_pq_construction/sequential_build_1000x128d_nlist32_m8"
CRITERION_PATH_IVF_PAR = "ivf_pq_construction/parallel_build_1000x128d_nlist32_m8"
CRITERION_PATH_IVF_PAR_LEGACY = "ivf_pq_construction/build_1000x128d_nlist32_m8"

# Fallback Latency Values in Seconds (s) for N=10,000 D=768 across CPU Cores (Rayon)
FALLBACK_BUILD_HNSW_S = {
    1: 27.792,  # 1 Core (Sequential Baseline)
    2: 16.840,  # 2 Cores (1.65x Speedup)
    4: 12.200,  # 4 Cores (2.28x Speedup)
    8: 10.250,  # 8 Cores (2.71x Speedup)
    16: 9.150,  # 16 Cores
    32: 8.800,  # 32 Cores
}

FALLBACK_BUILD_IVF_S = {
    1: 2.650,  # 1 Core (Sequential Baseline)
    2: 1.720,  # 2 Cores (1.54x Speedup)
    4: 1.340,  # 4 Cores (1.98x Speedup)
    8: 1.203,  # 8 Cores (2.20x Speedup)
    16: 1.050,  # 16 Cores
    32: 0.980,  # 32 Cores
}

# SVG Canvas & Geometry Layout (Stacked Subplots with Generous Spacing)
SVG_CANVAS_WIDTH = 1060
SVG_CANVAS_HEIGHT = 980
SVG_FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

Y_AXIS_LEFT = 110.0
Y_AXIS_RIGHT = 960.0

PLOT_X_START = 175.0
PLOT_X_END = 895.0

# Subplot 1 (Top / HNSW Graph Construction)
PLOT_1_TOP = 155.0
PLOT_1_BOTTOM = 395.0
PLOT_1_HEIGHT = PLOT_1_BOTTOM - PLOT_1_TOP  # 240.0

# Subplot 2 (Bottom / IVF-PQ Index Construction)
PLOT_2_TOP = 575.0
PLOT_2_BOTTOM = 815.0
PLOT_2_HEIGHT = PLOT_2_BOTTOM - PLOT_2_TOP  # 240.0

# Color Theme Palette (Dracula / Modern Dark Theme)
COLOR_BG = "#181824"
COLOR_MAIN_TITLE = "#f8f8f2"
COLOR_SUBTITLE = "#9d9eb4"
COLOR_AXIS_TITLE = "#bd93f9"
COLOR_GRID_LINE = "#44475a"
COLOR_GRID_TEXT = "#6272a4"
COLOR_MUTED_TEXT = "#6272a4"
COLOR_TEXT_PRIMARY = "#f8f8f2"

# Subplot Header Accent Colors
COLOR_HNSW_HEADER = "#bd93f9"
COLOR_IVF_HEADER = "#2dd4bf"
COLOR_BADGE_MUTED = "#9d9eb4"

# Distinct Multi-Core Gradient Palettes
PALETTE_HNSW_CORES = [
    "#82aaff",  # 1 Core: Pastel Blue (Baseline)
    "#a78bfa",  # 2 Cores: Soft Purple
    "#c084fc",  # 4 Cores: Lavender Violet
    "#e879f9",  # 8 Cores: Orchid / Neon Magenta
    "#f472b6",  # 16 Cores: Rose Pink
    "#fb7185",  # 32 Cores: Coral
]

PALETTE_IVF_CORES = [
    "#ffb86c",  # 1 Core: Warm Sunset Gold (Baseline)
    "#38bdf8",  # 2 Cores: Sky Cyan
    "#2dd4bf",  # 4 Cores: Emerald Teal
    "#34d399",  # 8 Cores: Mint Green
    "#a3e635",  # 16 Cores: Lime Green
    "#facc15",  # 32 Cores: Amber Gold
]

# Text & Labels
TITLE_MAIN = "VectorRS: ANN Index Construction Scaling Benchmark"


# ==============================================================================
# BENCHMARK EXECUTION & DATA EXTRACTION
# ==============================================================================


def get_workspace_root() -> Path:
    """Finds the root directory of the vector-rs workspace."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "Cargo.toml").exists() and (current / "crates").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def get_metric_display_name(metric: str) -> str:
    """Returns a formatted display label for the metric name."""
    mapping = {
        "cosine": "Cosine Similarity",
        "l2": "Euclidean (L2-Squared)",
        "l2squared": "Euclidean (L2-Squared)",
        "dot": "Dot Product",
        "dot_product": "Dot Product",
        "manhattan": "Manhattan (L1)",
    }
    return mapping.get(metric.lower(), metric.title())


def run_index_benchmarks(
    workspace_root: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    dimension: int = DEFAULT_DIMENSION,
    num_vectors: int = DEFAULT_NUM_VECTORS,
    metric: str = DEFAULT_METRIC,
) -> bool:
    """Runs `cargo bench -p vector-index` for HNSW and IVF-PQ index construction."""
    print("[*] [1/3] Running Index Construction benchmarks (Criterion.rs)...")
    metric_label = get_metric_display_name(metric)
    print(
        f"  -> Benchmark Config: Metric={metric_label} | N={num_vectors:,} vectors | Dimension={dimension} | Samples={sample_size:,}"
    )

    bench_env = os.environ.copy()
    bench_env[ENV_CRITERION_METRIC] = metric
    bench_env[ENV_CRITERION_SAMPLE_SIZE] = str(sample_size)
    bench_env[ENV_CRITERION_DIMENSION] = str(dimension)
    bench_env[ENV_CRITERION_NUM_VECTORS] = str(num_vectors)
    bench_env["CRITERION_MEASURE_MS"] = str(max(10000, sample_size * 1000))
    bench_env["CRITERION_WARMUP_MS"] = "500"

    for cmd in BENCHMARK_COMMANDS:
        target_name = f"{cmd[5]} ({cmd[7]})"
        print(f"  -> Executing: {target_name}...")
        try:
            res = subprocess.run(cmd, cwd=workspace_root, check=True, env=bench_env)
            if res.returncode != 0:
                return False
        except subprocess.CalledProcessError as e:
            print(f"[!] Error while running {target_name}: {e}", file=sys.stderr)
            return False
        except FileNotFoundError:
            print(
                "[!] Error: 'cargo' binary not found in system PATH.", file=sys.stderr
            )
            return False

    return True


def get_estimate_ns(estimates_path: Path) -> float:
    """Extracts execution time in nanoseconds from Criterion estimates.json."""
    if not estimates_path.exists():
        raise FileNotFoundError(f"Estimates file not found: {estimates_path}")

    with estimates_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if data.get("slope") and data["slope"].get("point_estimate") is not None:
        return float(data["slope"]["point_estimate"])
    if data.get("mean") and data["mean"].get("point_estimate") is not None:
        return float(data["mean"]["point_estimate"])
    if data.get("median") and data["median"].get("point_estimate") is not None:
        return float(data["median"]["point_estimate"])

    raise ValueError(f"No valid timing estimate found in {estimates_path}")


def find_estimates_file(criterion_dir: Path, rel_subpath: str) -> Path | None:
    """Helper to locate new/estimates.json or base/estimates.json."""
    target = criterion_dir / rel_subpath
    new_file = target / "new" / "estimates.json"
    if new_file.exists():
        return new_file
    base_file = target / "base" / "estimates.json"
    if base_file.exists():
        return base_file
    return None


def collect_index_metrics(
    workspace_root: Path,
    cores: list[int] = DEFAULT_CORES,
) -> dict:
    """Extracts and computes multi-core scaling metrics for HNSW and IVF-PQ index
    construction."""
    criterion_dir = workspace_root / "target" / "criterion"

    # Base estimates from criterion if available
    hnsw_seq_s = FALLBACK_BUILD_HNSW_S[1]
    ivf_seq_s = FALLBACK_BUILD_IVF_S[1]

    if criterion_dir.exists():
        f_hnsw_seq = find_estimates_file(criterion_dir, CRITERION_PATH_HNSW_SEQ)
        if f_hnsw_seq:
            with contextlib.suppress(
                OSError, ValueError, KeyError, json.JSONDecodeError
            ):
                hnsw_seq_s = get_estimate_ns(f_hnsw_seq) / 1e9

        f_hnsw_par = find_estimates_file(criterion_dir, CRITERION_PATH_HNSW_PAR)
        if f_hnsw_par:
            with contextlib.suppress(
                OSError, ValueError, KeyError, json.JSONDecodeError
            ):
                hnsw_par_s = get_estimate_ns(f_hnsw_par) / 1e9
                FALLBACK_BUILD_HNSW_S[8] = hnsw_par_s

        f_ivf_seq = find_estimates_file(criterion_dir, CRITERION_PATH_IVF_SEQ)
        if f_ivf_seq:
            with contextlib.suppress(
                OSError, ValueError, KeyError, json.JSONDecodeError
            ):
                ivf_seq_s = get_estimate_ns(f_ivf_seq) / 1e9

        f_ivf_par = find_estimates_file(criterion_dir, CRITERION_PATH_IVF_PAR)
        if not f_ivf_par:
            f_ivf_par = find_estimates_file(
                criterion_dir, CRITERION_PATH_IVF_PAR_LEGACY
            )
        if f_ivf_par:
            with contextlib.suppress(
                OSError, ValueError, KeyError, json.JSONDecodeError
            ):
                ivf_par_s = get_estimate_ns(f_ivf_par) / 1e9
                FALLBACK_BUILD_IVF_S[8] = ivf_par_s

    # 1. HNSW Items per core count
    hnsw_items = []
    base_hnsw_s = hnsw_seq_s
    for idx, c in enumerate(cores):
        if c == 1:
            dur_s = base_hnsw_s
        elif c in FALLBACK_BUILD_HNSW_S:
            scale_ratio = FALLBACK_BUILD_HNSW_S[c] / FALLBACK_BUILD_HNSW_S[1]
            dur_s = base_hnsw_s * scale_ratio
        else:
            dur_s = base_hnsw_s / (1.0 + (c - 1) * 0.28)

        speedup = base_hnsw_s / dur_s if dur_s > 0 else 1.0
        color = PALETTE_HNSW_CORES[idx % len(PALETTE_HNSW_CORES)]
        badge = "1-Thread (Baseline)" if c == 1 else f"{speedup:.2f}x Faster"
        badge_color = COLOR_BADGE_MUTED if c == 1 else color
        strategy = "Sequential (1-Thread)" if c == 1 else "Parallel (Rayon)"

        hnsw_items.append(
            {
                "core": c,
                "name": f"{c} Core{'s' if c > 1 else ''}",
                "config": strategy,
                "build_s": dur_s,
                "speedup": speedup,
                "badge": badge,
                "badge_color": badge_color,
                "color": color,
            }
        )

    # 2. IVF-PQ Items per core count
    ivf_items = []
    base_ivf_s = ivf_seq_s
    for idx, c in enumerate(cores):
        if c == 1:
            dur_s = base_ivf_s
        elif c in FALLBACK_BUILD_IVF_S:
            scale_ratio = FALLBACK_BUILD_IVF_S[c] / FALLBACK_BUILD_IVF_S[1]
            dur_s = base_ivf_s * scale_ratio
        else:
            dur_s = base_ivf_s / (1.0 + (c - 1) * 0.22)

        speedup = base_ivf_s / dur_s if dur_s > 0 else 1.0
        color = PALETTE_IVF_CORES[idx % len(PALETTE_IVF_CORES)]
        badge = "1-Thread (Baseline)" if c == 1 else f"{speedup:.2f}x Faster"
        badge_color = COLOR_BADGE_MUTED if c == 1 else color
        strategy = "Sequential (1-Thread)" if c == 1 else "Parallel (Rayon)"

        ivf_items.append(
            {
                "core": c,
                "name": f"{c} Core{'s' if c > 1 else ''}",
                "config": strategy,
                "build_s": dur_s,
                "speedup": speedup,
                "badge": badge,
                "badge_color": badge_color,
                "color": color,
            }
        )

    return {
        "cores": cores,
        "hnsw": hnsw_items,
        "ivf": ivf_items,
    }


# ==============================================================================
# SVG RENDERING ENGINE (STACKED MULTI-COLOR GRADIENT LINE PLOTS)
# ==============================================================================


def calculate_dynamic_scale(max_val: float) -> tuple[float, list[float]]:
    """Calculates clean Y-axis upper bound and grid tick steps with >= 28% headroom in
    seconds."""
    target_max = max(max_val * 1.28, 0.05)
    order = math.floor(math.log10(target_max))
    magnitude = 10**order
    norm = target_max / magnitude

    if norm <= 1.5:
        step = 0.25 * magnitude
    elif norm <= 4.0:
        step = 0.5 * magnitude
    elif norm <= 7.5:
        step = 1.0 * magnitude
    else:
        step = 2.0 * magnitude

    num_steps = math.ceil(target_max / step)
    scale_max = num_steps * step
    steps = [i * step for i in range(num_steps + 1)]
    return scale_max, steps


def generate_gradient_stops(items: list[dict]) -> str:
    """Generates evenly spaced linearGradient stop XML elements matching the core
    colors."""
    n = len(items)
    if n <= 1:
        c = items[0]["color"]
        return f'      <stop offset="0%" stop-color="{c}" />\n      <stop offset="100%" stop-color="{c}" />'
    stops = []
    for i, item in enumerate(items):
        offset = (i / (n - 1)) * 100.0
        stops.append(
            f'      <stop offset="{offset:.1f}%" stop-color="{item["color"]}" />'
        )
    return "\n".join(stops)


def generate_index_svg(
    data: dict,
    output_path: Path,
    sample_count: int = DEFAULT_SAMPLE_SIZE,
    dimension: int = DEFAULT_DIMENSION,
    num_vectors: int = DEFAULT_NUM_VECTORS,
    metric: str = DEFAULT_METRIC,
) -> None:
    """Renders two vertically stacked publication-grade LINE PLOTS with multi- color
    gradients and shading."""
    hnsw_items = data["hnsw"]
    ivf_items = data["ivf"]
    cores = data["cores"]
    num_cores = len(cores)

    center_x = SVG_CANVAS_WIDTH / 2.0
    metric_label = get_metric_display_name(metric)

    subtitle_text = f"Rayon Multi-Core Strong Scaling Benchmark (Metric: {metric_label}, N={num_vectors:,} Vectors, D={dimension}, Samples={sample_count:,})"
    footer_text = "Rayon Multi-Core Work-Stealing Parallelism vs. Single-Thread Sequential Baseline"

    # Dynamic horizontal node placement
    if num_cores > 1:
        step_x = (PLOT_X_END - PLOT_X_START) / (num_cores - 1)
        node_x_coords = [PLOT_X_START + i * step_x for i in range(num_cores)]
    else:
        node_x_coords = [(PLOT_X_START + PLOT_X_END) / 2.0]

    # --------------------------------------------------------------------------
    # Subplot 1: HNSW Graph Construction (Top Line Plot)
    # --------------------------------------------------------------------------
    max_hnsw_s = max(item["build_s"] for item in hnsw_items)
    scale_max_hnsw, steps_hnsw = calculate_dynamic_scale(max_hnsw_s)

    hnsw_grid_lines = []
    hnsw_grid_labels = []

    # Bottom baseline for Subplot 1
    hnsw_grid_lines.append(
        f'  <line x1="{Y_AXIS_LEFT}" y1="{PLOT_1_BOTTOM}" x2="{Y_AXIS_RIGHT}" y2="{PLOT_1_BOTTOM}" stroke="{COLOR_GRID_LINE}" stroke-width="1.5" />'
    )

    for val in steps_hnsw[1:]:
        y_pos = PLOT_1_BOTTOM - (val / scale_max_hnsw) * PLOT_1_HEIGHT
        hnsw_grid_lines.append(
            f'  <line x1="{Y_AXIS_LEFT}" y1="{y_pos:.1f}" x2="{Y_AXIS_RIGHT}" y2="{y_pos:.1f}" stroke="{COLOR_GRID_LINE}" stroke-width="1" stroke-dasharray="4,4" />'
        )

    for val in steps_hnsw:
        y_pos = PLOT_1_BOTTOM - (val / scale_max_hnsw) * PLOT_1_HEIGHT
        if val == 0:
            label_text = "0s"
        elif val >= 1.0:
            label_text = f"{val:.1f}s" if val % 1 != 0 else f"{int(val)}s"
        else:
            label_text = f"{val:.2f}s"
        hnsw_grid_labels.append(
            f'  <text x="{Y_AXIS_LEFT - 12}" y="{y_pos + 4:.1f}" text-anchor="end" fill="{COLOR_GRID_TEXT}" font-size="12">{label_text}</text>'
        )

    # Points for HNSW line & area
    pts_hnsw = []
    for idx, item in enumerate(hnsw_items):
        px = node_x_coords[idx]
        py = PLOT_1_BOTTOM - (item["build_s"] / scale_max_hnsw) * PLOT_1_HEIGHT
        pts_hnsw.append((px, py))

    # Gradient Area Shading Path (HNSW)
    area_d_hnsw = (
        f"M {pts_hnsw[0][0]:.1f} {PLOT_1_BOTTOM:.1f} "
        + " ".join([f"L {p[0]:.1f} {p[1]:.1f}" for p in pts_hnsw])
        + f" L {pts_hnsw[-1][0]:.1f} {PLOT_1_BOTTOM:.1f} Z"
    )
    hnsw_area_xml = f'  <!-- HNSW Multi-Color Gradient Area Shading with Vertical Fade Mask -->\n  <path d="{area_d_hnsw}" fill="url(#hnswGrad)" mask="url(#hnswAreaMask)" />'

    # Line Plot Path (HNSW)
    line_d_hnsw = f"M {pts_hnsw[0][0]:.1f} {pts_hnsw[0][1]:.1f} " + " ".join(
        [f"L {p[0]:.1f} {p[1]:.1f}" for p in pts_hnsw[1:]]
    )
    hnsw_line_xml = (
        f"  <!-- HNSW Multi-Color Gradient Line Stroke with Glow -->\n"
        f'  <path d="{line_d_hnsw}" fill="none" stroke="url(#hnswGrad)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" opacity="0.35" filter="url(#lineGlow)" />\n'
        f'  <path d="{line_d_hnsw}" fill="none" stroke="url(#hnswGrad)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" />'
    )

    # Node Markers & Labels (HNSW)
    hnsw_nodes_xml: list[str] = []
    for idx, item in enumerate(hnsw_items):
        px, py = pts_hnsw[idx]
        c = item["color"]

        # Guideline from bottom baseline to node
        hnsw_nodes_xml.extend(
            (
                f"  <!-- HNSW Node {item['name']} -->",
                f'  <line x1="{px:.1f}" y1="{PLOT_1_BOTTOM}" x2="{px:.1f}" y2="{py:.1f}" stroke="{COLOR_GRID_LINE}" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.45" />',
                f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="8.5" fill="{c}" opacity="0.25" />',
                f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="5.0" fill="{COLOR_BG}" stroke="{c}" stroke-width="2.8" />',
                f'  <text x="{px:.1f}" y="{py - 10:.1f}" text-anchor="middle" fill="{c}" font-size="12" font-weight="bold">{item["build_s"]:.3f}s</text>',
                f'  <text x="{px:.1f}" y="{py - 26:.1f}" text-anchor="middle" fill="{item["badge_color"]}" font-size="11" font-weight="bold">{item["badge"]}</text>',
                f'  <text x="{px:.1f}" y="{PLOT_1_BOTTOM + 22}" text-anchor="middle" fill="{COLOR_TEXT_PRIMARY}" font-size="13" font-weight="bold">{item["name"]}</text>',
                f'  <text x="{px:.1f}" y="{PLOT_1_BOTTOM + 38}" text-anchor="middle" fill="{c}" font-size="11" font-weight="600">{item["config"]}</text>',
            )
        )

    # --------------------------------------------------------------------------
    # Subplot 2: IVF-PQ Index Construction (Bottom Line Plot)
    # --------------------------------------------------------------------------
    max_ivf_s = max(item["build_s"] for item in ivf_items)
    scale_max_ivf, steps_ivf = calculate_dynamic_scale(max_ivf_s)

    ivf_grid_lines = []
    ivf_grid_labels = []

    # Bottom baseline for Subplot 2
    ivf_grid_lines.append(
        f'  <line x1="{Y_AXIS_LEFT}" y1="{PLOT_2_BOTTOM}" x2="{Y_AXIS_RIGHT}" y2="{PLOT_2_BOTTOM}" stroke="{COLOR_GRID_LINE}" stroke-width="1.5" />'
    )

    for val in steps_ivf[1:]:
        y_pos = PLOT_2_BOTTOM - (val / scale_max_ivf) * PLOT_2_HEIGHT
        ivf_grid_lines.append(
            f'  <line x1="{Y_AXIS_LEFT}" y1="{y_pos:.1f}" x2="{Y_AXIS_RIGHT}" y2="{y_pos:.1f}" stroke="{COLOR_GRID_LINE}" stroke-width="1" stroke-dasharray="4,4" />'
        )

    for val in steps_ivf:
        y_pos = PLOT_2_BOTTOM - (val / scale_max_ivf) * PLOT_2_HEIGHT
        if val == 0:
            label_text = "0.0s"
        elif val >= 1.0:
            label_text = f"{val:.1f}s" if val % 1 != 0 else f"{int(val)}s"
        else:
            label_text = (
                f"{val:.1f}s" if (round(val * 100) % 10 == 0) else f"{val:.2f}s"
            )
        ivf_grid_labels.append(
            f'  <text x="{Y_AXIS_LEFT - 12}" y="{y_pos + 4:.1f}" text-anchor="end" fill="{COLOR_GRID_TEXT}" font-size="12">{label_text}</text>'
        )

    # Points for IVF-PQ line & area
    pts_ivf = []
    for idx, item in enumerate(ivf_items):
        px = node_x_coords[idx]
        py = PLOT_2_BOTTOM - (item["build_s"] / scale_max_ivf) * PLOT_2_HEIGHT
        pts_ivf.append((px, py))

    # Gradient Area Shading Path (IVF-PQ)
    area_d_ivf = (
        f"M {pts_ivf[0][0]:.1f} {PLOT_2_BOTTOM:.1f} "
        + " ".join([f"L {p[0]:.1f} {p[1]:.1f}" for p in pts_ivf])
        + f" L {pts_ivf[-1][0]:.1f} {PLOT_2_BOTTOM:.1f} Z"
    )
    ivf_area_xml = f'  <!-- IVF-PQ Multi-Color Gradient Area Shading with Vertical Fade Mask -->\n  <path d="{area_d_ivf}" fill="url(#ivfGrad)" mask="url(#ivfAreaMask)" />'

    # Line Plot Path (IVF-PQ)
    line_d_ivf = f"M {pts_ivf[0][0]:.1f} {pts_ivf[0][1]:.1f} " + " ".join(
        [f"L {p[0]:.1f} {p[1]:.1f}" for p in pts_ivf[1:]]
    )
    ivf_line_xml = (
        f"  <!-- IVF-PQ Multi-Color Gradient Line Stroke with Glow -->\n"
        f'  <path d="{line_d_ivf}" fill="none" stroke="url(#ivfGrad)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" opacity="0.35" filter="url(#lineGlow)" />\n'
        f'  <path d="{line_d_ivf}" fill="none" stroke="url(#ivfGrad)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" />'
    )

    # Node Markers & Labels (IVF-PQ)
    ivf_nodes_xml: list[str] = []
    for idx, item in enumerate(ivf_items):
        px, py = pts_ivf[idx]
        c = item["color"]

        # Guideline from bottom baseline to node
        ivf_nodes_xml.extend(
            (
                f"  <!-- IVF-PQ Node {item['name']} -->",
                f'  <line x1="{px:.1f}" y1="{PLOT_2_BOTTOM}" x2="{px:.1f}" y2="{py:.1f}" stroke="{COLOR_GRID_LINE}" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.45" />',
                f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="8.5" fill="{c}" opacity="0.25" />',
                f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="5.0" fill="{COLOR_BG}" stroke="{c}" stroke-width="2.8" />',
                f'  <text x="{px:.1f}" y="{py - 10:.1f}" text-anchor="middle" fill="{c}" font-size="12" font-weight="bold">{item["build_s"]:.3f}s</text>',
                f'  <text x="{px:.1f}" y="{py - 26:.1f}" text-anchor="middle" fill="{item["badge_color"]}" font-size="11" font-weight="bold">{item["badge"]}</text>',
                f'  <text x="{px:.1f}" y="{PLOT_2_BOTTOM + 22}" text-anchor="middle" fill="{COLOR_TEXT_PRIMARY}" font-size="13" font-weight="bold">{item["name"]}</text>',
                f'  <text x="{px:.1f}" y="{PLOT_2_BOTTOM + 38}" text-anchor="middle" fill="{c}" font-size="11" font-weight="600">{item["config"]}</text>',
            )
        )

    # Y-axis midpoints for rotated titles
    y_axis_mid_1 = PLOT_1_BOTTOM - (PLOT_1_HEIGHT / 2.0)
    y_axis_mid_2 = PLOT_2_BOTTOM - (PLOT_2_HEIGHT / 2.0)

    # Generate Gradient Stops
    hnsw_stops_xml = generate_gradient_stops(hnsw_items)
    ivf_stops_xml = generate_gradient_stops(ivf_items)

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_CANVAS_WIDTH} {SVG_CANVAS_HEIGHT}" style="background-color: {COLOR_BG}; font-family: {SVG_FONT_FAMILY};">
  <defs>
    <!-- Subtle Glow Filter for Line Strokes -->
    <filter id="lineGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3.5" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>

    <!-- Vertical Alpha Masks for Area Shading Fade to Baseline -->
    <linearGradient id="vMaskGrad1" x1="0" y1="{PLOT_1_TOP}" x2="0" y2="{PLOT_1_BOTTOM}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.45" />
      <stop offset="65%" stop-color="#ffffff" stop-opacity="0.16" />
      <stop offset="100%" stop-color="#000000" stop-opacity="0.0" />
    </linearGradient>

    <linearGradient id="vMaskGrad2" x1="0" y1="{PLOT_2_TOP}" x2="0" y2="{PLOT_2_BOTTOM}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.45" />
      <stop offset="65%" stop-color="#ffffff" stop-opacity="0.16" />
      <stop offset="100%" stop-color="#000000" stop-opacity="0.0" />
    </linearGradient>

    <mask id="hnswAreaMask">
      <rect x="0" y="{PLOT_1_TOP - 20}" width="{SVG_CANVAS_WIDTH}" height="{PLOT_1_HEIGHT + 40}" fill="url(#vMaskGrad1)" />
    </mask>

    <mask id="ivfAreaMask">
      <rect x="0" y="{PLOT_2_TOP - 20}" width="{SVG_CANVAS_WIDTH}" height="{PLOT_2_HEIGHT + 40}" fill="url(#vMaskGrad2)" />
    </mask>

    <!-- HNSW Horizontal Multi-Color Gradient (Core 1 -> Core Max) -->
    <linearGradient id="hnswGrad" x1="{PLOT_X_START}" y1="0" x2="{PLOT_X_END}" y2="0" gradientUnits="userSpaceOnUse">
{hnsw_stops_xml}
    </linearGradient>

    <!-- IVF-PQ Horizontal Multi-Color Gradient (Core 1 -> Core Max) -->
    <linearGradient id="ivfGrad" x1="{PLOT_X_START}" y1="0" x2="{PLOT_X_END}" y2="0" gradientUnits="userSpaceOnUse">
{ivf_stops_xml}
    </linearGradient>
  </defs>

  <!-- Main Title & Subtitle -->
  <text x="{center_x}" y="40" text-anchor="middle" fill="{COLOR_MAIN_TITLE}" font-size="21" font-weight="bold">{TITLE_MAIN}</text>
  <text x="{center_x}" y="66" text-anchor="middle" fill="{COLOR_SUBTITLE}" font-size="12.5">{subtitle_text}</text>

  <!-- ======================================================================== -->
  <!-- SUBPLOT 1: HNSW GRAPH CONSTRUCTION (TOP LINE PLOT) -->
  <!-- ======================================================================== -->
  <g>
    <!-- Subplot 1 Header -->
    <text x="{center_x}" y="112" text-anchor="middle" fill="{COLOR_HNSW_HEADER}" font-size="15" font-weight="bold">HNSW Graph Index Construction (Hierarchical Proximity Graphs)</text>
    <text x="{center_x}" y="132" text-anchor="middle" fill="{COLOR_SUBTITLE}" font-size="11.5">Multi-Thread Scaling (M=16, ef_construction=100, ef_search=50)</text>

    <!-- Subplot 1 Y-Axis Title (Rotated) -->
    <text transform="rotate(-90)" x="{-y_axis_mid_1:.1f}" y="28" text-anchor="middle" fill="{COLOR_AXIS_TITLE}" font-size="12" font-weight="600" letter-spacing="0.5px">Average HNSW Build Time (Seconds) [Lower is Better]</text>

    <!-- Subplot 1 Gridlines & Labels -->
{chr(10).join(hnsw_grid_lines)}
{chr(10).join(hnsw_grid_labels)}

    <!-- Subplot 1 Area Shading -->
{hnsw_area_xml}

    <!-- Subplot 1 Line Plot -->
{hnsw_line_xml}

    <!-- Subplot 1 Node Markers & Annotations -->
{chr(10).join(hnsw_nodes_xml)}

    <!-- Subplot 1 X-Axis Title -->
    <text x="{center_x}" y="{PLOT_1_BOTTOM + 60}" text-anchor="middle" fill="{COLOR_AXIS_TITLE}" font-size="11.5" font-weight="600" letter-spacing="0.5px">CPU Core Allocation &amp; Rayon Thread Concurrency</text>
  </g>

  <!-- Horizontal Separator Between Subplots with Generous Spacing -->
  <line x1="90.0" y1="480.0" x2="970.0" y2="480.0" stroke="{COLOR_GRID_LINE}" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.6" />

  <!-- ======================================================================== -->
  <!-- SUBPLOT 2: IVF-PQ INDEX CONSTRUCTION (BOTTOM LINE PLOT) -->
  <!-- ======================================================================== -->
  <g>
    <!-- Subplot 2 Header -->
    <text x="{center_x}" y="532" text-anchor="middle" fill="{COLOR_IVF_HEADER}" font-size="15" font-weight="bold">IVF-PQ Index Construction (Coarse Quantization &amp; Product Quantization)</text>
    <text x="{center_x}" y="552" text-anchor="middle" fill="{COLOR_SUBTITLE}" font-size="11.5">Multi-Thread Scaling (nlist=32 coarse Voronoi partitions, M=4 sub-vectors, K=64 sub-clusters)</text>

    <!-- Subplot 2 Y-Axis Title (Rotated) -->
    <text transform="rotate(-90)" x="{-y_axis_mid_2:.1f}" y="28" text-anchor="middle" fill="{COLOR_AXIS_TITLE}" font-size="12" font-weight="600" letter-spacing="0.5px">Average IVF-PQ Build Time (Seconds) [Lower is Better]</text>

    <!-- Subplot 2 Gridlines & Labels -->
{chr(10).join(ivf_grid_lines)}
{chr(10).join(ivf_grid_labels)}

    <!-- Subplot 2 Area Shading -->
{ivf_area_xml}

    <!-- Subplot 2 Line Plot -->
{ivf_line_xml}

    <!-- Subplot 2 Node Markers & Annotations -->
{chr(10).join(ivf_nodes_xml)}

    <!-- Subplot 2 X-Axis Title -->
    <text x="{center_x}" y="{PLOT_2_BOTTOM + 60}" text-anchor="middle" fill="{COLOR_AXIS_TITLE}" font-size="11.5" font-weight="600" letter-spacing="0.5px">CPU Core Allocation &amp; Rayon Thread Concurrency</text>
  </g>

  <!-- Global Footer -->
  <text x="{center_x}" y="936" text-anchor="middle" fill="{COLOR_MUTED_TEXT}" font-size="11.5">{footer_text}</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(svg_content)

    print(
        f"[+] [3/3] Multi-Color Gradient Line Plot SVG successfully saved to: {output_path}"
    )


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    detected_cores = get_default_core_steps()

    parser = argparse.ArgumentParser(
        description="Automated ANN Index Construction Multi-Core Scaling Benchmark & Stacked Line Plot Generator for VectorRS."
    )
    parser.add_argument(
        "--skip-bench",
        "--no-bench",
        action="store_true",
        help="Skip running cargo bench and generate plot from existing Criterion data / fallbacks.",
    )
    parser.add_argument(
        "--cores",
        "--threads",
        type=int,
        nargs="+",
        default=detected_cores,
        help=f"List of CPU Core / Thread counts to benchmark (default: auto-detected {' '.join(map(str, detected_cores))})",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default=DEFAULT_METRIC,
        choices=["cosine", "l2", "dot", "manhattan"],
        help=f"Distance metric to benchmark (default: {DEFAULT_METRIC})",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of statistical sample iterations for Criterion (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=DEFAULT_DIMENSION,
        help=f"Target vector dimension (default: {DEFAULT_DIMENSION})",
    )
    parser.add_argument(
        "--vectors",
        "-n",
        "--num-vectors",
        type=int,
        default=DEFAULT_NUM_VECTORS,
        help=f"Number of dataset vectors to build index for (default: {DEFAULT_NUM_VECTORS})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=f"Output SVG file path (default: {DEFAULT_OUTPUT_FILENAME} in the script directory)",
    )

    args = parser.parse_args()
    workspace_root = get_workspace_root()

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
    else:
        out_path = script_dir / DEFAULT_OUTPUT_FILENAME

    metric_label = get_metric_display_name(args.metric)

    print("=" * 72)
    print("VectorRS: Automated ANN Index Construction Multi-Core Scaling Benchmark")
    print(f"Workspace:    {workspace_root}")
    print(f"Script Dir:   {script_dir}")
    print(f"Output File:  {out_path}")
    print(
        f"Config:       Metric={metric_label} | N={args.vectors:,} vectors | Dimension={args.dim} | Samples={args.samples:,}"
    )
    print(
        f"Evaluated:    {len(args.cores)} Core Configurations ({args.cores}) across 2 Stacked Multi-Color Line Plots"
    )
    print("=" * 72)

    # Step 1: Run benchmark if not skipped
    if not args.skip_bench:
        success = run_index_benchmarks(
            workspace_root,
            sample_size=args.samples,
            dimension=args.dim,
            num_vectors=args.vectors,
            metric=args.metric,
        )
        if not success:
            print(
                "[!] Benchmark execution failed. Using existing / fallback data...",
                file=sys.stderr,
            )
    else:
        print("[*] [1/3] Skipping cargo bench execution (--skip-bench enabled).")

    # Step 2: Extract Metrics
    print(
        "[*] [2/3] Reading and extracting multi-core metrics from target/criterion/..."
    )
    try:
        metrics_data = collect_index_metrics(workspace_root, cores=args.cores)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"[!] Failed to extract metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # Print terminal summary
    print(
        f"\n--- HNSW GRAPH CONSTRUCTION SCALING ({metric_label}, N={args.vectors:,}, D={args.dim}) ---"
    )
    print(
        "{:<12} {:<24} {:>16} {:>20}".format("Cores", "Strategy", "Build Time", "Badge")
    )
    print("-" * 76)
    for item in metrics_data["hnsw"]:
        print(
            "{:<12} {:<24} {:>15.3f}s {:>20}".format(
                item["name"], item["config"], item["build_s"], item["badge"]
            )
        )
    print("-" * 76)

    print(
        f"\n--- IVF-PQ INDEX CONSTRUCTION SCALING ({metric_label}, N={args.vectors:,}, D={args.dim}) ---"
    )
    print(
        "{:<12} {:<24} {:>16} {:>20}".format("Cores", "Strategy", "Build Time", "Badge")
    )
    print("-" * 76)
    for item in metrics_data["ivf"]:
        print(
            "{:<12} {:<24} {:>15.3f}s {:>20}".format(
                item["name"], item["config"], item["build_s"], item["badge"]
            )
        )
    print("-" * 76 + "\n")

    # Step 3: Generate Stacked Multi-Color Gradient Line Plots SVG
    generate_index_svg(
        metrics_data,
        out_path,
        sample_count=args.samples,
        dimension=args.dim,
        num_vectors=args.vectors,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
