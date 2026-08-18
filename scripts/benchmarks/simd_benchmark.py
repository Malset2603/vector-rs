#!/usr/bin/env python3
"""Automated SIMD Intrinsics Benchmark & Plot Generator for VectorRS.

This script executes Criterion distance micro-benchmarks (`cargo bench -p vector-simd`),
dynamically configured with the desired sample count (default: 100 samples),
extracts timing estimates from `target/criterion/` for 5 key metrics plus an Average summary,
and automatically renders a publication-quality SVG chart comparing Scalar vs SIMD implementations.

Usage Examples:
    # Run full benchmarks and generate the SVG plot
    python scripts/benchmarks/simd_benchmark.py

    # Generate the SVG plot ONLY without running cargo bench (uses existing criterion data)
    python scripts/benchmarks/simd_benchmark.py --skip-bench
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# Benchmark Execution
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_DIMENSION = 768
DEFAULT_OUTPUT_FILENAME = "simd_benchmark.svg"

# Environment Variables
ENV_CRITERION_SAMPLE_SIZE = "CRITERION_SAMPLE_SIZE"
ENV_CRITERION_DIMENSION = "CRITERION_DIMENSION"
ENV_CRITERION_METRICS = "CRITERION_METRICS"

# Metric Definitions to Benchmark & Plot: (criterion_group_name, display_label)
PRIMARY_METRIC_DEFINITIONS = [
    ("dot_product", "Dot Product"),
    ("l2_squared", "L2 Squared"),
    ("manhattan", "Manhattan (L1)"),
    ("cosine_similarity", "Cosine Sim"),
    ("minkowski_p3", "Minkowski (L3)"),
]

# SVG Canvas & Coordinate Geometry
SVG_CANVAS_WIDTH = 1040
SVG_CANVAS_HEIGHT = 610
PLOT_HEIGHT = 320.0
Y_AXIS_BOTTOM = 465.0
Y_AXIS_LEFT = 105.0
Y_AXIS_RIGHT_MARGIN = 55.0
Y_GRID_INTERVAL_COUNT = 4
Y_MAX_DEFAULT_FLOOR = 1600.0
Y_MAX_ROUNDING_STEP = 200.0

# Bar Geometry & Spacing
BAR_WIDTH = 40.0
BAR_GAP = 5.0
BAR_GROUP_INNER_PADDING = 20.0
MIN_SIMD_BAR_HEIGHT = 6.0
BAR_CORNER_RADIUS = 4.0

# Average Highlight Box Geometry
AVG_BOX_TOP_Y = 140.0
AVG_BOX_HEIGHT = 370.0
AVG_BOX_PADDING_X = 10.0
AVG_BOX_CORNER_RADIUS = 8.0

# Theme Color Palette (Dracula / Modern Dark)
COLOR_BACKGROUND = "#181824"
COLOR_TITLE_TEXT = "#f8f8f2"
COLOR_SUBTITLE_TEXT = "#9d9eb4"
COLOR_AXIS_TITLE = "#bd93f9"
COLOR_MUTED_TEXT = "#6272a4"
COLOR_GRID_LINE = "#44475a"

# Standard Metric Colors
COLOR_SCALAR_BAR = "#ff5555"
COLOR_SCALAR_LABEL = "#ff79c6"
COLOR_SIMD_BAR = "#50fa7b"
COLOR_SIMD_LABEL = "#50fa7b"
COLOR_SPEEDUP_BADGE = "#8be9fd"
COLOR_METRIC_LABEL = "#f8f8f2"

# Average Column Constants
COLOR_AVG_LABEL = "#50fa7b"
COLOR_AVG_BOX_STROKE = "#6272a4"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_workspace_root() -> Path:
    """Finds the root directory of the vector-rs workspace."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "Cargo.toml").exists() and (current / "crates").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def run_simd_benchmarks(
    workspace_root: Path,
    samples: int = DEFAULT_SAMPLE_SIZE,
    dimension: int | None = DEFAULT_DIMENSION,
    metrics: list[tuple[str, str]] | None = None,
) -> bool:
    """Runs `cargo bench -p vector-simd --bench distance_bench` with dynamic sample
    size, dimension, and metric filter."""
    dim_msg = f"for Dim={dimension} " if dimension else ""
    active_metrics = metrics if metrics is not None else PRIMARY_METRIC_DEFINITIONS
    metric_names = [m[0] for m in active_metrics]
    print(
        f"[*] [1/3] Running SIMD micro-benchmarks ({len(metric_names)} metrics, {dim_msg}{samples:,} samples) via Criterion.rs..."
    )
    cmd = ["cargo", "bench", "-p", "vector-simd", "--bench", "distance_bench"]

    env = os.environ.copy()
    env[ENV_CRITERION_SAMPLE_SIZE] = str(samples)
    if dimension:
        env[ENV_CRITERION_DIMENSION] = str(dimension)
    if metric_names:
        env[ENV_CRITERION_METRICS] = ",".join(metric_names)

    try:
        res = subprocess.run(cmd, cwd=workspace_root, env=env, check=True)
        return res.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[!] Error while running cargo bench: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("[!] Error: 'cargo' binary not found in system PATH.", file=sys.stderr)
        return False


def get_estimate_ns(estimates_path: Path) -> float:
    """Extracts execution time in nanoseconds from Criterion estimates.json."""
    if not estimates_path.exists():
        raise FileNotFoundError(f"Estimates file not found: {estimates_path}")

    with estimates_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # Criterion records slope if throughput is specified, otherwise mean
    if data.get("slope") and data["slope"].get("point_estimate") is not None:
        return float(data["slope"]["point_estimate"])
    if data.get("mean") and data["mean"].get("point_estimate") is not None:
        return float(data["mean"]["point_estimate"])
    if data.get("median") and data["median"].get("point_estimate") is not None:
        return float(data["median"]["point_estimate"])

    raise ValueError(f"No valid timing estimate found in {estimates_path}")


def get_sample_count(sample_path: Path) -> int | None:
    """Extracts measured sample count from Criterion sample.json."""
    if not sample_path.exists():
        return None
    with contextlib.suppress(OSError, json.JSONDecodeError, KeyError, TypeError):
        with sample_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if "times" in data and isinstance(data["times"], list):
            return len(data["times"])
    return None


def collect_simd_metrics(
    workspace_root: Path,
    dimension: int = DEFAULT_DIMENSION,
    default_samples: int = DEFAULT_SAMPLE_SIZE,
    metrics: list[tuple[str, str]] | None = None,
) -> dict:
    """Parses target/criterion JSON estimates for primary metrics and computes the
    overall average."""
    criterion_dir = workspace_root / "target" / "criterion"
    if not criterion_dir.exists():
        raise FileNotFoundError(
            f"Directory target/criterion not found at {criterion_dir}. "
            "Please run the benchmarks first."
        )

    active_metrics = metrics if metrics is not None else PRIMARY_METRIC_DEFINITIONS

    metrics_list: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "dimension": dimension,
        "backend_name": "AVX2 + FMA",
        "sample_count": default_samples,
        "metrics": metrics_list,
        "average": None,
    }

    dim_str = str(dimension)
    detected_samples = None

    for group_name, display_name in active_metrics:
        group_dir = criterion_dir / group_name
        if not group_dir.exists():
            print(f"[!] Warning: Benchmark directory {group_name} not found. Skipping.")
            continue

        # 1. Read Scalar estimate
        scalar_est_file = group_dir / "scalar" / dim_str / "new" / "estimates.json"
        if not scalar_est_file.exists():
            scalar_est_file = group_dir / "scalar" / dim_str / "base" / "estimates.json"
        if not scalar_est_file.exists():
            scalar_est_file = group_dir / dim_str / "scalar" / "new" / "estimates.json"
        if not scalar_est_file.exists():
            scalar_est_file = group_dir / dim_str / "scalar" / "base" / "estimates.json"

        if not scalar_est_file.exists():
            print(
                f"[!] Warning: Scalar estimate for {group_name} (dim={dim_str}) not found."
            )
            continue

        scalar_ns = get_estimate_ns(scalar_est_file)

        # Detect sample count from sample.json
        if detected_samples is None:
            sample_json = scalar_est_file.parent / "sample.json"
            cnt = get_sample_count(sample_json)
            if cnt:
                detected_samples = cnt

        # 2. Find SIMD backend folder (non-scalar, non-numeric subdirectories)
        simd_backend_folder = None
        for item in group_dir.iterdir():
            if (
                item.is_dir()
                and item.name not in ("scalar", "report")
                and not item.name.isdigit()
            ):
                simd_backend_folder = item
                break

        if not simd_backend_folder:
            dim_dir = group_dir / dim_str
            if dim_dir.exists() and dim_dir.is_dir():
                for item in dim_dir.iterdir():
                    if (
                        item.is_dir()
                        and item.name not in ("scalar", "report")
                        and not item.name.isdigit()
                    ):
                        simd_backend_folder = item
                        break

        if not simd_backend_folder:
            print(f"[!] Warning: SIMD backend folder for {group_name} not found.")
            continue

        backend_name_raw = simd_backend_folder.name
        if "avx" in backend_name_raw.lower():
            results["backend_name"] = "AVX2 + FMA"
        elif "neon" in backend_name_raw.lower():
            results["backend_name"] = "ARM NEON"
        else:
            results["backend_name"] = backend_name_raw.upper()

        simd_est_file = simd_backend_folder / dim_str / "new" / "estimates.json"
        if not simd_est_file.exists():
            simd_est_file = simd_backend_folder / dim_str / "base" / "estimates.json"
        if not simd_est_file.exists():
            simd_est_file = simd_backend_folder / "new" / "estimates.json"
        if not simd_est_file.exists():
            simd_est_file = simd_backend_folder / "base" / "estimates.json"
        if not simd_est_file.exists():
            simd_est_file = (
                group_dir / dim_str / backend_name_raw / "new" / "estimates.json"
            )

        if not simd_est_file.exists():
            print(
                f"[!] Warning: SIMD estimate for {group_name} ({simd_backend_folder.name}) not found."
            )
            continue

        simd_ns = get_estimate_ns(simd_est_file)
        speedup = scalar_ns / simd_ns if simd_ns > 0 else 0.0

        metrics_list.append(
            {
                "group": group_name,
                "display_name": display_name,
                "scalar_ns": scalar_ns,
                "simd_ns": simd_ns,
                "speedup": speedup,
                "is_average": False,
            }
        )

    if detected_samples:
        results["sample_count"] = detected_samples

    # Compute overall Average if metrics exist
    if metrics_list:
        count = len(metrics_list)
        avg_scalar = sum(m["scalar_ns"] for m in metrics_list) / count
        avg_simd = sum(m["simd_ns"] for m in metrics_list) / count
        avg_speedup = avg_scalar / avg_simd if avg_simd > 0 else 0.0

        results["average"] = {
            "group": "average",
            "display_name": "Average",
            "scalar_ns": avg_scalar,
            "simd_ns": avg_simd,
            "speedup": avg_speedup,
            "is_average": True,
        }

    return results


# ==============================================================================
# SVG GENERATION
# ==============================================================================


def generate_simd_svg(data: dict, output_path: Path) -> None:
    """Renders the dark-mode vector SVG visualization for 5 metrics plus Average."""
    metrics = list(data["metrics"])
    if not metrics:
        print("[!] No metric data found to plot.", file=sys.stderr)
        return

    # Append Average as the final element
    if data.get("average"):
        metrics.append(data["average"])

    dim = data["dimension"]
    backend_label = data["backend_name"]
    sample_count = data.get("sample_count", DEFAULT_SAMPLE_SIZE)
    num_items = len(metrics)

    center_x = SVG_CANVAS_WIDTH / 2.0
    y_axis_right = SVG_CANVAS_WIDTH - Y_AXIS_RIGHT_MARGIN

    # Calculate max scalar time to set dynamic Y scale
    max_time = max(m["scalar_ns"] for m in metrics) if metrics else Y_MAX_DEFAULT_FLOOR
    y_max = (
        Y_MAX_DEFAULT_FLOOR
        if max_time <= Y_MAX_DEFAULT_FLOOR
        else (int(max_time / Y_MAX_ROUNDING_STEP) + 1) * Y_MAX_ROUNDING_STEP
    )

    # Grid steps
    step = y_max / float(Y_GRID_INTERVAL_COUNT)
    grid_steps = [step * i for i in range(Y_GRID_INTERVAL_COUNT + 1)]

    # Construct Grid lines and labels (with explicit y2 to guarantee perfectly flat horizontal lines)
    grid_lines_xml = []
    grid_lines_xml.append(
        f'  <line x1="{Y_AXIS_LEFT}" y1="{Y_AXIS_BOTTOM}" x2="{y_axis_right}" y2="{Y_AXIS_BOTTOM}" stroke="{COLOR_GRID_LINE}" stroke-width="1.5" />'
    )

    for val in grid_steps[1:]:
        y_pos = Y_AXIS_BOTTOM - (val / y_max) * PLOT_HEIGHT
        grid_lines_xml.append(
            f'  <line x1="{Y_AXIS_LEFT}" y1="{y_pos:.1f}" x2="{y_axis_right}" y2="{y_pos:.1f}" stroke="{COLOR_GRID_LINE}" stroke-width="1" stroke-dasharray="4,4" />'
        )

    grid_labels_xml = []
    for val in grid_steps:
        y_pos = Y_AXIS_BOTTOM - (val / y_max) * PLOT_HEIGHT
        grid_labels_xml.append(
            f'  <text x="{Y_AXIS_LEFT - 12}" y="{y_pos + 4:.1f}" text-anchor="end" fill="{COLOR_MUTED_TEXT}" font-size="12">{int(val)} ns</text>'
        )

    # Construct Bars
    bars_xml = []
    available_width = y_axis_right - Y_AXIS_LEFT - 30.0
    group_width = available_width / max(num_items, 1)
    start_x = (
        Y_AXIS_LEFT
        + BAR_GROUP_INNER_PADDING
        + (group_width - (BAR_WIDTH * 2.0 + BAR_GAP)) / 2.0
    )

    for i, m in enumerate(metrics):
        base_x = start_x + (i * group_width)
        is_avg = m.get("is_average", False)

        # Scalar bar
        scalar_h = (m["scalar_ns"] / y_max) * PLOT_HEIGHT
        scalar_y = Y_AXIS_BOTTOM - scalar_h
        scalar_val_text = f"{round(m['scalar_ns'])} ns"

        # SIMD bar
        simd_h = max((m["simd_ns"] / y_max) * PLOT_HEIGHT, MIN_SIMD_BAR_HEIGHT)
        simd_y = Y_AXIS_BOTTOM - simd_h
        simd_val_text = f"{round(m['simd_ns'])} ns"
        speedup_text = f"{m['speedup']:.1f}x"

        # Center X coordinates
        scalar_cx = base_x + (BAR_WIDTH / 2.0)
        simd_cx = base_x + BAR_WIDTH + BAR_GAP + (BAR_WIDTH / 2.0)
        group_cx = base_x + BAR_WIDTH + (BAR_GAP / 2.0)

        # Styling
        scalar_color = COLOR_SCALAR_BAR
        simd_color = COLOR_SIMD_BAR
        title_color = COLOR_AVG_LABEL if is_avg else COLOR_METRIC_LABEL
        font_weight = "bold"

        avg_box_bg = ""
        if is_avg:
            # Clean dashed bounding frame for Average column
            box_x = base_x - AVG_BOX_PADDING_X
            box_w = BAR_WIDTH * 2.0 + BAR_GAP + (AVG_BOX_PADDING_X * 2.0)
            avg_box_bg = (
                f'  <rect x="{box_x:.1f}" y="{AVG_BOX_TOP_Y:.1f}" width="{box_w:.1f}" height="{AVG_BOX_HEIGHT:.1f}" '
                f'rx="{AVG_BOX_CORNER_RADIUS:.1f}" fill="none" stroke="{COLOR_AVG_BOX_STROKE}" '
                f'stroke-dasharray="3,3" stroke-width="1"/>\n'
            )

        bars_xml.append(
            f"""{avg_box_bg}  <!-- Group {i + 1}: {m["display_name"]} (Scalar: {m["scalar_ns"]:.1f}ns, SIMD: {m["simd_ns"]:.1f}ns, {m["speedup"]:.1f}x) -->
  <rect x="{base_x:.1f}" y="{scalar_y:.1f}" width="{BAR_WIDTH:.1f}" height="{scalar_h:.1f}" rx="{BAR_CORNER_RADIUS:.1f}" fill="{scalar_color}" opacity="0.9"/>
  <text x="{scalar_cx:.1f}" y="{scalar_y - 8:.1f}" text-anchor="middle" fill="{COLOR_SCALAR_LABEL}" font-size="11" font-weight="{font_weight}">{scalar_val_text}</text>

  <rect x="{base_x + BAR_WIDTH + BAR_GAP:.1f}" y="{simd_y:.1f}" width="{BAR_WIDTH:.1f}" height="{simd_h:.1f}" rx="{BAR_CORNER_RADIUS:.1f}" fill="{simd_color}" opacity="0.9"/>
  <text x="{simd_cx:.1f}" y="{simd_y - 8:.1f}" text-anchor="middle" fill="{simd_color}" font-size="11" font-weight="{font_weight}">{simd_val_text}</text>
  <text x="{simd_cx:.1f}" y="{simd_y - 27:.1f}" text-anchor="middle" fill="{COLOR_SPEEDUP_BADGE}" font-size="13" font-weight="bold">{speedup_text}</text>
  <text x="{group_cx:.1f}" y="{Y_AXIS_BOTTOM + 26}" text-anchor="middle" fill="{title_color}" font-size="12" font-weight="{font_weight}">{m["display_name"]}</text>"""
        )

    legend_x = center_x - 130
    legend_simd_x = center_x + 30
    y_axis_mid_y = Y_AXIS_BOTTOM - (PLOT_HEIGHT / 2.0)

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_CANVAS_WIDTH} {SVG_CANVAS_HEIGHT}" style="background-color: {COLOR_BACKGROUND}; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <!-- Title & Subtitle -->
  <text x="{center_x}" y="42" text-anchor="middle" fill="{COLOR_TITLE_TEXT}" font-size="22" font-weight="bold">VectorRS: CPU SIMD Intrinsics Acceleration (Dim = {dim})</text>
  <text x="{center_x}" y="68" text-anchor="middle" fill="{COLOR_SUBTITLE_TEXT}" font-size="13">Comparing Scalar Baseline vs. {backend_label} 256-bit Register Parallelism (Samples = {sample_count:,})</text>

  <!-- Legend -->
  <rect x="{legend_x}" y="88" width="16" height="16" rx="4" fill="{COLOR_SCALAR_BAR}" />
  <text x="{legend_x + 25}" y="101" fill="{COLOR_TITLE_TEXT}" font-size="13" font-weight="600">Scalar Baseline</text>
  
  <rect x="{legend_simd_x}" y="88" width="16" height="16" rx="4" fill="{COLOR_SIMD_BAR}" />
  <text x="{legend_simd_x + 25}" y="101" fill="{COLOR_TITLE_TEXT}" font-size="13" font-weight="600">{backend_label} SIMD</text>

  <!-- Y-Axis Title (Rotated) -->
  <text transform="rotate(-90)" x="{-y_axis_mid_y:.1f}" y="28" text-anchor="middle" fill="{COLOR_AXIS_TITLE}" font-size="12" font-weight="600" letter-spacing="0.5px">Average Execution Time (Lower is Better)</text>

  <!-- Y-Axis Grid & Labels -->
{chr(10).join(grid_lines_xml)}

{chr(10).join(grid_labels_xml)}

{chr(10).join(bars_xml)}

  <!-- X-Axis Title -->
  <text x="{center_x}" y="{Y_AXIS_BOTTOM + 58}" text-anchor="middle" fill="{COLOR_AXIS_TITLE}" font-size="12" font-weight="600" letter-spacing="0.5px">Distance &amp; Similarity Metric (Raw Vector Space)</text>

  <!-- Footer -->
  <text x="{center_x}" y="585" text-anchor="middle" fill="{COLOR_MUTED_TEXT}" font-size="12">Executed via Cargo Criterion Benchmark (black_box loop) | vector-rs</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"[+] [3/3] SVG plot successfully saved to: {output_path}")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Automated SIMD benchmark and plot generator for VectorRS."
    )
    parser.add_argument(
        "--skip-bench",
        "--no-bench",
        action="store_true",
        help="Skip running cargo bench and generate plot from existing Criterion data.",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of statistical measurement samples for Criterion (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=DEFAULT_DIMENSION,
        help=f"Vector dimension to benchmark and plot (default: {DEFAULT_DIMENSION})",
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

    # Determine output file path (defaults to same directory where this script is located)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
    else:
        out_path = script_dir / DEFAULT_OUTPUT_FILENAME

    print("=" * 65)
    print("VectorRS: Automated SIMD Benchmark & Visualization Runner")
    print(f"Workspace:   {workspace_root}")
    print(f"Script Dir:  {script_dir}")
    print(f"Output File: {out_path}")
    print(f"Dimension:   {args.dim}")
    print(f"Sample Size: {args.samples}")
    print("=" * 65)

    # Step 1: Run benchmark if not skipped
    if not args.skip_bench:
        success = run_simd_benchmarks(
            workspace_root,
            samples=args.samples,
            dimension=args.dim,
            metrics=PRIMARY_METRIC_DEFINITIONS,
        )
        if not success:
            print(
                "[!] Benchmark execution failed. Plot generation aborted.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print("[*] [1/3] Skipping cargo bench execution (--skip-bench enabled).")

    # Step 2: Extract Metrics
    print("[*] [2/3] Reading and extracting metrics from target/criterion/...")
    try:
        metrics_data = collect_simd_metrics(
            workspace_root,
            dimension=args.dim,
            default_samples=args.samples,
            metrics=PRIMARY_METRIC_DEFINITIONS,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"[!] Failed to extract metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # Print terminal summary
    print(f"\n--- SIMD BENCHMARK SUMMARY (Dim = {args.dim}) ---")
    print(
        "{:<22} {:>14} {:>14} {:>12}".format(
            "Metric", "Scalar (ns)", "SIMD (ns)", "Speedup"
        )
    )
    print("-" * 66)
    for m in metrics_data["metrics"]:
        print(
            "{:<22} {:>14.1f} {:>14.1f} {:>11.1f}x".format(
                m["display_name"], m["scalar_ns"], m["simd_ns"], m["speedup"]
            )
        )
    if metrics_data.get("average"):
        avg = metrics_data["average"]
        print("=" * 66)
        print(
            "{:<22} {:>14.1f} {:>14.1f} {:>11.1f}x".format(
                avg["display_name"], avg["scalar_ns"], avg["simd_ns"], avg["speedup"]
            )
        )
    print("-" * 66 + "\n")

    # Step 3: Generate SVG
    generate_simd_svg(metrics_data, out_path)


if __name__ == "__main__":
    main()
