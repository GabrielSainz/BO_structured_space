"""Create publication-style iteration plots for BO result histories.

Examples
--------

python reports/plot_iteration_metrics.py --problem osimetrinib_mpo --methods cowboys nflow --seeds 1 2 3 4 5 --stride 10
python reports/plot_iteration_metrics.py --problem median_2 --methods cowboys nflow --seeds 1 2 3 4 5 --stride 10
python reports/plot_iteration_metrics.py --problem amlodipine_mpo --methods cowboys nflow --seeds 1 2 3 4 5 --stride 10
python reports/plot_iteration_metrics.py --problem perindopril_mpo --methods cowboys nflow --seeds 1 2 3 4 5 --stride 10
python reports/plot_iteration_metrics.py --problem ranolazine_mpo --methods cowboys nflow --seeds 1 2 3 4 5 --stride 10
python reports/plot_iteration_metrics.py --problem zaleplon_mpo --methods cowboys nflow --seeds 1 2 3 4 5 --stride 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


RESULT_FILE_SUFFIX = "_iteration_results.json"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("reports") / "iteration_plots"
DEFAULT_FIGURE_FORMATS = ("png", "pdf")

METHOD_LABELS = {
    "cowboys": "COWBOYS",
    "cowboys_flow": "NFlow",
    "cowboys_diffusion": "DGBO",
}

METHOD_ALIASES = {
    "cowboys_flow": {"nflow"},
    "cowboys_diffusion": {"dgbo"},
}

COLOR_CYCLE = (
    "#FF8811",
    "#392F5A",
    "#5DA271",
    "#F25C54",
    "#2A9D8F",
    "#6D597A",
    "#3A5A40",
    "#577590",
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    filename: str
    ymin: float | None = None
    yformatter: str | None = None


METRICS = (
    MetricSpec(
        key="best_so_far_objective",
        title="Best-so-Far Objective",
        ylabel="Best objective value found so far",
        filename="plot_1_best_so_far_objective",
        yformatter="{x:.3f}",
    ),
    MetricSpec(
        key="mean_selected_nearest_previous_tanimoto_distance",
        title="Distance to Previous Selected Molecules",
        ylabel="Nearest-previous Tanimoto distance",
        filename="plot_2_nearest_previous_tanimoto_distance",
        ymin=0.0,
        yformatter="{x:.3f}",
    ),
    MetricSpec(
        key="sample_unique_decoded_molecules_in_iteration",
        title="Distinct Valid Molecules",
        ylabel="Distinct decoded molecules in iteration",
        filename="plot_3_distinct_valid_molecules",
        ymin=0.0,
        yformatter="{x:.0f}",
    ),
)


@dataclass(frozen=True)
class MethodConfig:
    solver_name: str
    label: str
    folder: Path
    aliases: tuple[str, ...]


@dataclass
class RunRecord:
    solver_name: str
    function_name: str
    seed: int
    source_path: Path
    metric_series: dict[str, np.ndarray]


@dataclass(frozen=True)
class AggregatedSeries:
    iterations: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_runs: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def normalize_token(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def to_long_path(path: Path) -> str:
    if path.is_absolute():
        path_str = os.path.normpath(str(path))
    else:
        path_str = os.path.normpath(str(Path.cwd() / path))

    if os.name != "nt" or path_str.startswith("\\\\?\\"):
        return path_str
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str.lstrip("\\")
    return "\\\\?\\" + path_str


def prettify_name(value: str) -> str:
    return value.replace("_", " ").title()


def iter_iteration_folders(results_dir: Path) -> Iterable[Path]:
    for path in sorted(results_dir.iterdir()):
        if path.is_dir() and path.name.endswith("_iteration"):
            yield path


def iter_iteration_result_files(folder: Path) -> Iterable[Path]:
    with os.scandir(to_long_path(folder)) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.is_file() and entry.name.endswith(RESULT_FILE_SUFFIX):
                yield folder / entry.name


def read_json(path: Path) -> dict:
    with open(to_long_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_method_config(folder: Path) -> MethodConfig | None:
    files = list(iter_iteration_result_files(folder))
    if not files:
        return None

    sample = read_json(files[0])
    solver_name = str(sample["solver_name"])
    label = METHOD_LABELS.get(solver_name, prettify_name(solver_name))

    aliases = {
        normalize_token(solver_name),
        normalize_token(label),
    }
    aliases.update(normalize_token(alias) for alias in METHOD_ALIASES.get(solver_name, set()))

    return MethodConfig(
        solver_name=solver_name,
        label=label,
        folder=folder,
        aliases=tuple(sorted(alias for alias in aliases if alias)),
    )


def discover_methods(results_dir: Path) -> list[MethodConfig]:
    methods = [
        method
        for method in (
            build_method_config(folder)
            for folder in iter_iteration_folders(results_dir)
        )
        if method is not None
    ]

    if not methods:
        raise FileNotFoundError(
            f"No iteration result folders were found under '{results_dir}'."
        )

    return methods


def resolve_methods(
    requested_methods: list[str] | None, available_methods: list[MethodConfig]
) -> list[MethodConfig]:
    if not requested_methods:
        return available_methods

    alias_map: dict[str, MethodConfig] = {}
    for method in available_methods:
        for alias in method.aliases:
            alias_map[alias] = method

    selected_methods: list[MethodConfig] = []
    seen_solver_names: set[str] = set()

    for requested in requested_methods:
        normalized = normalize_token(requested)
        method = alias_map.get(normalized)
        if method is None:
            available = ", ".join(sorted(method.solver_name for method in available_methods))
            raise ValueError(
                f"Unknown method '{requested}'. Available methods: {available}."
            )
        if method.solver_name not in seen_solver_names:
            selected_methods.append(method)
            seen_solver_names.add(method.solver_name)

    return selected_methods


def load_runs(methods: list[MethodConfig]) -> list[RunRecord]:
    required_metric_keys = {metric.key for metric in METRICS}
    required_metric_keys.add("bo_iteration")

    runs: list[RunRecord] = []
    for method in methods:
        for path in iter_iteration_result_files(method.folder):
            payload = read_json(path)
            metric_series = payload.get("metric_series", {})
            missing_metrics = sorted(
                key for key in required_metric_keys if key not in metric_series
            )
            if missing_metrics:
                missing_text = ", ".join(missing_metrics)
                raise KeyError(f"Missing metrics [{missing_text}] in '{path.name}'.")

            parsed_metric_series = {
                key: np.asarray(metric_series[key], dtype=float)
                for key in required_metric_keys
            }
            iterations = parsed_metric_series["bo_iteration"]

            for metric_key, values in parsed_metric_series.items():
                if metric_key == "bo_iteration":
                    continue
                if len(values) != len(iterations):
                    raise ValueError(
                        f"Mismatched series lengths in '{path.name}' for '{metric_key}'."
                    )

            runs.append(
                RunRecord(
                    solver_name=str(payload["solver_name"]),
                    function_name=str(payload["function_name"]),
                    seed=int(payload["seed"]),
                    source_path=path,
                    metric_series=parsed_metric_series,
                )
            )

    if not runs:
        raise FileNotFoundError("No iteration result JSON files were found.")

    return runs


def choose_seeds(
    runs: list[RunRecord],
    methods: list[MethodConfig],
    explicit_seeds: list[int] | None,
    seed_policy: str,
) -> list[int]:
    seeds_per_method: dict[str, set[int]] = defaultdict(set)
    for run in runs:
        seeds_per_method[run.solver_name].add(run.seed)

    if explicit_seeds:
        missing: list[str] = []
        for method in methods:
            missing_seeds = [
                seed for seed in explicit_seeds if seed not in seeds_per_method[method.solver_name]
            ]
            if missing_seeds:
                missing.append(f"{method.solver_name}: {missing_seeds}")

        if missing:
            missing_text = "; ".join(missing)
            raise ValueError(
                f"Some requested seeds are missing from the selected methods: {missing_text}"
            )
        return sorted(dict.fromkeys(explicit_seeds))

    all_seed_sets = [seeds_per_method[method.solver_name] for method in methods]
    if seed_policy == "intersection":
        selected = sorted(set.intersection(*all_seed_sets))
    else:
        selected = sorted(set.union(*all_seed_sets))

    if not selected:
        raise ValueError("No seeds are available for the selected methods.")

    return selected


def sampling_indices(length: int, stride: int) -> np.ndarray:
    indices = np.arange(0, length, stride, dtype=int)
    if len(indices) == 0 or indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return np.unique(indices)


def aggregate_metric(
    runs: list[RunRecord],
    methods: list[MethodConfig],
    metric_key: str,
    stride: int,
) -> dict[str, AggregatedSeries]:
    aggregated: dict[str, AggregatedSeries] = {}

    for method in methods:
        method_runs = [run for run in runs if run.solver_name == method.solver_name]
        if not method_runs:
            continue

        sampled_runs: list[dict[int, float]] = []
        for run in method_runs:
            iterations = run.metric_series["bo_iteration"]
            values = run.metric_series[metric_key]
            idx = sampling_indices(len(iterations), stride)
            sampled_points = {
                int(iteration): float(value)
                for iteration, value in zip(iterations[idx], values[idx])
                if np.isfinite(value)
            }
            sampled_runs.append(sampled_points)

        common_iterations = set(sampled_runs[0])
        for sampled_points in sampled_runs[1:]:
            common_iterations &= set(sampled_points)

        if not common_iterations:
            raise ValueError(
                f"No common plotted iterations were found for method '{method.solver_name}'."
            )

        sorted_iterations = np.asarray(sorted(common_iterations), dtype=int)
        means = np.asarray(
            [
                np.mean([sampled_points[int(iteration)] for sampled_points in sampled_runs])
                for iteration in sorted_iterations
            ]
        )
        stds = np.asarray(
            [
                np.std([sampled_points[int(iteration)] for sampled_points in sampled_runs])
                for iteration in sorted_iterations
            ]
        )

        aggregated[method.solver_name] = AggregatedSeries(
            iterations=sorted_iterations,
            mean=means,
            std=stds,
            n_runs=len(method_runs),
        )

    return aggregated


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCF9",
            "axes.facecolor": "#FCFCF9",
            "savefig.facecolor": "#FCFCF9",
            "axes.edgecolor": "#8F8F88",
            "axes.labelcolor": "#1F1F1C",
            "axes.titlecolor": "#1F1F1C",
            "xtick.color": "#3F3F39",
            "ytick.color": "#3F3F39",
            "grid.color": "#DDDDD6",
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


def save_metric_plot(
    problem: str,
    metric: MetricSpec,
    methods: list[MethodConfig],
    aggregated: dict[str, AggregatedSeries],
    output_dir: Path,
    formats: tuple[str, ...],
    stride: int,
    seeds: list[int],
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.2, 4.9))

    for index, method in enumerate(methods):
        series = aggregated.get(method.solver_name)
        if series is None:
            continue

        color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
        axis.plot(
            series.iterations,
            series.mean,
            color=color,
            linewidth=2.4,
            label=method.label,
            solid_capstyle="round",
        )
        axis.fill_between(
            series.iterations,
            series.mean - series.std,
            series.mean + series.std,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    axis.set_title(metric.title, loc="left", pad=14)
    axis.text(
        0.0,
        1.01,
        prettify_name(problem),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#66665F",
    )
    axis.set_xlabel("BO iteration")
    axis.set_ylabel(metric.ylabel)
    axis.set_xlim(left=1)

    if metric.ymin is not None:
        _, ymax = axis.get_ylim()
        axis.set_ylim(bottom=metric.ymin, top=ymax)

    if metric.yformatter is not None:
        axis.yaxis.set_major_formatter(StrMethodFormatter(metric.yformatter))

    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    axis.grid(axis="y", linewidth=0.9)
    axis.grid(axis="x", linewidth=0.5, alpha=0.4)
    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9A9A92")
    axis.spines["bottom"].set_color("#9A9A92")

    legend = axis.legend(
        loc="upper left",
        frameon=False,
        ncols=min(3, len(methods)),
        handlelength=2.6,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.6)

    axis.text(
        1.0,
        -0.16,
        (
            f"Seeds: {', '.join(map(str, seeds))}"
            f"   |   Every {stride} iteration(s)"
            f"   |   Full-seed iterations only"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#6C6C66",
    )

    figure.tight_layout()

    output_paths: list[Path] = []
    for file_format in formats:
        destination = output_dir / f"{problem}_{metric.filename}.{file_format}"
        figure.savefig(destination, dpi=400, bbox_inches="tight")
        output_paths.append(destination)

    plt.close(figure)
    return output_paths


def print_available_options(methods: list[MethodConfig], runs: list[RunRecord]) -> None:
    problems = sorted({run.function_name for run in runs})
    print("Available problems:")
    for problem in problems:
        print(f"  - {problem}")

    print("\nAvailable methods:")
    for method in methods:
        alias_text = ", ".join(sorted(method.aliases))
        print(f"  - {method.solver_name} ({method.label}) [{alias_text}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis-ready iteration plots from BO result histories."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the method folders that end with '_iteration'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figures will be written.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Problem/function name to plot, for example 'osimetrinib_mpo'.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional subset of methods to compare. Accepts aliases like 'cowboys', 'nflow', and 'dgbo'.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional explicit seed list. If omitted, the script uses the default seed policy.",
    )
    parser.add_argument(
        "--seed-policy",
        choices=("intersection", "union"),
        default="intersection",
        help="How to choose seeds when --seeds is omitted. Default: intersection.",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=1,
        help="Plot every Nth BO iteration. The final iteration is always kept.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FIGURE_FORMATS),
        help="Output formats to save. Default: png pdf.",
    )
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="Print the available problems and methods, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    methods = discover_methods(args.results_dir)
    selected_methods = resolve_methods(args.methods, methods)
    runs = load_runs(selected_methods)

    if args.list_problems or not args.problem:
        print_available_options(selected_methods, runs)
        if not args.problem:
            return 0

    available_problems = {run.function_name for run in runs}
    if args.problem not in available_problems:
        available = ", ".join(sorted(available_problems))
        raise ValueError(
            f"Problem '{args.problem}' was not found. Available problems: {available}"
        )

    problem_runs = [run for run in runs if run.function_name == args.problem]
    selected_seeds = choose_seeds(
        runs=problem_runs,
        methods=selected_methods,
        explicit_seeds=args.seeds,
        seed_policy=args.seed_policy,
    )
    filtered_runs = [run for run in problem_runs if run.seed in selected_seeds]

    if not filtered_runs:
        raise ValueError("No runs matched the chosen problem, methods, and seeds.")

    output_dir = args.output_dir / args.problem
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for metric in METRICS:
        aggregated = aggregate_metric(
            runs=filtered_runs,
            methods=selected_methods,
            metric_key=metric.key,
            stride=args.stride,
        )
        saved_paths.extend(
            save_metric_plot(
                problem=args.problem,
                metric=metric,
                methods=selected_methods,
                aggregated=aggregated,
                output_dir=output_dir,
                formats=tuple(args.formats),
                stride=args.stride,
                seeds=selected_seeds,
            )
        )

    print(f"Problem: {args.problem}")
    print(f"Methods: {', '.join(method.label for method in selected_methods)}")
    print(f"Seeds: {', '.join(map(str, selected_seeds))}")
    print(f"Output directory: {output_dir}")
    print("Saved figures:")
    for path in saved_paths:
        print(f"  - {path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command line error handling
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
