"""Create publication-style iteration plots for BO result histories.

Examples
--------
python reports/plot_iteration_metrics.py --problem zaleplon_mpo --methods cowboys nflow dgbo --stride 10
python reports/plot_iteration_metrics.py --problems all --methods cowboys nflow dgbo --stride 10
python reports/plot_iteration_metrics.py --problems osimetrinib_mpo median_2 amlodipine_mpo --methods dgbo nflow3 cowboys --folders gs10_clip20_dgbo_iteration new_vae_10_chain_100_steps_with_stoch_sampling_nflow3_iteration new_vae_10_chain_100_steps_with_stoch_sampling_cowboys_iteration --seeds 1 2 3 4 5 --stride 10
python reports/plot_iteration_metrics.py --problems all --folders gs10_clip20_dgbo_iteration new_vae_10_chain_100_steps_with_stoch_sampling_nflow3_iteration new_vae_10_chain_100_steps_with_stoch_sampling_cowboys_iteration --methods dgbo nflow3 cowboys --seeds 1 2 3 4 5 --stride 10
python reports/plot_iteration_metrics.py --problems all --folders gs10_clip20_dgbo_iteration2 new_vae_10_chain_100_steps_with_stoch_sampling_nflow3_iteration2 new_vae_10_chain_100_steps_with_stoch_sampling_cowboys_iteration2 --methods dgbo nflow3 cowboys --seeds 1 2 3 4 5 --stride 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from string import ascii_lowercase
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
    "cowboys_flow_2": "NFlow 3",
    "cowboys_diffusion": "DGBO",
}

METHOD_ALIASES = {
    "cowboys_flow": {"nflow"},
    "cowboys_flow_2": {"nflow3"},
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

TOP_K_MEAN_METRIC_KEY = "mean_available_unique_top_10_candidate_objective"
TOP_K_COUNT_METRIC_KEY = "available_unique_top_10_candidate_count"
ADJUSTED_TOP_K_METRIC_KEY = "adjusted_available_unique_top_10_candidate_objective"

SCATTER_X_METRIC_KEY = "mean_selected_nearest_previous_tanimoto_distance"
SCATTER_Y_METRIC_KEY = ADJUSTED_TOP_K_METRIC_KEY
SCATTER_TITLE = "Exploration vs Adjusted Candidate Quality"
SCATTER_FILENAME = "plot_4_distance_vs_adjusted_top_k_candidate_objective"
SCATTER_XLABEL = "Avg. nearest-prev. distance"
SCATTER_YLABEL = "Avg. adjusted top-k objective"

MATRIX_GRID_FILENAME = "summary_6_problems_3_metrics_grid"
MATRIX_METRIC_TITLES = {
    "best_so_far_objective": "Best Objective",
    "mean_selected_nearest_previous_tanimoto_distance": "Nearest-Prev. Distance",
    ADJUSTED_TOP_K_METRIC_KEY: "Adjusted Top-k Objective",
}


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
        ylabel="Best objective",
        filename="plot_1_best_so_far_objective",
        yformatter="{x:.3f}",
    ),
    MetricSpec(
        key="mean_selected_nearest_previous_tanimoto_distance",
        title="Distance to Previous Selected Molecules",
        ylabel="Nearest-prev. distance",
        filename="plot_2_nearest_previous_tanimoto_distance",
        ymin=0.0,
        yformatter="{x:.3f}",
    ),
    MetricSpec(
        key=ADJUSTED_TOP_K_METRIC_KEY,
        title="Adjusted Top-k Candidate Objective",
        ylabel="Adjusted top-k objective",
        filename="plot_3_adjusted_top_k_candidate_objective",
        yformatter="{x:.3f}",
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


@dataclass
class PreparedProblemData:
    problem: str
    seeds: list[int]
    aggregates: dict[str, dict[str, AggregatedSeries]]


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
        iteration_suffix = path.name.rsplit("_iteration", 1)
        has_iteration_suffix = (
            path.name.endswith("_iteration")
            or (len(iteration_suffix) == 2 and iteration_suffix[1].isdigit())
        )
        if path.is_dir() and has_iteration_suffix:
            yield path


def iter_folder_files(folder: Path, suffix: str) -> Iterable[Path]:
    with os.scandir(to_long_path(folder)) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.is_file() and entry.name.endswith(suffix):
                yield folder / entry.name


def iter_iteration_result_files(folder: Path) -> Iterable[Path]:
    yield from iter_folder_files(folder, RESULT_FILE_SUFFIX)


def read_json(path: Path) -> dict:
    with open(to_long_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def missing_iteration_results_message(folder: Path) -> str:
    status_files = list(iter_folder_files(folder, "_status.json"))
    if status_files:
        sample_status = read_json(status_files[0])
        if sample_status.get("iteration_results_enabled") is False:
            return (
                f"Folder '{folder.name}' has no '{RESULT_FILE_SUFFIX}' files. "
                "Its status files report 'iteration_results_enabled=false', "
                "so the required iteration metrics were never written."
            )

    return f"Folder '{folder.name}' has no '{RESULT_FILE_SUFFIX}' files."


def build_method_config(
    folder: Path, *, require_iteration_results: bool = False
) -> MethodConfig | None:
    files = list(iter_iteration_result_files(folder))
    if not files:
        if require_iteration_results:
            raise FileNotFoundError(missing_iteration_results_message(folder))
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


def resolve_iteration_folders(
    results_dir: Path, requested_folders: list[str] | None
) -> list[Path]:
    available_folders = {folder.name: folder for folder in iter_iteration_folders(results_dir)}
    if not available_folders:
        raise FileNotFoundError(
            f"No iteration result folders were found under '{results_dir}'."
        )

    if not requested_folders:
        return list(available_folders.values())

    missing_folders = [
        folder_name for folder_name in requested_folders if folder_name not in available_folders
    ]
    if missing_folders:
        available = ", ".join(sorted(available_folders))
        missing = ", ".join(missing_folders)
        raise ValueError(
            f"Unknown folder(s): {missing}. Available iteration folders: {available}."
        )

    return [available_folders[folder_name] for folder_name in requested_folders]


def discover_methods(
    results_dir: Path, requested_folders: list[str] | None = None
) -> list[MethodConfig]:
    methods = [
        method
        for method in (
            build_method_config(
                folder,
                require_iteration_results=requested_folders is not None,
            )
            for folder in resolve_iteration_folders(results_dir, requested_folders)
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


def required_metric_keys_for_plotting() -> set[str]:
    required_metric_keys: set[str] = {"bo_iteration"}
    for metric in METRICS:
        if metric.key == ADJUSTED_TOP_K_METRIC_KEY:
            required_metric_keys.update({TOP_K_MEAN_METRIC_KEY, TOP_K_COUNT_METRIC_KEY})
        else:
            required_metric_keys.add(metric.key)

    return required_metric_keys


def compute_adjusted_top_k_objective(
    mean_top_k_objective: np.ndarray,
    available_unique_top_k_count: np.ndarray,
) -> np.ndarray:
    adjusted = mean_top_k_objective * available_unique_top_k_count / 10.0
    no_available_candidates = (
        np.isfinite(available_unique_top_k_count)
        & (available_unique_top_k_count <= 0)
        & ~np.isfinite(mean_top_k_objective)
    )
    return np.where(no_available_candidates, 0.0, adjusted)


def load_runs(
    methods: list[MethodConfig], explicit_seeds: list[int] | None = None
) -> list[RunRecord]:
    required_metric_keys = required_metric_keys_for_plotting()
    explicit_seed_set = set(explicit_seeds or [])

    runs: list[RunRecord] = []
    for method in methods:
        for path in iter_iteration_result_files(method.folder):
            payload = read_json(path)
            seed = int(payload["seed"])
            if explicit_seed_set and seed not in explicit_seed_set:
                continue

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
            parsed_metric_series[ADJUSTED_TOP_K_METRIC_KEY] = (
                compute_adjusted_top_k_objective(
                    mean_top_k_objective=parsed_metric_series[TOP_K_MEAN_METRIC_KEY],
                    available_unique_top_k_count=parsed_metric_series[TOP_K_COUNT_METRIC_KEY],
                )
            )
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
                    seed=seed,
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
            print(
                "Warning: skipping "
                f"'{method.solver_name}' for '{metric_key}' because it has no finite "
                "values at common plotted iterations."
            )
            continue

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


def configure_metric_axis(
    axis: plt.Axes,
    metric: MetricSpec,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    axis.set_xlabel("BO iteration" if show_xlabel else "")
    axis.set_ylabel(metric.ylabel if show_ylabel else "")
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


def plot_metric_series(
    axis: plt.Axes,
    metric: MetricSpec,
    methods: list[MethodConfig],
    aggregated: dict[str, AggregatedSeries],
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    plotted_any_series = False
    for index, method in enumerate(methods):
        series = aggregated.get(method.solver_name)
        if series is None:
            continue
        plotted_any_series = True

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

    if not plotted_any_series:
        axis.text(
            0.5,
            0.5,
            "No finite values",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#6C6C66",
            fontsize=11,
        )

    configure_metric_axis(
        axis,
        metric,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
    )


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
    plot_metric_series(
        axis,
        metric,
        methods,
        aggregated,
        show_xlabel=True,
        show_ylabel=True,
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

    handles, _ = axis.get_legend_handles_labels()
    if handles:
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
        figure.savefig(to_long_path(destination), dpi=400, bbox_inches="tight")
        output_paths.append(destination)

    plt.close(figure)
    return output_paths


def build_summary_footer(
    prepared_problem_data: list[PreparedProblemData],
    stride: int,
) -> str:
    unique_seed_sets = {tuple(problem_data.seeds) for problem_data in prepared_problem_data}
    if len(unique_seed_sets) == 1:
        seed_text = f"Seeds: {', '.join(map(str, prepared_problem_data[0].seeds))}"
    else:
        seed_text = "Seeds: per-problem common seeds across methods"

    return (
        f"{seed_text}"
        f"   |   Every {stride} iteration(s)"
        f"   |   Full-seed iterations only"
    )


def save_metric_summary_grid(
    metric: MetricSpec,
    methods: list[MethodConfig],
    prepared_problem_data: list[PreparedProblemData],
    output_dir: Path,
    formats: tuple[str, ...],
    stride: int,
) -> list[Path]:
    n_problems = len(prepared_problem_data)
    n_cols = 1 if n_problems == 1 else 2
    n_rows = math.ceil(n_problems / n_cols)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.6 * n_cols, 3.8 * n_rows + 1.0),
        squeeze=False,
    )

    for idx, problem_data in enumerate(prepared_problem_data):
        row, col = divmod(idx, n_cols)
        axis = axes[row][col]
        plot_metric_series(
            axis,
            metric,
            methods,
            problem_data.aggregates[metric.key],
            show_xlabel=row == n_rows - 1,
            show_ylabel=col == 0,
        )
        panel_letter = ascii_lowercase[idx] if idx < len(ascii_lowercase) else f"p{idx + 1}"
        axis.set_title(
            f"({panel_letter}) {prettify_name(problem_data.problem)}",
            loc="left",
            pad=10,
            fontsize=14,
        )

    for idx in range(n_problems, n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(methods)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
            handlelength=2.4,
            columnspacing=1.2,
        )

    figure.suptitle(metric.title, x=0.07, y=0.985, ha="left", fontsize=18)
    figure.text(
        0.5,
        0.085,
        build_summary_footer(prepared_problem_data, stride),
        ha="center",
        va="center",
        fontsize=10,
        color="#6C6C66",
    )
    figure.tight_layout(rect=(0.03, 0.12, 0.995, 0.95))

    summary_dir = output_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for file_format in formats:
        destination = (
            summary_dir
            / f"summary_{len(prepared_problem_data)}_problems_{metric.filename}.{file_format}"
        )
        figure.savefig(to_long_path(destination), dpi=400, bbox_inches="tight")
        output_paths.append(destination)

    plt.close(figure)
    return output_paths


def mean_of_aggregated_series(series: AggregatedSeries | None) -> float | None:
    if series is None:
        return None
    finite_values = series.mean[np.isfinite(series.mean)]
    if len(finite_values) == 0:
        return None
    return float(np.mean(finite_values))


def scatter_points_for_problem(
    methods: list[MethodConfig],
    aggregates: dict[str, dict[str, AggregatedSeries]],
) -> list[tuple[MethodConfig, float, float]]:
    x_series_by_method = aggregates.get(SCATTER_X_METRIC_KEY, {})
    y_series_by_method = aggregates.get(SCATTER_Y_METRIC_KEY, {})

    points: list[tuple[MethodConfig, float, float]] = []
    for method in methods:
        x_value = mean_of_aggregated_series(x_series_by_method.get(method.solver_name))
        y_value = mean_of_aggregated_series(y_series_by_method.get(method.solver_name))
        if x_value is None or y_value is None:
            continue
        points.append((method, x_value, y_value))

    return points


def configure_scatter_axis(
    axis: plt.Axes,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    axis.set_xlabel(SCATTER_XLABEL if show_xlabel else "")
    axis.set_ylabel(SCATTER_YLABEL if show_ylabel else "")
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    axis.grid(axis="both", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9A9A92")
    axis.spines["bottom"].set_color("#9A9A92")


def plot_tradeoff_scatter(
    axis: plt.Axes,
    methods: list[MethodConfig],
    aggregates: dict[str, dict[str, AggregatedSeries]],
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    points = scatter_points_for_problem(methods, aggregates)

    if not points:
        axis.text(
            0.5,
            0.5,
            "No finite values",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#6C6C66",
            fontsize=11,
        )
    else:
        for index, method in enumerate(methods):
            method_points = [
                (x_value, y_value)
                for point_method, x_value, y_value in points
                if point_method.solver_name == method.solver_name
            ]
            if not method_points:
                continue

            x_value, y_value = method_points[0]
            color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
            axis.scatter(
                x_value,
                y_value,
                s=92,
                color=color,
                edgecolors="#1F1F1C",
                linewidths=0.65,
                label=method.label,
                zorder=3,
            )

    configure_scatter_axis(
        axis,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
    )


def save_tradeoff_scatter_plot(
    problem: str,
    methods: list[MethodConfig],
    aggregates: dict[str, dict[str, AggregatedSeries]],
    output_dir: Path,
    formats: tuple[str, ...],
    stride: int,
    seeds: list[int],
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.2, 4.9))
    plot_tradeoff_scatter(
        axis,
        methods,
        aggregates,
        show_xlabel=True,
        show_ylabel=True,
    )

    axis.set_title(SCATTER_TITLE, loc="left", pad=14)
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

    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            loc="best",
            frameon=False,
            ncols=min(3, len(methods)),
            handletextpad=0.45,
            columnspacing=1.0,
        )

    axis.text(
        1.0,
        -0.16,
        (
            f"Seeds: {', '.join(map(str, seeds))}"
            f"   |   Every {stride} iteration(s)"
            f"   |   Mean over plotted iterations"
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
        destination = output_dir / f"{problem}_{SCATTER_FILENAME}.{file_format}"
        figure.savefig(to_long_path(destination), dpi=400, bbox_inches="tight")
        output_paths.append(destination)

    plt.close(figure)
    return output_paths


def save_tradeoff_summary_grid(
    methods: list[MethodConfig],
    prepared_problem_data: list[PreparedProblemData],
    output_dir: Path,
    formats: tuple[str, ...],
    stride: int,
) -> list[Path]:
    n_problems = len(prepared_problem_data)
    n_cols = 1 if n_problems == 1 else 2
    n_rows = math.ceil(n_problems / n_cols)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.6 * n_cols, 3.8 * n_rows + 1.0),
        squeeze=False,
    )

    for idx, problem_data in enumerate(prepared_problem_data):
        row, col = divmod(idx, n_cols)
        axis = axes[row][col]
        plot_tradeoff_scatter(
            axis,
            methods,
            problem_data.aggregates,
            show_xlabel=row == n_rows - 1,
            show_ylabel=col == 0,
        )
        panel_letter = ascii_lowercase[idx] if idx < len(ascii_lowercase) else f"p{idx + 1}"
        axis.set_title(
            f"({panel_letter}) {prettify_name(problem_data.problem)}",
            loc="left",
            pad=10,
            fontsize=14,
        )

    for idx in range(n_problems, n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(methods)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
            handletextpad=0.45,
            columnspacing=1.2,
        )

    figure.suptitle(SCATTER_TITLE, x=0.07, y=0.985, ha="left", fontsize=18)
    figure.text(
        0.5,
        0.085,
        build_summary_footer(prepared_problem_data, stride).replace(
            "Full-seed iterations only", "Mean over plotted iterations"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#6C6C66",
    )
    figure.tight_layout(rect=(0.03, 0.12, 0.995, 0.95))

    summary_dir = output_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for file_format in formats:
        destination = (
            summary_dir
            / f"summary_{len(prepared_problem_data)}_problems_{SCATTER_FILENAME}.{file_format}"
        )
        figure.savefig(to_long_path(destination), dpi=400, bbox_inches="tight")
        output_paths.append(destination)

    plt.close(figure)
    return output_paths


def save_problem_metric_matrix_grid(
    methods: list[MethodConfig],
    prepared_problem_data: list[PreparedProblemData],
    output_dir: Path,
    formats: tuple[str, ...],
    stride: int,
) -> list[Path]:
    n_rows = len(prepared_problem_data)
    n_cols = len(METRICS)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.25 * n_cols, 2.35 * n_rows + 1.3),
        squeeze=False,
    )

    for row, problem_data in enumerate(prepared_problem_data):
        for col, metric in enumerate(METRICS):
            axis = axes[row][col]
            plot_metric_series(
                axis,
                metric,
                methods,
                problem_data.aggregates[metric.key],
                show_xlabel=row == n_rows - 1,
                show_ylabel=col == 0,
            )

            if row == 0:
                axis.set_title(
                    MATRIX_METRIC_TITLES.get(metric.key, metric.title),
                    loc="left",
                    pad=8,
                    fontsize=13,
                )
            else:
                axis.set_title("")

            if col == 0:
                axis.text(
                    -0.32,
                    0.5,
                    prettify_name(problem_data.problem),
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=11,
                    color="#1F1F1C",
                )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(methods)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
            handlelength=2.4,
            columnspacing=1.2,
        )

    figure.suptitle("Iteration Metrics by Problem", x=0.07, y=0.985, ha="left", fontsize=18)
    figure.text(
        0.5,
        0.085,
        build_summary_footer(prepared_problem_data, stride),
        ha="center",
        va="center",
        fontsize=10,
        color="#6C6C66",
    )
    figure.tight_layout(rect=(0.11, 0.12, 0.995, 0.95))

    summary_dir = output_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for file_format in formats:
        destination = summary_dir / f"{MATRIX_GRID_FILENAME}.{file_format}"
        figure.savefig(to_long_path(destination), dpi=400, bbox_inches="tight")
        output_paths.append(destination)

    plt.close(figure)
    return output_paths


def print_available_options(methods: list[MethodConfig], runs: list[RunRecord]) -> None:
    problems = sorted({run.function_name for run in runs})
    print("Available problems:")
    for problem in problems:
        print(f"  - {problem}")

    print("\nFolders with iteration JSONs:")
    for method in methods:
        print(f"  - {method.folder.name}")

    print("\nAvailable methods:")
    for method in methods:
        alias_text = ", ".join(sorted(method.aliases))
        print(
            f"  - {method.solver_name} ({method.label}) "
            f"[{alias_text}] from {method.folder.name}"
        )


def resolve_selected_problems(
    requested_problem: str | None,
    requested_problems: list[str] | None,
    available_problems: set[str],
) -> list[str]:
    if requested_problem and requested_problems:
        raise ValueError("Use either --problem or --problems, not both.")

    if requested_problems:
        if len(requested_problems) == 1 and normalize_token(requested_problems[0]) == "all":
            return sorted(available_problems)
        selected = list(dict.fromkeys(requested_problems))
    elif requested_problem:
        selected = [requested_problem]
    else:
        return []

    missing = [problem for problem in selected if problem not in available_problems]
    if missing:
        available = ", ".join(sorted(available_problems))
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Problem(s) '{missing_text}' were not found. Available problems: {available}"
        )

    return selected


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
        help="Single problem/function name to plot, for example 'osimetrinib_mpo'.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        help="One or more problems to plot together. Use 'all' to include every available problem.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional subset of methods to compare. Accepts aliases like 'cowboys', 'nflow', 'nflow3', and 'dgbo'.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        help="Optional exact result folder names under results/ to use for this comparison.",
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

    methods = discover_methods(args.results_dir, args.folders)
    selected_methods = resolve_methods(args.methods, methods)
    runs = load_runs(selected_methods, args.seeds)
    available_problems = {run.function_name for run in runs}
    selected_problem_names = resolve_selected_problems(
        args.problem,
        args.problems,
        available_problems,
    )

    if args.list_problems or not selected_problem_names:
        print_available_options(selected_methods, runs)
        if not selected_problem_names:
            return 0

    prepared_problem_data: list[PreparedProblemData] = []
    saved_paths: list[Path] = []
    for problem_name in selected_problem_names:
        problem_runs = [run for run in runs if run.function_name == problem_name]
        selected_seeds = choose_seeds(
            runs=problem_runs,
            methods=selected_methods,
            explicit_seeds=args.seeds,
            seed_policy=args.seed_policy,
        )
        filtered_runs = [run for run in problem_runs if run.seed in selected_seeds]

        if not filtered_runs:
            raise ValueError(
                f"No runs matched the chosen problem, methods, and seeds for '{problem_name}'."
            )

        output_dir = args.output_dir / problem_name
        output_dir.mkdir(parents=True, exist_ok=True)

        aggregates_for_problem: dict[str, dict[str, AggregatedSeries]] = {}
        for metric in METRICS:
            aggregated = aggregate_metric(
                runs=filtered_runs,
                methods=selected_methods,
                metric_key=metric.key,
                stride=args.stride,
            )
            aggregates_for_problem[metric.key] = aggregated
            saved_paths.extend(
                save_metric_plot(
                    problem=problem_name,
                    metric=metric,
                    methods=selected_methods,
                    aggregated=aggregated,
                    output_dir=output_dir,
                    formats=tuple(args.formats),
                    stride=args.stride,
                    seeds=selected_seeds,
                )
            )

        saved_paths.extend(
            save_tradeoff_scatter_plot(
                problem=problem_name,
                methods=selected_methods,
                aggregates=aggregates_for_problem,
                output_dir=output_dir,
                formats=tuple(args.formats),
                stride=args.stride,
                seeds=selected_seeds,
            )
        )

        prepared_problem_data.append(
            PreparedProblemData(
                problem=problem_name,
                seeds=selected_seeds,
                aggregates=aggregates_for_problem,
            )
        )

    if len(prepared_problem_data) > 1:
        for metric in METRICS:
            saved_paths.extend(
                save_metric_summary_grid(
                    metric=metric,
                    methods=selected_methods,
                    prepared_problem_data=prepared_problem_data,
                    output_dir=args.output_dir,
                    formats=tuple(args.formats),
                    stride=args.stride,
                )
            )
        saved_paths.extend(
            save_tradeoff_summary_grid(
                methods=selected_methods,
                prepared_problem_data=prepared_problem_data,
                output_dir=args.output_dir,
                formats=tuple(args.formats),
                stride=args.stride,
            )
        )
        saved_paths.extend(
            save_problem_metric_matrix_grid(
                methods=selected_methods,
                prepared_problem_data=prepared_problem_data,
                output_dir=args.output_dir,
                formats=tuple(args.formats),
                stride=args.stride,
            )
        )

    print(f"Problems: {', '.join(selected_problem_names)}")
    print(f"Methods: {', '.join(method.label for method in selected_methods)}")
    for problem_data in prepared_problem_data:
        print(f"Seeds for {problem_data.problem}: {', '.join(map(str, problem_data.seeds))}")
    print(f"Base output directory: {args.output_dir}")
    if len(prepared_problem_data) > 1:
        print(f"Summary directory: {args.output_dir / '_summary'}")
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
