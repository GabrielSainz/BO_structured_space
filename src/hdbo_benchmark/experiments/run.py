"""
This script is an entry_point for all experiments.

For documentation on how to run each individual experiment,
please refer to the README.md.
"""

# mypy: disable-error-code="import-untyped"
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import click
import numpy as np
import torch
from poli.core.exceptions import BudgetExhaustedException
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.core.data_package import DataPackage
from hdbo_benchmark.utils.experiments.load_generative_models import (
    load_generative_model_and_bounds,
)
from hdbo_benchmark.generative_models.latent_diffusion import (
    resolve_default_latent_diffusion_checkpoint,
)
from hdbo_benchmark.benchmarks.load_problems import load_problem
from hdbo_benchmark.solvers.registry import (
    CONTINUOUS_SPACE_SOLVERS,
    SOLVER_NAMES,
    load_solver_from_problem,
)
from hdbo_benchmark.benchmarks.problem_transformations import (
    transform_problem_from_discrete_to_continuous,
)
from hdbo_benchmark.utils.experiments.verify_status_pre_experiment import (
    verify_repos_are_clean,
)
from hdbo_benchmark.utils.logging.iteration_metrics import (
    build_iteration_metric_artifacts,
)
from hdbo_benchmark.utils.logging.idempotence_of_experiments import (
    experiment_has_already_run,
)
from hdbo_benchmark.utils.logging.wandb_observer import ObserverConfig
from hdbo_benchmark.utils.logging.gp_predictions import (
    PREDICTION_CONVENTIONS,
    attach_gp_prediction_recorder,
)

ITERATION_METRIC_SOLVERS = {
    "afgg_seqflow",
    "cowboys",
    "cowboys_flow",
    "cowboys_flow_2",
    "cowboys_diffusion",
    "lsbo",
}


def _format_parameter_for_path(value: float | int | str | None) -> str:
    if value is None:
        return "default"
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return str(value).replace("/", "_")


def _build_output_dir(
    solver_name: str,
    sufix: str,
    diffusion_config: dict[str, float | int | str | None],
) -> Path:
    if solver_name in {"cowboys", "afgg_seqflow"}:
        return Path("./results") / solver_name / sufix
    if solver_name != "cowboys_diffusion":
        return Path(f"./results/new_vae_10_chain_100_steps_with_stoch_sampling_{sufix}")

    components = [
        f"{name}-{_format_parameter_for_path(value)}"
        for name, value in diffusion_config.items()
    ]
    if sufix:
        components.append(f"tag-{_format_parameter_for_path(sufix)}")

    return Path("./results/diffusion") / "__".join(components)


def _path_for_io(path: Path) -> str | Path:
    resolved_path = path.resolve()
    path_str = str(resolved_path)
    if os.name != "nt" or len(path_str) < 240:
        return resolved_path

    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str.lstrip("\\")
    return "\\\\?\\" + path_str


def _write_with_retries(
    path: Path,
    writer,
    description: str,
    attempts: int = 6,
) -> bool:
    """Write through a temporary file, retrying transient Windows file locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        tmp_path = path.with_name(
            f".{path.stem}.{os.getpid()}.{uuid4().hex}{path.suffix}"
        )
        try:
            writer(tmp_path)
            os.replace(_path_for_io(tmp_path), _path_for_io(path))
            return True
        except (PermissionError, OSError) as exc:
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

            if attempt == attempts:
                break
            time.sleep(min(0.25 * attempt, 2.0))

    print(f"Warning: could not write {description} to {path}: {last_error}")
    return False


def _write_json(path: Path, payload: dict) -> None:
    with open(_path_for_io(path), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _save_solver_progress(
    solver,
    output_dir: Path,
    solver_name: str,
    function_name: str,
    seed: int,
    completed_iterations: int,
    status: str,
    initial_history_size: int,
    save_iteration_results: bool,
    evaluate_top_candidate_objectives: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_performance = np.asarray(solver.get_best_performance())
    best_path = output_dir / f"{solver_name}_{function_name}_{seed}.npy"
    _write_with_retries(
        best_path,
        lambda tmp_path: np.save(_path_for_io(tmp_path), best_performance),
        "best performance checkpoint",
    )
    history_path = output_dir / f"{solver_name}_{function_name}_{seed}_history.npz"
    iteration_results_path = (
        output_dir / f"{solver_name}_{function_name}_{seed}_iteration_results.json"
    )
    status_path = output_dir / f"{solver_name}_{function_name}_{seed}_status.json"

    iteration_results_requested = bool(
        save_iteration_results and solver_name in ITERATION_METRIC_SOLVERS
    )
    iteration_results_written = False
    history_written = False
    if hasattr(solver, "get_history_as_arrays"):
        try:
            history_x, history_y = solver.get_history_as_arrays()
            history_payload = {
                "x": np.asarray(history_x),
                "y": np.asarray(history_y),
                "best_performance": best_performance,
            }
            if getattr(solver, "observed_structures", None) is not None:
                history_payload["structures"] = np.asarray(solver.observed_structures)

            if iteration_results_requested:
                try:
                    iteration_json_payload, iteration_series = (
                        build_iteration_metric_artifacts(
                            solver=solver,
                            history_x=np.asarray(history_x),
                            history_y=np.asarray(history_y),
                            initial_history_size=initial_history_size,
                            completed_iterations=completed_iterations,
                            evaluate_top_candidate_objectives=(
                                evaluate_top_candidate_objectives and status != "running"
                            ),
                        )
                    )
                    history_payload.update(iteration_series)
                    iteration_results_payload = {
                        "solver_name": solver_name,
                        "function_name": function_name,
                        "seed": int(seed),
                        **iteration_json_payload,
                    }
                    iteration_results_written = _write_with_retries(
                        iteration_results_path,
                        lambda tmp_path: _write_json(
                            tmp_path,
                            iteration_results_payload,
                        ),
                        "iteration metrics",
                    )
                except Exception as exc:
                    print(f"Warning: could not save iteration metrics: {exc}")

            history_written = _write_with_retries(
                history_path,
                lambda tmp_path: np.savez_compressed(
                    _path_for_io(tmp_path),
                    **history_payload,
                ),
                "history snapshot",
            )
        except Exception as exc:
            print(f"Warning: could not save full history snapshot: {exc}")

    best_value = None
    if best_performance.size > 0 and np.isfinite(best_performance).any():
        best_value = float(np.nanmax(best_performance))

    # Separate additive sidecar: legacy checkpoint/iteration formats remain intact.
    predictions = getattr(solver, "pre_evaluation_prediction_history", None)
    predictions_path = output_dir / f"{solver_name}_{function_name}_{seed}_predictions.json"
    predictions_written = False
    if predictions is not None:
        predictions_payload = {
            "schema_version": "1.0.0", "solver_name": solver_name,
            "function_name": function_name, "seed": int(seed),
            "conventions": PREDICTION_CONVENTIONS, "predictions": predictions,
        }
        predictions_written = _write_with_retries(
            predictions_path, lambda tmp_path: _write_json(tmp_path, predictions_payload),
            "pre-evaluation GP predictions",
        )

    status_payload = {
        "solver_name": solver_name,
        "function_name": function_name,
        "seed": int(seed),
        "initial_history_size": int(initial_history_size),
        "status": status,
        "completed_iterations": int(completed_iterations),
        "best_value": best_value,
        "history_file": history_path.name if history_written else None,
        "iteration_results_enabled": iteration_results_requested,
        "iteration_results_file": (
            iteration_results_path.name if iteration_results_written else None
        ),
        "diagnostic_objective_evaluations_enabled": evaluate_top_candidate_objectives,
    }
    if predictions is not None:
        status_payload["pre_evaluation_predictions_enabled"] = True
        status_payload["predictions_file"] = predictions_path.name if predictions_written else None
    if hasattr(solver, "get_run_configuration"):
        status_payload["solver_configuration"] = solver.get_run_configuration()
    _write_with_retries(
        status_path,
        lambda tmp_path: _write_json(tmp_path, status_payload),
        "status checkpoint",
    )


def _main(
    function_name: str,
    solver_name: str,
    n_dimensions: int,
    seed: int,
    max_iter: int,
    strict_on_hash: bool,
    force_run: bool,
    wandb_mode: str = "disabled",
    tag: str = "default",
    sufix: str = "default",
    diffusion_checkpoint_path: str | None = None,
    num_candidates: int | None = None,
    distillation_n: int | None = None,
    guidance_scale: float | None = None,
    clip_guidance: float | None = None,
    guide_every: int | None = None,
    guidance_alpha_bar_lower: float | None = None,
    guidance_alpha_bar_upper: float | None = None,
    diffusion_eta: float | None = None,
    save_iteration_results: bool = True,
    checkpoint_every: int = 10,
    evaluate_top_candidate_objectives: bool = False,
    afgg_root: Path | None = None,
    afgg_nfbo_root: Path | None = None,
    afgg_checkpoint: Path | None = None,
):
    checkpoint_every = max(1, int(checkpoint_every))

    # Defining a unique experiment id
    experiment_id = f"{uuid4()}"

    # Checking if there are uncommitted changes in the repositories
    verify_repos_are_clean(strict_on_hash)

    # Checking if this experimenr has already been run
    if (
        not force_run
        and wandb_mode == "online"
        and experiment_has_already_run(
            experiment_name="hdbo_benchmark_results",
            solver_name=solver_name,
            function_name=function_name,
            n_dimensions=n_dimensions,
            seed=seed,
        )
    ):
        print(
            f"The experiment for solver {solver_name} with function "
            f"{function_name} and n_dimensions {n_dimensions} "
            f" and seed {seed} has already been run."
        )
        return

    # Seeding
    if seed is None:
        seed = np.random.randint(0, 10_000)

    seed_python_numpy_and_torch(seed)

    problem_kwargs, afgg_solver_kwargs = {}, {}
    if solver_name == "afgg_seqflow":
        from hdbo_benchmark.solvers.afgg_seqflow import SeqFlowReference
        from hdbo_benchmark.utils.constants import DEVICE

        if evaluate_top_candidate_objectives:
            raise click.UsageError("AFGG metrics use the budgeted objective observations and GP candidate scores")
        output_dir = _build_output_dir(solver_name, sufix, {})
        output_stem = f"{solver_name}_{function_name}_{seed}"
        if any(output_dir.glob(f"{output_stem}.*")) or any(output_dir.glob(f"{output_stem}_*")):
            raise FileExistsError(f"AFGG run artifacts already exist in {output_dir}; use a fresh --sufix")
        reference = SeqFlowReference(afgg_root, afgg_nfbo_root, afgg_checkpoint, DEVICE)
        initial_smiles, initial_selfies, initial_latents = reference.initial_design(seed)
        n_dimensions = reference.flow.seq_len * reference.flow.zsize
        problem_kwargs["initial_smiles"] = initial_smiles
        afgg_solver_kwargs.update(reference=reference, initial_selfies=initial_selfies,
                                 initial_latents=initial_latents)

    # Setting the observer configuration
    observer_config = ObserverConfig(
        experiment_name="hdbo_benchmark_results",
        function_name=function_name,
        solver_name=solver_name,
        n_dimensions=n_dimensions,
        seed=seed,
        max_iter=max_iter,
        strict_on_hash=strict_on_hash,
        force_run=force_run,
        experiment_id=experiment_id,
        wandb_mode=wandb_mode,
        tags=[tag],
    )

    # Load the problem
    problem = load_problem(
        function_name=function_name,
        max_iter=max_iter,
        set_observer=True,
        observer_config=observer_config,
        **problem_kwargs,
    )
    print(problem)

    if solver_name in CONTINUOUS_SPACE_SOLVERS:
        # Load the generative model
        generative_model, bounds = load_generative_model_and_bounds(
            function_name=function_name,
            latent_dim=n_dimensions,
            problem=problem,
        )

        # Make the problem continuous
        problem = transform_problem_from_discrete_to_continuous(
            problem, generative_model, bounds
        )


    # print("hacked in an initial random sample of 10")
    # x0 = np.random.randn(10, generative_model.latent_dim)
    # y0 =  problem.black_box(x0)
    # problem.data_package = DataPackage(unsupervised_data=x0, supervised_data=(x0, y0))

    # load the solver
    initial_history_size = int(np.asarray(problem.data_package.supervised_data[0]).shape[0])
    diffusion_solver_kwargs = {}
    if solver_name == "cowboys_diffusion":
        if num_candidates is not None:
            diffusion_solver_kwargs["num_candidates"] = num_candidates
        if distillation_n is not None:
            diffusion_solver_kwargs["distillation_n"] = distillation_n
        if guidance_scale is not None:
            diffusion_solver_kwargs["guidance_scale"] = guidance_scale
        if clip_guidance is not None:
            diffusion_solver_kwargs["clip_guidance"] = clip_guidance
        if guide_every is not None:
            diffusion_solver_kwargs["guide_every"] = guide_every
        if guidance_alpha_bar_lower is not None:
            diffusion_solver_kwargs["guidance_alpha_bar_lower"] = guidance_alpha_bar_lower
        if guidance_alpha_bar_upper is not None:
            diffusion_solver_kwargs["guidance_alpha_bar_upper"] = guidance_alpha_bar_upper
        if diffusion_eta is not None:
            diffusion_solver_kwargs["eta"] = diffusion_eta

    solver = load_solver_from_problem(
        solver_name=solver_name,
        problem=problem,
        seed=seed,
        **diffusion_solver_kwargs,
        **afgg_solver_kwargs,
    )


    print(solver)
    if (
        solver_name in CONTINUOUS_SPACE_SOLVERS
        and hasattr(solver, "set_vae_and_bounds")
    ):
        solver.set_vae_and_bounds(generative_model, bounds)
        if solver_name != "cowboys":
            solver.bounds = bounds
        assert solver._given_vae
        if hasattr(solver, "load_diffusion_model_from_checkpoint"):
            checkpoint_path = (
                Path(diffusion_checkpoint_path)
                if diffusion_checkpoint_path is not None
                else resolve_default_latent_diffusion_checkpoint(n_dimensions)
            )
            solver.load_diffusion_model_from_checkpoint(checkpoint_path)

    diffusion_config_for_path = {
        "guide_mode": getattr(solver, "guide_mode", None),
        "num_candidates": getattr(solver, "num_candidates", None),
        "distillation_n": getattr(solver, "distillation_n", None),
        "guidance_scale": getattr(solver, "guidance_scale", None),
        "clip_guidance": getattr(solver, "clip_guidance", None),
        "guide_every": getattr(solver, "guide_every", None),
        "guidance_alpha_bar_lower": getattr(solver, "guidance_alpha_bar_lower", None),
        "guidance_alpha_bar_upper": getattr(solver, "guidance_alpha_bar_upper", None),
        "eta": getattr(solver, "eta", None),
    }
    output_dir = _build_output_dir(
        solver_name=solver_name,
        sufix=sufix,
        diffusion_config=diffusion_config_for_path,
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if solver_name == "cowboys":
        attach_gp_prediction_recorder(solver, initial_history_size)

    # 3. Optimize with checkpointing after every iteration when step() is available
    completed_iterations = 0
    final_status = "completed"
    run_error: Exception | None = None
    try:
        if callable(getattr(solver, "step", None)):
            for iteration_idx in range(1, max_iter + 1):
                solver.step()
                completed_iterations = iteration_idx
                if iteration_idx % checkpoint_every == 0:
                    _save_solver_progress(
                        solver=solver,
                        output_dir=output_dir,
                        solver_name=solver_name,
                        function_name=function_name,
                        seed=seed,
                        completed_iterations=completed_iterations,
                        status="running",
                        initial_history_size=initial_history_size,
                        save_iteration_results=save_iteration_results,
                        evaluate_top_candidate_objectives=evaluate_top_candidate_objectives,
                    )
        else:
            solver.solve(max_iter=max_iter)
            completed_iterations = max_iter
    except KeyboardInterrupt:
        final_status = "interrupted"
        print("Interrupted optimization.")
    except BudgetExhaustedException:
        final_status = "budget_exhausted"
        print("Budget exhausted.")
    except torch.OutOfMemoryError as exc:
        final_status = "oom"
        run_error = exc
        print(
            f"CUDA OOM after {completed_iterations} optimization iterations. "
            "Saving partial progress before exiting."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        final_status = "failed"
        run_error = exc
        print(
            f"Run failed with {type(exc).__name__} after {completed_iterations} "
            "optimization iterations. Saving partial progress before exiting."
        )
    finally:
        _save_solver_progress(
            solver=solver,
            output_dir=output_dir,
            solver_name=solver_name,
            function_name=function_name,
            seed=seed,
            completed_iterations=completed_iterations,
            status=final_status,
            initial_history_size=initial_history_size,
            save_iteration_results=save_iteration_results,
            evaluate_top_candidate_objectives=evaluate_top_candidate_objectives,
        )

    if run_error is not None:
        raise run_error

@click.command()
@click.option(
    "--function-name",
    type=str,
    default="foldx_stability",
    help="The name of the objective function to optimize.",
)
@click.option(
    "--solver-name",
    type=str,
    default="directed_evolution",
    help=f"The name of the solver to run. All solvers available are: {SOLVER_NAMES}",
)
@click.option("--n-dimensions", type=int, default=128)
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=100)
@click.option("--strict-on-hash/--no-strict-on-hash", type=bool, default=True)
@click.option("--force-run/--no-force-run", default=True)
@click.option("--wandb-mode", type=str, default="disabled")
@click.option("--tag", type=str, default="default")
@click.option("--sufix", type=str, default="default")
@click.option("--diffusion-checkpoint-path", type=str, default=None)
@click.option("--afgg-root", type=click.Path(path_type=Path, file_okay=False), envvar="AFGG_ROOT")
@click.option("--afgg-nfbo-root", type=click.Path(path_type=Path, file_okay=False), envvar="NFBO_ROOT")
@click.option("--afgg-checkpoint", type=click.Path(path_type=Path, dir_okay=False), envvar="AFGG_SEQFLOW_CKPT")
@click.option("--num-candidates", type=int, default=None)
@click.option("--distillation-n", type=int, default=None)
@click.option("--guidance-scale", type=float, default=None)
@click.option("--clip-guidance", type=float, default=None)
@click.option("--guide-every", type=int, default=None)
@click.option("--guidance-alpha-bar-lower", type=float, default=None)
@click.option("--guidance-alpha-bar-upper", type=float, default=None)
@click.option("--diffusion-eta", type=float, default=None)
@click.option(
    "--save-iteration-results/--no-save-iteration-results",
    default=True,
)
@click.option("--checkpoint-every", type=int, default=10)
@click.option(
    "--evaluate-top-candidate-objectives/--no-evaluate-top-candidate-objectives",
    default=False,
    help="Opt into extra post-run oracle calls outside the BO budget.",
)
def main(
    function_name: str,
    solver_name: str,
    n_dimensions: int,
    seed: int,
    max_iter: int,
    strict_on_hash: bool,
    force_run: bool,
    wandb_mode: str,
    tag: str,
    sufix: str,
    diffusion_checkpoint_path: str | None,
    num_candidates: int | None,
    distillation_n: int | None,
    guidance_scale: float | None,
    clip_guidance: float | None,
    guide_every: int | None,
    guidance_alpha_bar_lower: float | None,
    guidance_alpha_bar_upper: float | None,
    diffusion_eta: float | None,
    save_iteration_results: bool,
    checkpoint_every: int,
    evaluate_top_candidate_objectives: bool,
    afgg_root: Path | None,
    afgg_nfbo_root: Path | None,
    afgg_checkpoint: Path | None,
):
    _main(
        function_name,
        solver_name,
        n_dimensions,
        seed,
        max_iter,
        strict_on_hash,
        force_run,
        wandb_mode,
        tag,
        sufix,
        diffusion_checkpoint_path,
        num_candidates,
        distillation_n,
        guidance_scale,
        clip_guidance,
        guide_every,
        guidance_alpha_bar_lower,
        guidance_alpha_bar_upper,
        diffusion_eta,
        save_iteration_results,
        checkpoint_every,
        evaluate_top_candidate_objectives,
        afgg_root,
        afgg_nfbo_root,
        afgg_checkpoint,
    )


if __name__ == "__main__":
    main()
