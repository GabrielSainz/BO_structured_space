"""
This script is an entry_point for all experiments.

For documentation on how to run each individual experiment,
please refer to the README.md.
"""

# mypy: disable-error-code="import-untyped"
import json
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
from hdbo_benchmark.utils.experiments.load_problems import load_problem
from hdbo_benchmark.utils.experiments.load_solvers import (
    CONTINUOUS_SPACE_SOLVERS,
    SOLVER_NAMES,
    load_solver_from_problem,
)
from hdbo_benchmark.utils.experiments.problem_transformations import (
    transform_problem_from_discrete_to_continuous,
)
from hdbo_benchmark.utils.experiments.verify_status_pre_experiment import (
    verify_repos_are_clean,
)
from hdbo_benchmark.utils.logging.idempotence_of_experiments import (
    experiment_has_already_run,
)
from hdbo_benchmark.utils.logging.wandb_observer import ObserverConfig


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
    if solver_name != "cowboys_diffusion":
        return Path(f"./results/new_vae_10_chain_100_steps_with_stoch_sampling_{sufix}")

    components = [
        f"{name}-{_format_parameter_for_path(value)}"
        for name, value in diffusion_config.items()
    ]
    if sufix:
        components.append(f"tag-{_format_parameter_for_path(sufix)}")

    return Path("./results/diffusion") / "__".join(components)


def _save_solver_progress(
    solver,
    output_dir: Path,
    solver_name: str,
    function_name: str,
    seed: int,
    completed_iterations: int,
    status: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_performance = np.asarray(solver.get_best_performance())
    best_path = output_dir / f"{solver_name}_{function_name}_{seed}.npy"
    np.save(best_path, best_performance)

    if hasattr(solver, "get_history_as_arrays"):
        try:
            history_x, history_y = solver.get_history_as_arrays()
            np.savez_compressed(
                output_dir / f"{solver_name}_{function_name}_{seed}_history.npz",
                x=np.asarray(history_x),
                y=np.asarray(history_y),
                best_performance=best_performance,
            )
        except Exception as exc:
            print(f"Warning: could not save full history snapshot: {exc}")

    best_value = None
    if best_performance.size > 0 and np.isfinite(best_performance).any():
        best_value = float(np.nanmax(best_performance))

    with open(
        output_dir / f"{solver_name}_{function_name}_{seed}_status.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            {
                "status": status,
                "completed_iterations": int(completed_iterations),
                "best_value": best_value,
            },
            fp,
            indent=2,
        )


def _main(
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
):
    checkpoint_every = 10

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
    )


    print(solver)
    if (
        solver_name in CONTINUOUS_SPACE_SOLVERS
        and hasattr(solver, "set_vae_and_bounds")
    ):
        solver.set_vae_and_bounds(generative_model, bounds)
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
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
@click.option("--num-candidates", type=int, default=None)
@click.option("--distillation-n", type=int, default=None)
@click.option("--guidance-scale", type=float, default=None)
@click.option("--clip-guidance", type=float, default=None)
@click.option("--guide-every", type=int, default=None)
@click.option("--guidance-alpha-bar-lower", type=float, default=None)
@click.option("--guidance-alpha-bar-upper", type=float, default=None)
@click.option("--diffusion-eta", type=float, default=None)
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
    )


if __name__ == "__main__":
    main()
