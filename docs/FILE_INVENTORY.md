# Original tracked file inventory

Recorded before migration. Binary data/checkpoints are inventoried by path; they are not loaded. Python definitions and direct dependencies were inspected using AST without importing or executing training/analysis modules.

| Original file | Definitions / role | Direct import dependencies |
|---|---|---|
| `.github/workflows/hdbo-ax.yml` | Resource / configuration / documentation | ? |
| `.github/workflows/hdbo-base.yml` | Resource / configuration / documentation | ? |
| `.github/workflows/hdbo-baxus.yml` | Resource / configuration / documentation | ? |
| `.github/workflows/preprocessing-env-builds.yml` | Resource / configuration / documentation | ? |
| `.gitignore` | Resource / configuration / documentation | ? |
| `.pre-commit-config.yaml` | Resource / configuration / documentation | ? |
| `README.md` | Resource / configuration / documentation | ? |
| `batch.sh` | Resource / configuration / documentation | ? |
| `batch_gpu.sh` | Resource / configuration / documentation | ? |
| `cowboys.py` | COWBOYS | botorch.acquisition, botorch.acquisition.analytic, botorch.fit, botorch.models, botorch.optim, gauche.kernels.fingerprint_kernels, gpytorch, gpytorch.kernels, gpytorch.mlls, hdbo_benchmark.generative_models.vae_factory, hdbo_benchmark.utils.experiments.normalization, hdbo_benchmark.utils.logging.candidate_diagnostics, numpy, poli.core.abstract_black_box, poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization, rdkit, rdkit.Chem, selfies, torch, typing, warnings |
| `cowboys_flow.py` | StructuredBOState, LatentTargetEvaluation, WeightedLatentPool, MHSamplingResult, AffineCouplingLayer, WeightedRealNVPProposal, FlowMixtureProposal, COWBOYSFlow | __future__, botorch.acquisition, botorch.acquisition.analytic, botorch.fit, botorch.models, botorch.optim, dataclasses, gauche.kernels.fingerprint_kernels, gpytorch, gpytorch.kernels, gpytorch.mlls, hdbo_benchmark.utils.experiments.normalization, hdbo_benchmark.utils.logging.candidate_diagnostics, math, numpy, poli.core.abstract_black_box, poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization, rdkit, rdkit.Chem, selfies, torch, torch.nn, typing, warnings, zlib |
| `cowboys_flow_2.py` | StructuredBOState, LatentTargetEvaluation, WeightedLatentPool, MHSamplingResult, ProposalDiagnostics, AffineCouplingLayer, WeightedRealNVPProposal, FlowMixtureProposal, COWBOYSFlow | __future__, botorch.acquisition, botorch.acquisition.analytic, botorch.fit, botorch.models, botorch.optim, dataclasses, gauche.kernels.fingerprint_kernels, gpytorch, gpytorch.kernels, gpytorch.mlls, hdbo_benchmark.utils.experiments.normalization, hdbo_benchmark.utils.logging.candidate_diagnostics, math, numpy, poli.core.abstract_black_box, poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization, rdkit, rdkit.Chem, selfies, torch, torch.nn, typing, warnings, zlib |
| `data/esm_embeddings/closest_pdbs_clean.json` | Resource / configuration / documentation | ? |
| `data/esm_embeddings/esm_embeddings.json` | Resource / configuration / documentation | ? |
| `data/mario/all_levels_onehot.npz` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/2vad_A/2vad_A.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/2vad_A/2vad_A_Repair.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/2vae_D/2vae_D.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/2vae_D/2vae_D_Repair.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/3e5v_A/3e5v_A.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/3e5v_A/3e5v_A_Repair.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/3ned_A/3ned_A.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/3ned_A/3ned_A_Repair.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/5lk4_A/5lk4_A.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/5lk4_A/5lk4_A_Repair.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/6aa7_A/6aa7_A.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/6aa7_A/6aa7_A_Repair.pdb` | Resource / configuration / documentation | ? |
| `data/rfp_pdbs/readme.md` | Resource / configuration / documentation | ? |
| `data/small_molecule_datasets/processed/zinc250k_alphabet_stoi.json` | Resource / configuration / documentation | ? |
| `data/small_molecule_datasets/processed/zinc250k_metadata.json` | Resource / configuration / documentation | ? |
| `data/small_molecule_datasets/processed/zinc250k_sequence_lengths.csv` | Resource / configuration / documentation | ? |
| `data/small_molecule_datasets/processed/zinc250k_sequence_lengths.jpg` | Resource / configuration / documentation | ? |
| `data/trained_models/ae_for_esm/latent_dim_32/version_23/checkpoints/epoch=99-step=700.ckpt` | Resource / configuration / documentation | ? |
| `data/trained_models/ae_for_esm/latent_dim_32/version_23/hparams.yaml` | Resource / configuration / documentation | ? |
| `data/trained_models/training_diffusion_on_zinc_250k/latent_diffusion_latent_dim-128-seed-0.json` | Resource / configuration / documentation | ? |
| `data/trained_models/training_diffusion_on_zinc_250k/latent_diffusion_latent_dim-128-seed-0.pt` | Resource / configuration / documentation | ? |
| `data/trained_models/training_vae_on_zinc_250k/latent_dim-128-batch_size-512-lr-0.0005-seed-0.pt` | Resource / configuration / documentation | ? |
| `data/trained_models/training_vae_on_zinc_250k/latent_dim-128-batch_size-512-lr-0.0005-seed-1.pt` | Resource / configuration / documentation | ? |
| `data/trained_models/training_vae_on_zinc_250k/latent_dim-2-batch_size-512-lr-0.0005-seed-1.pt` | Resource / configuration / documentation | ? |
| `data/trained_models/training_vae_on_zinc_250k/latent_dim-64-batch_size-512-lr-0.0005-seed-0.pt` | Resource / configuration / documentation | ? |
| `data/trained_models/vae_on_zinc_250k/latent_dim_2.pt` | Resource / configuration / documentation | ? |
| `environment.data_preprocessing.yml` | Resource / configuration / documentation | ? |
| `environment.yml` | Resource / configuration / documentation | ? |
| `envs/environment.ax.yml` | Resource / configuration / documentation | ? |
| `envs/environment.baxus.yml` | Resource / configuration / documentation | ? |
| `envs/environment.bounce.yml` | Resource / configuration / documentation | ? |
| `envs/environment.lambo2.yml` | Resource / configuration / documentation | ? |
| `envs/environment.pr.yml` | Resource / configuration / documentation | ? |
| `gitignore` | Resource / configuration / documentation | ? |
| `pyproject.toml` | Resource / configuration / documentation | ? |
| `reports/plot_iteration_metrics.py` | MetricSpec, MethodConfig, RunRecord, AggregatedSeries, PreparedProblemData, positive_int, normalize_token, to_long_path, prettify_name, iter_iteration_folders, iter_folder_files, iter_iteration_result_files, read_json, missing_iteration_results_message, build_method_config, resolve_iteration_folders, discover_methods, resolve_methods, required_metric_keys_for_plotting, compute_adjusted_top_k_objective, load_runs, choose_seeds, sampling_indices, aggregate_metric, configure_metric_axis, plot_metric_series, configure_matplotlib, save_metric_plot, save_metric_summary_grid, mean_of_aggregated_series, scatter_points_for_problem, configure_scatter_axis, plot_tradeoff_scatter, save_tradeoff_scatter_plot, save_tradeoff_summary_grid, save_problem_metric_matrix_grid, print_available_options, resolve_selected_problems, parse_args, main | __future__, argparse, collections, dataclasses, json, math, matplotlib, matplotlib.pyplot, matplotlib.ticker, numpy, os, pathlib, string, sys, typing |
| `reports/table_results.py` | compute_npy_group_stats | numpy, pathlib, re |
| `requirements-dev.txt` | Resource / configuration / documentation | ? |
| `requirements.txt` | Resource / configuration / documentation | ? |
| `run.py` | _format_parameter_for_path, _build_output_dir, _path_for_io, _write_with_retries, _write_json, _save_solver_progress, _main, main | click, hdbo_benchmark.generative_models.latent_diffusion, hdbo_benchmark.utils.experiments.load_generative_models, hdbo_benchmark.utils.experiments.load_problems, hdbo_benchmark.utils.experiments.load_solvers, hdbo_benchmark.utils.experiments.problem_transformations, hdbo_benchmark.utils.experiments.verify_status_pre_experiment, hdbo_benchmark.utils.logging.idempotence_of_experiments, hdbo_benchmark.utils.logging.iteration_metrics, hdbo_benchmark.utils.logging.wandb_observer, json, numpy, os, pathlib, poli.core.data_package, poli.core.exceptions, poli.core.util.seeding, time, torch, uuid |
| `run.sh` | Resource / configuration / documentation | ? |
| `run_diffusion.sh` | Resource / configuration / documentation | ? |
| `run_diffusion2.sh` | Resource / configuration / documentation | ? |
| `run_diffusion_guidance_sweep.sh` | Resource / configuration / documentation | ? |
| `run_diffusion_iterations.sh` | Resource / configuration / documentation | ? |
| `run_flow.sh` | Resource / configuration / documentation | ? |
| `run_flow_2.sh` | Resource / configuration / documentation | ? |
| `run_iterations.sh` | Resource / configuration / documentation | ? |
| `run_iterations_nf.sh` | Resource / configuration / documentation | ? |
| `run_lsbo_poli.sh` | Resource / configuration / documentation | ? |
| `setup.cfg` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/assets/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/assets/training_vae_on_zinc_250k/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/assets/training_vae_on_zinc_250k/latent_dim-128-batch_size-512-lr-0.0005-seed-1-old.pt` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/assets/training_vae_on_zinc_250k/latent_dim-128-batch_size-512-lr-0.0005-seed-1.pt` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/assets/training_vae_on_zinc_250k/latent_dim-2-batch_size-512-lr-0.0005-seed-1.pt` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/assets/zinc250k_alphabet_stoi.json` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/assets/zinc250k_metadata.json` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/data_preprocessing/zinc250k/00_downloading_the_dataset.py` | Package marker / script-level statements | pathlib, pickle, torchdrug |
| `src/hdbo_benchmark/data_preprocessing/zinc250k/01_from_smiles_to_selfies.py` | translate_smiles_to_selfies | pathlib, pickle, selfies, typing |
| `src/hdbo_benchmark/data_preprocessing/zinc250k/02_save_as_csv.py` | Package marker / script-level statements | pandas, pathlib, pickle |
| `src/hdbo_benchmark/data_preprocessing/zinc250k/03_example_loadup.py` | Package marker / script-level statements | pandas, pathlib |
| `src/hdbo_benchmark/data_preprocessing/zinc250k/04_computing_alphabet.py` | Package marker / script-level statements | __future__, collections, json, matplotlib.pyplot, pandas, pathlib, selfies |
| `src/hdbo_benchmark/data_preprocessing/zinc250k/05_computing_integer_and_onehot_representations.py` | Package marker / script-level statements | json, numpy, pandas, pathlib, selfies |
| `src/hdbo_benchmark/data_preprocessing/zinc250k/06_human_readable_metadata.py` | Package marker / script-level statements | json, pandas, pathlib |
| `src/hdbo_benchmark/data_preprocessing/zinc250k/readme.md` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/experiments/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_ehrlich/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_ehrlich/batch.sh` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_ehrlich/compile.py` | condition_to_write | datetime, hdbo_benchmark.utils.experiments.load_solvers |
| `src/hdbo_benchmark/experiments/benchmark_on_ehrlich/run.py` | _from_sequence_to_one_hot, _from_one_hot_to_sequence, _in_onehot_space, main | click, hdbo_benchmark, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.experiments.load_solvers, hdbo_benchmark.utils.logging.idempotence_of_experiments, hdbo_benchmark.utils.logging.uncommited_changes, hdbo_benchmark.utils.logging.wandb_observer, numpy, poli, poli.core.exceptions, poli.core.util.seeding, poli.objective_repository, poli_baselines, torch, typing, uuid |
| `src/hdbo_benchmark/experiments/benchmark_on_foldx/readme.md` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_foldx/run.py` | in_latent_space, define_initial_values, main | click, hdbo_benchmark, hdbo_benchmark.generative_models.ae_for_esm, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.experiments.load_generative_models, hdbo_benchmark.utils.experiments.load_solvers, hdbo_benchmark.utils.experiments.normalization, hdbo_benchmark.utils.logging.idempotence_of_experiments, hdbo_benchmark.utils.logging.uncommited_changes, hdbo_benchmark.utils.logging.wandb_observer, json, numpy, pandas, poli, poli.core.abstract_black_box, poli.core.exceptions, poli.core.problem, poli.core.util.seeding, poli.repository, poli_baselines, torch, typing, uuid |
| `src/hdbo_benchmark/experiments/benchmark_on_pmo/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_pmo/batch.sh` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_pmo/compile.py` | condition_to_write | datetime, hdbo_benchmark.utils.experiments.load_solvers, poli.benchmarks |
| `src/hdbo_benchmark/experiments/benchmark_on_pmo/run.py` | in_latent_space, main | click, hdbo_benchmark, hdbo_benchmark.generative_models.vae_factory, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.experiments.load_metadata_for_vaes, hdbo_benchmark.utils.experiments.load_solvers, hdbo_benchmark.utils.logging.idempotence_of_experiments, hdbo_benchmark.utils.logging.uncommited_changes, hdbo_benchmark.utils.logging.wandb_observer, hdbo_benchmark.utils.tdc_compat, numpy, poli, poli.core.abstract_black_box, poli.core.exceptions, poli.core.util.seeding, poli_baselines, selfies, torch, typing, uuid |
| `src/hdbo_benchmark/experiments/benchmark_on_rasp/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_rasp/batch.sh` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_rasp/compile.py` | condition_to_write | datetime, hdbo_benchmark.utils.experiments.load_solvers |
| `src/hdbo_benchmark/experiments/benchmark_on_rasp/rasp_seed_data.npz` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_rasp/run.py` | in_latent_space, main | click, esm, hdbo_benchmark, hdbo_benchmark.generative_models.ae_for_esm, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.experiments.load_solvers, hdbo_benchmark.utils.experiments.normalization, hdbo_benchmark.utils.logging.idempotence_of_experiments, hdbo_benchmark.utils.logging.uncommited_changes, hdbo_benchmark.utils.logging.wandb_observer, json, numpy, pandas, poli, poli.core.abstract_black_box, poli.core.exceptions, poli.core.util.seeding, poli.repository, poli_baselines, torch, typing, uuid |
| `src/hdbo_benchmark/experiments/benchmark_on_toy_objective/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_toy_objective/compile.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_toy_objective/compiler.run_toy_objective.sh` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_toy_objective/readme.md` | Resource / configuration / documentation | ? |
| `src/hdbo_benchmark/experiments/benchmark_on_toy_objective/run.py` | run | click, hdbo_benchmark, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.experiments.load_solvers, hdbo_benchmark.utils.logging, numpy, poli, poli.core.util.seeding, poli.objective_repository, poli_baselines, torch, uuid |
| `src/hdbo_benchmark/experiments/pool.py` | run, run_parallel | click, multiprocessing, subprocess |
| `src/hdbo_benchmark/experiments/training_ae_on_rfp_pool/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/training_ae_on_rfp_pool/run.py` | main | click, esm, hdbo_benchmark.generative_models.ae_for_esm, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.download.from_drive, json, lightning, numpy, torch, torch.utils.data |
| `src/hdbo_benchmark/experiments/training_diffusion_on_zinc_250k/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/training_diffusion_on_zinc_250k/train_latent_diffusion.py` | _validation_loss, main | __future__, click, hdbo_benchmark.generative_models.latent_diffusion, hdbo_benchmark.generative_models.latent_diffusion_dataset, hdbo_benchmark.generative_models.vae_factory, hdbo_benchmark.utils.constants, numpy, pathlib, torch, wandb |
| `src/hdbo_benchmark/experiments/training_vae_on_zinc250k/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/experiments/training_vae_on_zinc250k/train_vae.py` | main | click, hdbo_benchmark, hdbo_benchmark.generative_models.vae_selfies, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.data.zinc_250k, hdbo_benchmark.utils.experiments.training_vaes, hdbo_benchmark.utils.logging, numpy, torch, wandb |
| `src/hdbo_benchmark/experiments/training_vae_on_zinc250k/train_vae_rnn.py` | main | click, hdbo_benchmark, hdbo_benchmark.generative_models.vae_rnn_selfies, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.data.zinc_250k, hdbo_benchmark.utils.experiments.training_vaes, hdbo_benchmark.utils.logging, numpy, torch, wandb |
| `src/hdbo_benchmark/generative_models/__init__.py` | Package marker / script-level statements | .latent_diffusion, .vae_mario, .vae_selfies |
| `src/hdbo_benchmark/generative_models/ae_for_esm.py` | LitAutoEncoder | esm, hdbo_benchmark.utils.constants, json, lightning, matplotlib.pyplot, numpy, pandas, pathlib, torch, torch.nn, torch.optim, torch.utils.data |
| `src/hdbo_benchmark/generative_models/latent_diffusion.py` | LatentDiffusionConfig, _extract, SinusoidalTimeEmbedding, LatentDiffusionDenoiser, LatentDiffusionModel, save_latent_diffusion_checkpoint, _strip_compiled_prefix_from_state_dict, load_latent_diffusion_checkpoint, resolve_default_latent_diffusion_checkpoint | __future__, dataclasses, hdbo_benchmark.utils.constants, json, math, pathlib, torch, torch.nn, torch.nn.functional, typing |
| `src/hdbo_benchmark/generative_models/latent_diffusion_dataset.py` | LatentDatasetBundle, resolve_default_latent_cache_path, _encode_dataset_to_latents, _load_or_build_latents, build_latent_diffusion_dataloaders | __future__, dataclasses, hdbo_benchmark.generative_models.vae_factory, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.data.zinc_250k, numpy, pathlib, torch, torch.utils.data |
| `src/hdbo_benchmark/generative_models/onehot.py` | OneHot | numpy |
| `src/hdbo_benchmark/generative_models/protein_ae_factory.py` | ProteinAEFactory | esm, hdbo_benchmark.generative_models.ae_for_esm, hdbo_benchmark.utils.constants |
| `src/hdbo_benchmark/generative_models/vae.py` | VAE | numpy, torch, torch.distributions, typing |
| `src/hdbo_benchmark/generative_models/vae_factory.py` | _strip_compiled_prefix_from_state_dict, infer_zinc_vae_latent_dim_from_checkpoint, load_zinc_vae_from_checkpoint, VAEFactory | hdbo_benchmark.assets, hdbo_benchmark.generative_models.vae, hdbo_benchmark.generative_models.vae_mario, hdbo_benchmark.generative_models.vae_selfies, hdbo_benchmark.utils.constants, pathlib, torch, typing |
| `src/hdbo_benchmark/generative_models/vae_mario.py` | VAEMario | hdbo_benchmark.generative_models.vae, hdbo_benchmark.utils.data.mario, hdbo_benchmark.utils.visualization.levels, itertools, matplotlib.pyplot, numpy, torch, torch.distributions, torch.nn, typing |
| `src/hdbo_benchmark/generative_models/vae_rnn_selfies.py` | Encoder, Decoder, VAERNNSelfies | hdbo_benchmark.generative_models.vae, hdbo_benchmark.utils.constants, json, numpy, torch, torch.distributions, torch.nn |
| `src/hdbo_benchmark/generative_models/vae_selfies-old.py` | VAESelfies | hdbo_benchmark.generative_models.vae, hdbo_benchmark.utils.selfies.tokens, hdbo_benchmark.utils.selfies.visualization, itertools, json, matplotlib.pyplot, numpy, pathlib, torch, torch.distributions, torch.nn, typing |
| `src/hdbo_benchmark/generative_models/vae_selfies.py` | VAESelfies | __future__, hdbo_benchmark.generative_models.vae, hdbo_benchmark.utils.experiments.load_metadata_for_vaes, hdbo_benchmark.utils.selfies.tokens, hdbo_benchmark.utils.selfies.visualization, itertools, matplotlib.pyplot, numpy, torch, torch.distributions, torch.nn, typing |
| `src/hdbo_benchmark/results/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/results/benchmark_on_pmo/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/results/benchmark_on_pmo/create_table.py` | convert_data_to_dataframes, create_base_table | hdbo_benchmark.utils.constants, hdbo_benchmark.utils.results.download_from_wandb, pandas, wandb |
| `src/hdbo_benchmark/results/benchmark_on_pmo/create_tables_for_website_and_paper.py` | compute_pretty_names, select_runs, summary_per_function, plot_heatmap, compute_pmo_benchmark_table, print_table_as_tex | hdbo_benchmark.results.benchmark_on_pmo.create_table, hdbo_benchmark.utils.constants, matplotlib, matplotlib.pyplot, numpy, pandas, seaborn |
| `src/hdbo_benchmark/results/benchmark_on_pmo/visualize_function_vs_solver_heatmap.py` | summary_per_function, plot_heatmap, print_table | hdbo_benchmark.results.benchmark_on_pmo.create_table, hdbo_benchmark.utils.constants, matplotlib, matplotlib.pyplot, numpy, pandas, seaborn |
| `src/hdbo_benchmark/results/benchmark_on_rasp/create_table_for_paper_and_website.py` | compute_pretty_names, summary_per_function, plot_heatmap, compute_rasp_benchmark_table, print_table_as_tex | hdbo_benchmark.results.benchmark_on_pmo.create_table, hdbo_benchmark.utils.constants, matplotlib, matplotlib.pyplot, numpy, pandas, seaborn |
| `src/hdbo_benchmark/results/download_tables.py` | convert_data_to_dataframes, create_base_table_for_ehrlich | hdbo_benchmark.utils.constants, hdbo_benchmark.utils.results.download_from_wandb, pandas, wandb |
| `src/hdbo_benchmark/results/other_results/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/results/other_results/comparison_between_experimental_setups.py` | Package marker / script-level statements | hdbo_benchmark.utils.constants, matplotlib, matplotlib.pyplot, pandas, seaborn |
| `src/hdbo_benchmark/results/other_results/comparison_between_sequence_lengths_and_dict_sizes.py` | Package marker / script-level statements | hdbo_benchmark.utils.constants, matplotlib, matplotlib.pyplot, pandas, seaborn |
| `src/hdbo_benchmark/results/other_results/visualizing_molecules.py` | Package marker / script-level statements | hdbo_benchmark.utils.constants, hdbo_benchmark.utils.selfies.visualization, json, matplotlib.pyplot, pandas, pickle |
| `src/hdbo_benchmark/results/print_tables_for_website_and_paper.py` | compute_pretty_names_for_solvers, compute_pretty_names_for_functions, select_runs, summary_per_function, print_table_as_tex, print_table_for_ehrlich | hdbo_benchmark.results.download_tables, hdbo_benchmark.utils.constants, json, numpy, pandas |
| `src/hdbo_benchmark/sanity_checks/bayesian_optimization_in_2d.py` | Package marker / script-level statements | matplotlib.pyplot, numpy, poli, poli_baselines.core.utils.visualization.objective_functions, poli_baselines.solvers, warnings |
| `src/hdbo_benchmark/solvers/__init__.py` | Package marker / script-level statements | .cowboys_diffusion, .latent_space_bo |
| `src/hdbo_benchmark/solvers/cowboys_diffusion.py` | StructuredBOState, DiffusionSamplingResult, SafeLogExpectedImprovement, LatentCritic, COWBOYSDiffusion | __future__, botorch.acquisition, botorch.acquisition.analytic, botorch.fit, botorch.models, botorch.optim, dataclasses, gauche.kernels.fingerprint_kernels, gpytorch, gpytorch.kernels, gpytorch.mlls, hdbo_benchmark.generative_models.latent_diffusion, hdbo_benchmark.generative_models.vae, hdbo_benchmark.utils.experiments.normalization, hdbo_benchmark.utils.logging.candidate_diagnostics, numpy, pathlib, poli.core.abstract_black_box, poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization, rdkit, rdkit.Chem, selfies, torch, torch.nn, torch.nn.functional, typing, warnings |
| `src/hdbo_benchmark/solvers/latent_space_bo.py` | LatentBOState, LatentSpaceBayesianOptimization | __future__, botorch.acquisition.analytic, botorch.fit, botorch.models, botorch.optim, dataclasses, gpytorch, gpytorch.kernels, gpytorch.mlls, hdbo_benchmark.utils.experiments.normalization, hdbo_benchmark.utils.logging.candidate_diagnostics, numpy, poli.core.abstract_black_box, poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization, rdkit, selfies, torch, torch.nn, typing, warnings |
| `src/hdbo_benchmark/utils/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/utils/ax/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/utils/ax/ax_solver.py` | AxSolver | ax.modelbridge.generation_strategy, ax.service.ax_client, hdbo_benchmark.utils.ax.interface, numpy, poli.core.abstract_black_box, poli.objective_repository, poli_baselines.core.abstract_solver, typing, uuid |
| `src/hdbo_benchmark/utils/ax/interface.py` | define_search_space, from_black_box_to_ax_metric, generator_run_from_initial_data | ax, ax.core.generator_run, ax.core.metric, ax.metrics.noisy_function, numpy, poli.core.abstract_black_box |
| `src/hdbo_benchmark/utils/constants.py` | Package marker / script-level statements | os, pathlib, torch |
| `src/hdbo_benchmark/utils/data/__init__.py` | Package marker / script-level statements | .experiments.loading_results |
| `src/hdbo_benchmark/utils/data/experiments/__init__.py` | Package marker / script-level statements | .loading_results |
| `src/hdbo_benchmark/utils/data/experiments/loading_results.py` | load_all_results, load_results_as_dataframe, load_results_as_dataframe_from_wandb, _load_run_as_row, load_all_results_as_dataframe_from_wandb | hdbo_benchmark.utils.constants, json, pandas, pathlib, typing, wandb |
| `src/hdbo_benchmark/utils/data/mario.py` | load_mario_tensors, load_mario_dataloaders | hdbo_benchmark.utils.constants, numpy, torch, torch.utils.data |
| `src/hdbo_benchmark/utils/data/zinc_250k.py` | resolve_zinc_250k_processed_dir, resolve_zinc_250k_dataset_path, load_zinc_250k_dataset, load_zinc_250k_alphabet, load_zinc_250k_dataloaders | hdbo_benchmark.utils.constants, json, numpy, pathlib, torch, torch.utils.data, typing |
| `src/hdbo_benchmark/utils/download/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/utils/download/from_drive.py` | download_file_from_google_drive, get_confirm_token, save_response_content | hashlib, requests |
| `src/hdbo_benchmark/utils/experiments/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/utils/experiments/load_generative_models.py` | load_generative_model_and_bounds | __future__, hdbo_benchmark.generative_models.onehot, hdbo_benchmark.generative_models.protein_ae_factory, hdbo_benchmark.generative_models.vae_factory, poli.benchmarks, poli.core.problem, typing |
| `src/hdbo_benchmark/utils/experiments/load_metadata_for_vaes.py` | load_alphabet_for_pmo, load_sequence_length_for_pmo | hdbo_benchmark.assets, json, pathlib |
| `src/hdbo_benchmark/utils/experiments/load_problems.py` | turn_into_supervised_problem, tokenize_selfies, _load_pmo_problem, _load_rasp, _load_foldx_stability, _load_pest_control_equivalent, _load_ehrlich_holo, _load_problem, load_problem | hdbo_benchmark.utils.constants, hdbo_benchmark.utils.experiments.load_metadata_for_vaes, hdbo_benchmark.utils.logging.wandb_observer, hdbo_benchmark.utils.tdc_compat, numpy, poli, poli.benchmarks, poli.core.chemistry.data_packages, poli.core.data_package, poli.core.problem, poli.repository, selfies, typing |
| `src/hdbo_benchmark/utils/experiments/load_solvers.py` | load_solver_class, load_solver_from_problem | cowboys, cowboys_flow, cowboys_flow_2, gpytorch, hdbo_benchmark.solvers.cowboys_diffusion, hdbo_benchmark.solvers.latent_space_bo, hdbo_benchmark.utils.constants, numpy, poli.core.problem, poli_baselines.core.abstract_solver, poli_baselines.solvers.bayesian_optimization.alebo, poli_baselines.solvers.bayesian_optimization.baxus, poli_baselines.solvers.bayesian_optimization.bounce, poli_baselines.solvers.bayesian_optimization.lambo2, poli_baselines.solvers.bayesian_optimization.line_bayesian_optimization, poli_baselines.solvers.bayesian_optimization.pr, poli_baselines.solvers.bayesian_optimization.saasbo, poli_baselines.solvers.bayesian_optimization.turbo.turbo_wrapper, poli_baselines.solvers.bayesian_optimization.vanilla_bayesian_optimization, poli_baselines.solvers.bayesian_optimization.vanilla_bo_hvarfner, poli_baselines.solvers.evolutionary_strategies.cma_es, poli_baselines.solvers.simple.continuous_random_mutation, poli_baselines.solvers.simple.genetic_algorithm, poli_baselines.solvers.simple.random_mutation, torch, typing |
| `src/hdbo_benchmark/utils/experiments/normalization.py` | from_unit_cube_to_range, from_range_to_unit_cube | numpy |
| `src/hdbo_benchmark/utils/experiments/problem_transformations.py` | _resolve_diagnostic_function, _in_latent_space_of_proteins, _in_the_latent_space_of_molecules, _in_onehot_space, transform_problem_from_discrete_to_continuous | __future__, hdbo_benchmark.generative_models.ae_for_esm, hdbo_benchmark.generative_models.onehot, hdbo_benchmark.generative_models.vae, hdbo_benchmark.utils.experiments.normalization, numpy, poli.core.black_box_information, poli.core.data_package, poli.core.lambda_black_box, poli.core.problem, typing |
| `src/hdbo_benchmark/utils/experiments/training_gps.py` | filter_close_points, train_exact_gp_model | gpytorch, pathlib, torch |
| `src/hdbo_benchmark/utils/experiments/training_vaes.py` | training_loop, testing_loop, train_model | hdbo_benchmark.generative_models, hdbo_benchmark.utils.constants, torch, torch.utils.data, typing |
| `src/hdbo_benchmark/utils/experiments/verify_status_pre_experiment.py` | repos_have_uncommited_changes, verify_repos_are_clean | hdbo_benchmark, hdbo_benchmark.utils.logging, poli, poli_baselines |
| `src/hdbo_benchmark/utils/latent_space_solver.py` | LatentSpaceSolver | numpy, poli.core.abstract_black_box, poli_baselines.core.abstract_solver, typing |
| `src/hdbo_benchmark/utils/logging/__init__.py` | Package marker / script-level statements | .idempotence_of_experiments, .library_hashes, .uncommited_changes, .wandb_observer |
| `src/hdbo_benchmark/utils/logging/back_up.py` | back_up | hdbo_benchmark.utils.constants, hdbo_benchmark.utils.data.experiments.loading_results, pandas, time, wandb |
| `src/hdbo_benchmark/utils/logging/benchmark_observer.py` | BenchmarkObserver | hdbo_benchmark.utils.constants, json, numpy, poli.core.black_box_information, poli.core.util.abstract_observer, uuid |
| `src/hdbo_benchmark/utils/logging/candidate_diagnostics.py` | resolve_diagnostic_evaluator, evaluate_candidate_objectives, preserve_rng_state, _call_preserving_diagnostic_state, build_top_candidate_diagnostics, _snapshot_diagnostic_state, _restore_diagnostic_state, _iter_state_targets, _iter_state_attribute_names, _copy_state_value | __future__, contextlib, copy, numpy, random, torch, typing |
| `src/hdbo_benchmark/utils/logging/idempotence_of_experiments.py` | experiment_has_already_run | hdbo_benchmark.utils.constants, wandb |
| `src/hdbo_benchmark/utils/logging/iteration_metrics.py` | build_iteration_metric_artifacts, _decode_unit_latents_to_structures, _normalize_structure_array, _structure_to_molecule_info, _structure_to_mol, _mol_to_count_fingerprint, _nearest_previous_tanimoto_distance, _nanmean, _serialize_vector, _serialize_number, _serialize_candidate_diagnostic | __future__, hdbo_benchmark.utils.logging.candidate_diagnostics, numpy, rdkit, rdkit.Chem, selfies, typing |
| `src/hdbo_benchmark/utils/logging/library_hashes.py` | get_git_hash_of_library, get_git_hash_of_library_from_pip | re, subprocess |
| `src/hdbo_benchmark/utils/logging/uncommited_changes.py` | has_uncommitted_changes | subprocess, types |
| `src/hdbo_benchmark/utils/logging/wandb_observer.py` | ObserverConfig, WandbObserver, initialize_observer | dataclasses, hdbo_benchmark, hdbo_benchmark.utils.constants, hdbo_benchmark.utils.logging.library_hashes, hdbo_benchmark.utils.logging.uncommited_changes, numpy, poli, poli.core.black_box_information, poli.core.problem, poli.core.util.abstract_observer, poli_baselines, typing, wandb |
| `src/hdbo_benchmark/utils/mario/__init__.py` | Package marker / script-level statements | .levels |
| `src/hdbo_benchmark/utils/mario/levels.py` | level_to_list, level_to_array, levels_to_onehot, vectorized, onehot_to_levels, tensor_to_sim_level, add_padding_to_level, clean_level | itertools, numpy, torch |
| `src/hdbo_benchmark/utils/results/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/utils/results/download_from_wandb.py` | get_all_runs_for_experiment, get_all_runs_for_function_names | hdbo_benchmark.utils.constants, wandb |
| `src/hdbo_benchmark/utils/selfies/__init__.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/utils/selfies/tokens.py` | from_selfie_to_tokens, from_tokens_to_ids, from_selfie_to_ids, from_ids_to_tensor, from_selfie_to_tensor, from_tensor_to_selfie, from_latent_code_to_selfie | json, pathlib, selfies, torch, typing |
| `src/hdbo_benchmark/utils/selfies/visualization.py` | draw_molecule_from_selfies, selfie_to_png, selfie_to_image, selfie_to_numpy_image_array | PIL.Image, cairosvg, io, numpy, pathlib, rdkit, rdkit.Chem.Draw, selfies |
| `src/hdbo_benchmark/utils/tdc_compat.py` | ensure_tdc_is_available, patch_broken_tdc_chem_utils | __future__, importlib, importlib.util, pathlib, sys, types |
| `src/hdbo_benchmark/utils/visualization/__init__.py` | Package marker / script-level statements | .model_quality |
| `src/hdbo_benchmark/utils/visualization/colors.py` | Package marker / script-level statements | ? |
| `src/hdbo_benchmark/utils/visualization/constants.py` | Package marker / script-level statements | matplotlib.colors, pathlib |
| `src/hdbo_benchmark/utils/visualization/levels.py` | absolute, save_level_from_array, plot_level_from_array, get_img_from_level, plot_level_from_decoded_tensor | PIL, hdbo_benchmark.utils.mario.levels, matplotlib.pyplot, numpy, pathlib, torch |
| `src/hdbo_benchmark/utils/visualization/model_quality.py` | plot_comparison_between_actual_and_predicted_values | .constants, matplotlib.pyplot, numpy, pathlib, seaborn, torch, typing |
| `src/hdbo_benchmark/utils/visualization/plot_optima_comparison.py` | plot_optima_comparison, plot_optima_comparison_from_df | .colors, hdbo_benchmark.utils.data.experiments, matplotlib.pyplot, pandas, seaborn |
| `test_main_entry_point.py` | construct_setups, test_main_run, test_main_run_ax, test_main_run_baxus, test_main_run_alebo, test_main_run_bounce, test_main_run_pr, test_main_run_lambo2 | hdbo_benchmark.utils.experiments.load_solvers, os, poli.benchmarks, pytest, run |
| `tox.ini` | Resource / configuration / documentation | ? |
| `workflows/website.yml` | Resource / configuration / documentation | ? |
