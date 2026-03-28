# python reports/table_results.py

from pathlib import Path
import re
import numpy as np


def compute_npy_group_stats(folder):
    """
    Reads .npy files with names ending in _1, ..., _5 and computes
    the mean and std across the 5 files for each base filename.

    Example:
        cowboys_flow_albuterol_similarity_1.npy
        ...
        cowboys_flow_albuterol_similarity_5.npy

    Returns
    -------
    results : dict
        {
            "cowboys_flow_albuterol_similarity": {
                "files": [...],
                "stacked": np.ndarray,   # shape: (n_files, ...)
                "mean": np.ndarray,      # mean across files
                "std": np.ndarray        # std across files
            },
            ...
        }
    """
    folder = Path(folder)

    pattern = re.compile(r"(.+)_([1-5])\.npy$")
    groups = {}

    for file in folder.glob("*.npy"):
        match = pattern.match(file.name)
        if match:
            base_name = match.group(1)
            run_id = int(match.group(2))
            groups.setdefault(base_name, {})[run_id] = file

    results = {}

    for base_name, files_dict in groups.items():
        missing = [i for i in range(1, 6) if i not in files_dict]
        if missing:
            print(f"Skipping {base_name}: missing files {missing}")
            continue

        files_ordered = [files_dict[i] for i in range(1, 6)]
        arrays = [np.load(f, allow_pickle=True) for f in files_ordered]

        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) != 1:
            raise ValueError(
                f"Shape mismatch in group '{base_name}': {shapes}"
            )

        stacked = np.stack(arrays, axis=0)   # shape: (5, ...)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

        results[base_name] = {
            "files": files_ordered,
            "stacked": stacked,
            "mean": mean,
            "std": std,
        }

    return results


folder = "results/new_vae_10_chain_100_steps_with_stoch_sampling_1"
stats = compute_npy_group_stats(folder)

for name, res in stats.items():
    print(f"\n{name}")
    print(res['stacked'])
    print("mean ± std:")
    print(f"{res['mean']} ± {res['std']}")