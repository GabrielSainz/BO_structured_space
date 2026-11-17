"""Compare completed corrected COWBOYS runs with paper Table 1 (no objectives)."""

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path


REFERENCE = Path(__file__).resolve().parents[3] / "configs/cowboys_paper_table1.json"


def compare_results(directory: Path, reference: dict) -> list[dict]:
    directory = directory.resolve()
    if os.name == "nt" and not str(directory).startswith("\\\\?\\"):
        directory = Path("\\\\?\\" + str(directory))
    rows = []
    for task, target in reference["tasks"].items():
        values, excluded = [], []
        for seed in range(1, reference["repetitions"] + 1):
            path = directory / f"cowboys_{task}_{seed}_status.json"
            if not path.exists():
                excluded.append(f"{seed}:missing_or_initialization_failed")
                continue
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
                config = status.get("solver_configuration", {})
                eligible = (
                    status.get("status") == "completed"
                    and status.get("completed_iterations") == reference["bo_iterations"]
                    and status.get("initial_history_size") == reference["initial_molecules"]
                    and status.get("diagnostic_objective_evaluations_enabled") is False
                    and config.get("implementation") == "cowboys_corrected_pcn_v1"
                    and config.get("num_chains") == reference["num_chains"]
                    and config.get("n_steps") == reference["n_steps"]
                    and config.get("batch_size") == reference["batch_size"]
                    and config.get("latent_dim") == reference["latent_dim"]
                    and config.get("vae_checkpoint_sha256") == reference["checkpoint_sha256"]
                )
                value = status.get("best_value")
                if eligible and isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(float(value))
                else:
                    excluded.append(f"{seed}:{status.get('status', 'unknown')}_or_protocol_mismatch")
            except (ValueError, OSError) as exc:
                excluded.append(f"{seed}:unreadable_{type(exc).__name__}")
        mean = statistics.mean(values) if values else None
        rows.append({
            "task": task,
            "completed_matching_seeds": len(values),
            "expected_seeds": reference["repetitions"],
            "comparison_complete": len(values) == reference["repetitions"],
            "local_mean": mean,
            "local_sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "paper_mean": target["mean"],
            "paper_sd": target["sd"],
            "local_minus_paper_mean": mean - target["mean"] if mean is not None else None,
            "excluded_seeds": ";".join(excluded),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    rows = compare_results(args.directory, reference)
    directory = args.directory.resolve()
    if os.name == "nt" and not str(directory).startswith("\\\\?\\"):
        directory = Path("\\\\?\\" + str(directory))
    with (directory / "paper_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "source": reference["source"],
        "note": "Differences are descriptive; incomplete seed sets do not establish reproduction.",
        "tasks": rows,
    }
    (directory / "paper_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        mean = "unavailable" if row["local_mean"] is None else f"{row['local_mean']:.4f}"
        print(f"{row['task']}: {row['completed_matching_seeds']}/5 seeds; "
              f"local {mean}; paper {row['paper_mean']:.4f}")


if __name__ == "__main__":
    main()
