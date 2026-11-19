"""Offline metrics CLI; no solver, oracle, VAE, or benchmark execution."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export the selected COWBOYS checkpoint directory")
    export.add_argument("--results-dir", type=Path, required=True)
    export.add_argument("--recursive", action="store_true", help="Explicitly include child experiment groups, excluding preflight")
    export.add_argument("--planned-bo-steps", type=int, help="Known planned budget; never inferred from final horizon")
    export.add_argument("--task-config", type=Path, help="Task-keyed thresholds/direction/oracle metadata JSON")
    aggregate = commands.add_parser("aggregate", help="Aggregate run.json files and plot offline")
    aggregate.add_argument("--results-dir", type=Path, action="append", required=True)
    aggregate.add_argument("--output-dir", type=Path)
    aggregate.add_argument("--no-plots", action="store_true")
    improvement = commands.add_parser("improvement", help="Analyze recorded EI versus realized record gains offline")
    improvement.add_argument("--status-file", type=Path, required=True)
    improvement.add_argument("--output-dir", type=Path)
    validate = commands.add_parser("validate", help="Validate generated run.json files")
    validate.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "improvement":
            from .improvement_analysis import improvement_report
            report = improvement_report(args.status_file, args.output_dir)
            print(f"Analyzed {report['summary']['n_bo_steps']} BO steps. Plot: {report['files']['png']}")
            return 0
        if args.command == "export":
            from .export_runs import export_directory
            from .metric_io import read_json
            config = read_json(args.task_config) if args.task_config else None
            manifest = export_directory(args.results_dir, recursive=args.recursive,
                                        task_configs=config, planned_bo_steps=args.planned_bo_steps)
            print(f"Exported {len(manifest['runs'])} run(s); {len(manifest['issues'])} issue(s). "
                  f"Manifest: {args.results_dir / 'metrics/export_manifest.json'}")
            if manifest["issues"]:
                for issue in manifest["issues"]:
                    print(f"  {issue}")
            return 1 if manifest["issues"] or not manifest["runs"] else 0
        if args.command == "aggregate":
            from .aggregate_runs import aggregate_directories
            result = aggregate_directories(args.results_dir, output_dir=args.output_dir, plots=not args.no_plots)
            output = args.output_dir or args.results_dir[0] / "metrics/aggregate"
            print(f"Aggregated {len(result['run_summary_rows'])} run(s), "
                  f"{len(result['compatibility_groups'])} compatible group(s). Output: {output}")
            return 1 if result["issues"] else 0
        from .export_runs import validate_run
        from .metric_io import read_json
        for path in args.paths:
            validate_run(read_json(path))
        print(f"Validated {len(args.paths)} run(s)")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Metrics error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
