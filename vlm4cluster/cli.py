from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vlm4cluster.config import DatasetConfig, load_experiment_config, parse_config_overrides
from vlm4cluster.datasets import list_image_dataset_specs, list_text_dataset_specs
from vlm4cluster.features import list_image_extractors, list_text_extractors
from vlm4cluster.methods import list_methods
from vlm4cluster.runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm4cluster", description="Run clustering benchmark experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one experiment from a TOML config plus CLI overrides.")
    run_parser.add_argument("--config", required=True, help="Path to an experiment TOML config.")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate config and print the execution plan.")
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config value. Can be repeated.",
    )
    run_parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Config overrides in KEY=VALUE form, e.g. dataset=cifar10 method=ssc_omp "
            "method.n_clusters=10 runtime.device=cuda."
        ),
    )

    subparsers.add_parser("list-datasets", help="List available image/text datasets.")
    stats_parser = subparsers.add_parser(
        "dataset-stats",
        help="List local sample and class counts for image dataset splits.",
    )
    stats_parser.add_argument(
        "datasets",
        nargs="*",
        help="Optional image dataset names to inspect. Defaults to all registered image datasets.",
    )
    stats_parser.add_argument("--root", default="data", help="Dataset root directory. Defaults to data.")
    stats_parser.add_argument(
        "--download",
        action="store_true",
        help="Allow Python dataset builders to download missing datasets. Disabled by default.",
    )
    subparsers.add_parser("list-extractors", help="List available image/text feature extractors.")
    subparsers.add_parser("list-methods", help="List available clustering methods.")
    return parser


def _print_section(title: str, rows: list[str]) -> None:
    print(f"[{title}]")
    for row in rows:
        print(f"  {row}")


def _handle_list_datasets() -> int:
    _print_section(
        "image datasets",
        [
            f"{spec.name}: {spec.description} | download={spec.download_mode} | splits={','.join(spec.supported_splits)}"
            for spec in list_image_dataset_specs()
        ],
    )
    _print_section(
        "text datasets",
        [f"{spec.name}: {spec.description} | source={spec.download_mode}" for spec in list_text_dataset_specs()],
    )
    return 0


def _short_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 140:
        return f"{message[:137]}..."
    return message


def _handle_dataset_stats(dataset_names: list[str], root: str, download: bool) -> int:
    specs = list_image_dataset_specs()
    if dataset_names:
        requested = {name.strip().lower() for name in dataset_names}
        specs = [spec for spec in specs if spec.name.lower() in requested]
        found = {spec.name.lower() for spec in specs}
        missing = sorted(requested - found)
        if missing:
            print(f"Unknown image dataset(s): {', '.join(missing)}", file=sys.stderr)
            return 1

    print("[image dataset stats]")
    print(f"root={root} download={'true' if download else 'false'}")
    for spec in specs:
        split_rows = []
        for split in spec.supported_splits:
            config = DatasetConfig(name=spec.name, root=root, split=split, download=download)
            try:
                loaded = spec.builder(config)
            except Exception as exc:  # noqa: BLE001 - keep CLI summary useful across dataset backends.
                split_rows.append(f"{split}: unavailable ({_short_error(exc)})")
                continue
            split_rows.append(f"{split}: samples={loaded.num_samples} classes={loaded.num_classes}")
        print(f"  {spec.name}: {' | '.join(split_rows)}")
    return 0


def _handle_list_extractors() -> int:
    _print_section(
        "image extractors",
        [f"{cls.name}: {cls.description}" for cls in list_image_extractors()],
    )
    _print_section(
        "text extractors",
        [f"{cls.name}: {cls.description}" for cls in list_text_extractors()],
    )
    return 0


def _handle_list_methods() -> int:
    _print_section(
        "methods",
        [
            (
                f"{cls.name}: {cls.description} | category={cls.category} | "
                f"raw={cls.requires_raw_images} image_features={cls.requires_image_features} "
                f"text_features={cls.requires_text_features}"
            )
            for cls in list_methods()
        ],
    )
    return 0


def _handle_run(config_path: str, dry_run: bool, override_items: list[str]) -> int:
    overrides = parse_config_overrides(override_items)
    config = load_experiment_config(config_path, overrides=overrides)
    runner = ExperimentRunner(config, project_root=Path.cwd())
    report = runner.run(dry_run=dry_run)

    if dry_run:
        print(f"Dry-run plan written to {report['plan_path']}")
        for step in report["plan"]["steps"]:
            print(f"- {step}")
        return 0

    report_path = Path(report["artifacts"]["output_dir"]) / "report.json"
    print(f"Experiment completed. Report: {report_path}", flush=True)
    if report["metrics"]:
        print("Metrics:", flush=True)
        for name, value in report["metrics"].items():
            print(f"- {name}: {value:.4f}", flush=True)
    selected_hyperparameters = report.get("result_metadata", {}).get("selected_hyperparameters")
    if isinstance(selected_hyperparameters, dict) and selected_hyperparameters:
        print("Selected hyperparameters:", flush=True)
        for key in (
            "selection_mode",
            "selection_metric",
            "inner_lr",
            "outer_lr",
            "warm_start",
            "cross_val_score",
            "candidate_rank",
            "num_candidates",
            "T",
            "M",
            "gamma",
            "batch_size",
        ):
            if key in selected_hyperparameters:
                value = selected_hyperparameters[key]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                print(f"- {key}: {value}", flush=True)
    for group_name, group_metrics in report.get("extra_metrics", {}).items():
        if not group_metrics:
            continue
        print(f"Extra metrics ({group_name}):", flush=True)
        for name, value in group_metrics.items():
            print(f"- {name}: {value:.4f}", flush=True)
    return 0


def _normalize_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        argv = sys.argv[1:]
    commands = {"run", "list-datasets", "dataset-stats", "list-extractors", "list-methods"}
    if argv and argv[0] not in commands and "--config" in argv:
        return ["run", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))

    if args.command == "list-datasets":
        return _handle_list_datasets()
    if args.command == "dataset-stats":
        return _handle_dataset_stats(args.datasets, args.root, args.download)
    if args.command == "list-extractors":
        return _handle_list_extractors()
    if args.command == "list-methods":
        return _handle_list_methods()
    if args.command == "run":
        return _handle_run(args.config, args.dry_run, [*args.set, *args.overrides])

    parser.error(f"Unknown command: {args.command}")
    return 1
