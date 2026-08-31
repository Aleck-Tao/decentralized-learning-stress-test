from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import load_adult
from .pipeline import run_benchmark
from .provenance import verify_result_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlstress",
        description="CPU peer-to-peer learning stress test",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="verify and parse the pinned UCI Adult archive")
    validate.add_argument("--root", type=Path, default=Path("."))

    smoke = subparsers.add_parser("smoke", help="run the explicitly synthetic CPU smoke fixture")
    smoke.add_argument("--root", type=Path, default=Path("."))
    smoke.add_argument("--config", type=Path)
    smoke.add_argument("--output-dir", type=Path)

    benchmark = subparsers.add_parser("benchmark", help="run the pinned Adult benchmark")
    benchmark.add_argument("--root", type=Path, default=Path("."))
    benchmark.add_argument("--config", type=Path)
    benchmark.add_argument("--output-dir", type=Path)

    verify = subparsers.add_parser("verify", help="verify generated result and source bindings")
    verify.add_argument("--root", type=Path, default=Path("."))
    verify.add_argument("--result-dir", type=Path, default=Path("results/benchmark"))
    return parser


def _resolved_under(root: Path, supplied: Path | None, default: str) -> Path:
    path = supplied if supplied is not None else Path(default)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "validate-data":
        dataset = load_adult(root)
        print(
            json.dumps(
                {
                    "valid": True,
                    "provenance": dataset.provenance,
                    "split_rows": {
                        "train_pool": len(dataset.train_pool),
                        "global_test": len(dataset.global_test),
                        "attack_calibration_nonmember": len(dataset.attack_calibration_nonmember),
                        "attack_evaluation_nonmember": len(dataset.attack_evaluation_nonmember),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"smoke", "benchmark"}:
        is_smoke = args.command == "smoke"
        config = _resolved_under(
            root,
            args.config,
            "configs/smoke.json" if is_smoke else "configs/benchmark.json",
        )
        output = _resolved_under(
            root,
            args.output_dir,
            "results/smoke" if is_smoke else "results/benchmark",
        )
        result = run_benchmark(root, config, output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        result_dir = _resolved_under(root, args.result_dir, "results/benchmark")
        print(json.dumps(verify_result_manifest(root, result_dir), indent=2, sort_keys=True))
        return 0
    raise AssertionError("Unhandled command")

