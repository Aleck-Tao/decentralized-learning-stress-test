from __future__ import annotations

import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import BenchmarkConfig, load_config
from .data import (
    DatasetBundle,
    assert_disjoint,
    augment_local_non_private,
    load_adult,
    make_fixture,
    partition_clients,
)
from .evaluation import evaluate_with_privacy
from .protocol import make_topology, simulate
from .provenance import (
    sha256_file,
    source_tree_digest,
    write_json,
    write_result_manifest,
)
from .reporting import markdown_report, mean, write_csv


def _dataset(root: Path, kind: str) -> DatasetBundle:
    if kind == "adult":
        return load_adult(root)
    if kind == "fixture":
        return make_fixture()
    raise ValueError(f"Unknown dataset kind: {kind}")


def _scenario_partition(config: BenchmarkConfig, scenario: str) -> tuple[float | None, float, float]:
    if scenario == "iid_real":
        return None, 1.0, 0.0
    return config.partition_alpha, config.scarcity_fraction, config.scarce_client_fraction


def _uses_synthetic(scenario: str) -> bool:
    return scenario in {
        "noniid_scarce_synthetic",
        "noniid_sign_flip",
        "noniid_synthetic_poison",
        "noniid_synthetic_privacy_audit",
    }


def _malicious_peers(
    config: BenchmarkConfig, scenario: str, scarce_ids: tuple[int, ...], seed: int
) -> tuple[int, ...]:
    if scenario not in {"noniid_sign_flip", "noniid_synthetic_poison"}:
        return ()
    count = config.byzantine_peer_count
    if scenario == "noniid_synthetic_poison":
        if len(scarce_ids) < count:
            raise ValueError("Synthetic poisoning requires enough scarce peers for the attacker set")
        return tuple(sorted(scarce_ids[:count]))
    rng = np.random.default_rng(seed + 91_003)
    return tuple(sorted(int(value) for value in rng.choice(config.clients, size=count, replace=False)))


def _round_floats(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite result metric: {key}")
            rounded[key] = round(value, 8)
        else:
            rounded[key] = value
    return rounded


def _add_robustness_drop(rows: list[dict[str, Any]]) -> None:
    references = {
        (row["aggregator"], row["seed"]): row["ensemble_balanced_accuracy"]
        for row in rows
        if row["scenario"] == "noniid_scarce_synthetic"
    }
    for row in rows:
        key = (row["aggregator"], row["seed"])
        if row["scenario"] in {"noniid_sign_flip", "noniid_synthetic_poison"} and key in references:
            row["robustness_drop_from_synthetic_clean"] = round(
                float(references[key]) - float(row["ensemble_balanced_accuracy"]), 8
            )
        else:
            row["robustness_drop_from_synthetic_clean"] = None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), str(row["aggregator"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (scenario, aggregator), items in sorted(groups.items()):
        attacked_drops = [
            float(item["robustness_drop_from_synthetic_clean"])
            for item in items
            if item["robustness_drop_from_synthetic_clean"] is not None
        ]
        summary.append(
            _round_floats(
                {
                    "scenario": scenario,
                    "aggregator": aggregator,
                    "runs": len(items),
                    "mean_ensemble_balanced_accuracy": mean(
                        float(item["ensemble_balanced_accuracy"]) for item in items
                    ),
                    "mean_ensemble_roc_auc": mean(float(item["ensemble_roc_auc"]) for item in items),
                    "mean_worst_peer_holdout_balanced_accuracy": mean(
                        float(item["worst_peer_holdout_balanced_accuracy"]) for item in items
                    ),
                    "mean_prediction_disagreement_rate": mean(
                        float(item["prediction_disagreement_rate"]) for item in items
                    ),
                    "mean_final_parameter_consensus_distance": mean(
                        float(item["final_parameter_consensus_distance"]) for item in items
                    ),
                    "mean_subgroup_tpr_gap": mean(
                        float(item["ensemble_subgroup_tpr_gap"]) for item in items
                    ),
                    "mean_membership_roc_auc": mean(
                        float(item["membership_roc_auc"]) for item in items
                    ),
                    "mean_membership_advantage": mean(
                        float(item["membership_advantage"]) for item in items
                    ),
                    "mean_communication_mib": mean(
                        float(item["communication_bytes"]) / (1024.0 * 1024.0) for item in items
                    ),
                    "mean_robustness_drop_from_synthetic_clean": (
                        mean(attacked_drops) if attacked_drops else None
                    ),
                }
            )
        )
    return summary


def run_benchmark(
    root: Path,
    config_path: Path,
    result_dir: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_dir = result_dir.resolve()
    if not config_path.is_relative_to(root):
        raise ValueError("Configuration must remain within repository root")
    if not result_dir.is_relative_to(root / "results"):
        raise ValueError("Result directory must remain under results/")
    config = load_config(config_path)
    dataset = _dataset(root, config.dataset)
    topology = make_topology(config.topology, config.clients)
    result_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = root / "data" / "generated" / result_dir.name
    generated_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = generated_dir / "source_manifest.json"
    write_json(source_manifest_path, dataset.provenance)
    rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    partition_manifests: list[dict[str, Any]] = []
    synthetic_manifests: list[dict[str, Any]] = []

    for seed in config.seeds:
        for scenario in config.scenarios:
            alpha, scarcity_fraction, scarce_peer_fraction = _scenario_partition(config, scenario)
            partition = partition_clients(
                dataset.train_pool,
                config.clients,
                seed=seed,
                alpha=alpha,
                scarcity_fraction=scarcity_fraction,
                scarce_client_fraction=scarce_peer_fraction,
            )
            named_records = {f"peer_{peer}_train": value for peer, value in partition.train.items()}
            named_records.update({f"peer_{peer}_holdout": value for peer, value in partition.holdout.items()})
            named_records.update(
                {
                    "global_test": dataset.global_test,
                    "attack_calibration_nonmember": dataset.attack_calibration_nonmember,
                    "attack_evaluation_nonmember": dataset.attack_evaluation_nonmember,
                }
            )
            assert_disjoint(named_records)
            partition_manifest = dict(partition.partition_manifest)
            partition_manifest.update({"scenario": scenario, "benchmark_seed": seed})
            partition_manifests.append(partition_manifest)

            malicious = _malicious_peers(config, scenario, partition.scarce_peer_ids, seed)
            train_records = dict(partition.train)
            if _uses_synthetic(scenario):
                ratio = config.synthetic_ratio * (
                    2.0 if scenario == "noniid_synthetic_privacy_audit" else 1.0
                )
                for peer_id in partition.scarce_peer_ids:
                    train_records[peer_id], manifest = augment_local_non_private(
                        train_records[peer_id],
                        peer_id=peer_id,
                        ratio=ratio,
                        seed=seed * 100 + peer_id,
                        poison_labels=(
                            scenario == "noniid_synthetic_poison" and peer_id in malicious
                        ),
                    )
                    manifest.update({"scenario": scenario, "benchmark_seed": seed})
                    synthetic_manifests.append(manifest)

            for aggregator in config.aggregators:
                simulation = simulate(
                    train_records,
                    topology=topology,
                    aggregator=aggregator,
                    rounds=config.rounds,
                    hidden_dim=config.hidden_dim,
                    local_epochs=config.local_epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    trim_count=config.trim_count,
                    seed=seed,
                    malicious_peer_ids=malicious,
                    update_attack="sign_flip" if scenario == "noniid_sign_flip" else None,
                    sign_flip_scale=config.sign_flip_scale,
                )
                metrics = evaluate_with_privacy(
                    simulation.states,
                    train_records,
                    partition.holdout,
                    dataset.global_test,
                    dataset.attack_calibration_nonmember,
                    dataset.attack_evaluation_nonmember,
                    hidden_dim=config.hidden_dim,
                    seed=seed + 1_000,
                )
                row: dict[str, Any] = {
                    "run_id": f"{scenario}__{aggregator}__seed-{seed}",
                    "scenario": scenario,
                    "aggregator": aggregator,
                    "seed": seed,
                    "dataset_kind": dataset.provenance["kind"],
                    "topology": config.topology,
                    "clients": config.clients,
                    "rounds": config.rounds,
                    "malicious_peer_ids": list(malicious),
                    "scarce_peer_ids": list(partition.scarce_peer_ids),
                    "synthetic_ratio": (
                        config.synthetic_ratio
                        * (2.0 if scenario == "noniid_synthetic_privacy_audit" else 1.0)
                        if _uses_synthetic(scenario)
                        else 0.0
                    ),
                    "communication_bytes": simulation.communication_bytes,
                    "final_parameter_consensus_distance": float(
                        simulation.round_records[-1]["parameter_consensus_distance"]
                    ),
                    **metrics,
                }
                rows.append(_round_floats(row))
                for round_record in simulation.round_records:
                    round_rows.append(
                        _round_floats(
                            {
                                "run_id": row["run_id"],
                                "scenario": scenario,
                                "aggregator": aggregator,
                                "seed": seed,
                                **round_record,
                            }
                        )
                    )

    _add_robustness_drop(rows)
    summary = summarize(rows)
    partition_manifest_path = generated_dir / "partition_manifest.json"
    synthetic_manifest_path = generated_dir / "synthetic_manifest.json"
    write_json(
        partition_manifest_path,
        {
            "schema_version": "1.0",
            "partition_count": len(partition_manifests),
            "partitions": partition_manifests,
        },
    )
    write_json(
        synthetic_manifest_path,
        {
            "schema_version": "1.0",
            "generator_claim": "non-private local bootstrap with bounded numeric jitter",
            "privacy_guarantee": "none",
            "entry_count": len(synthetic_manifests),
            "entries": synthetic_manifests,
        },
    )
    write_csv(result_dir / "benchmark_runs.csv", rows)
    write_csv(result_dir / "round_metrics.csv", round_rows)
    metadata = {
        "dataset_kind": dataset.provenance["kind"],
        "clients": config.clients,
        "run_count": len(rows),
        "topology": config.topology,
        "scenario_count": len(config.scenarios),
        "aggregator_count": len(config.aggregators),
        "seed_count": len(config.seeds),
        "claim_boundary": "Single-process peer simulation; no network, BFT theorem, DP, or deployment claim.",
    }
    write_json(
        result_dir / "benchmark_summary.json",
        {"schema_version": "1.0", "metadata": metadata, "summary": summary},
    )
    (result_dir / "report.md").write_text(
        markdown_report(summary, metadata), encoding="utf-8", newline="\n"
    )
    source_hash, source_entries = source_tree_digest(root)
    run_manifest = {
        "schema_version": "1.0",
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "byteorder": sys.byteorder,
        },
        "source_tree_sha256": source_hash,
        "source_files": source_entries,
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "dataset_source_manifest_path": source_manifest_path.relative_to(root).as_posix(),
        "dataset_source_manifest_sha256": sha256_file(source_manifest_path),
        "partition_manifest_path": partition_manifest_path.relative_to(root).as_posix(),
        "partition_manifest_sha256": sha256_file(partition_manifest_path),
        "synthetic_manifest_path": synthetic_manifest_path.relative_to(root).as_posix(),
        "synthetic_manifest_sha256": sha256_file(synthetic_manifest_path),
    }
    write_json(result_dir / "run_manifest.json", run_manifest)
    result_manifest = write_result_manifest(root, result_dir)
    return {
        **metadata,
        "result_dir": result_dir.relative_to(root).as_posix(),
        "result_manifest_sha256": sha256_file(result_manifest),
    }
