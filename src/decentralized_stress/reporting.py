from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv(row.get(key)) for key in fieldnames})


def _format_csv(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.8f}"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def markdown_report(summary: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    lines = [
        "# Decentralized learning stress-test report",
        "",
        "> This report is generated from a peer-to-peer simulation. It is not a network deployment, Byzantine-tolerance proof, privacy guarantee, or public-sector pilot.",
        "",
        f"Dataset class: `{metadata['dataset_kind']}`  ",
        f"Peers: {metadata['clients']}  ",
        f"Runs: {metadata['run_count']}  ",
        f"Topology: `{metadata['topology']}`",
        "",
        "| Scenario | Aggregator | Runs | Ensemble balanced accuracy | Worst-peer holdout balanced accuracy | Prediction disagreement | MIA ROC-AUC | Communication MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {scenario} | {aggregator} | {runs} | {ensemble:.3f} | {worst:.3f} | {disagreement:.3f} | {mia:.3f} | {communication:.3f} |".format(
                scenario=row["scenario"],
                aggregator=row["aggregator"],
                runs=row["runs"],
                ensemble=row["mean_ensemble_balanced_accuracy"],
                worst=row["mean_worst_peer_holdout_balanced_accuracy"],
                disagreement=row["mean_prediction_disagreement_rate"],
                mia=row["mean_membership_roc_auc"],
                communication=row["mean_communication_mib"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The client partitions are generated from a public benchmark and do not represent real organisations. Synthetic augmentation is a non-private local bootstrap with bounded jitter. Membership inference is one loss-based black-box attack; a low score does not establish privacy. Group metrics are diagnostics, not a fairness certification.",
            "",
        ]
    )
    return "\n".join(lines)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")

