"""Summaries of passive aggregation instrumentation, with seed-paired controls."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean

from .provenance import write_json
from .reporting import write_csv


def write_diagnostic_analysis(root: Path, diagnostics: list[dict], runs: list[dict]) -> None:
    targets: dict[int, set[int]] = defaultdict(set)
    for row in diagnostics:
        if row["attacked"]:
            targets[row["seed"]].add(row["sender_id"])
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in diagnostics:
        grouped[(row["scenario"], row["seed"], row["sender_id"])].append(row)
    senders = []
    for (scenario, seed, sender), rows in sorted(grouped.items()):
        coordinates = sum(row["coordinates"] for row in rows)
        senders.append({
            "scenario": scenario, "seed": seed, "sender_id": sender,
            "role": "targeted_sender" if sender in targets[seed] else "other_sender",
            "round_receiver_observations": len(rows),
            "sender_positive_fraction": rows[0]["sender_positive_fraction"],
            "sender_train_rows": rows[0]["sender_train_rows"],
            "rejection_rate": sum(row["rejected_coordinates"] for row in rows) / coordinates,
            "tie_averaged_rejection_rate": sum(row["tie_averaged_rejected_coordinates"] for row in rows) / coordinates,
            "ambiguous_boundary_rate": sum(row["ambiguous_boundary_coordinates"] for row in rows) / coordinates,
            "tied_coordinate_rate": sum(row["tied_coordinates"] for row in rows) / coordinates,
            "update_l2_norm_mean": fmean(row["update_l2_norm"] for row in rows),
        })
    write_csv(root / "per_sender_diagnostics.csv", senders)
    role_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in senders:
        role_groups[(row["scenario"], row["role"])].append(row)
    summary = []
    for (scenario, role), rows in sorted(role_groups.items()):
        summary.append({"scenario": scenario, "role": role,
                        "seed_count": len({row["seed"] for row in rows}),
                        "tie_averaged_rejection_rate": fmean(row["tie_averaged_rejection_rate"] for row in rows),
                        "ambiguous_boundary_rate": fmean(row["ambiguous_boundary_rate"] for row in rows),
                        "update_l2_norm_mean": fmean(row["update_l2_norm_mean"] for row in rows)})
    clean = {(row["aggregator"], row["seed"]): row for row in runs
             if row["scenario"] == "noniid_scarce_synthetic"}
    attacked = {(row["aggregator"], row["seed"]): row for row in runs
                if row["scenario"] == "noniid_sign_flip"}
    if not clean or set(clean) != set(attacked):
        raise ValueError("diagnostic report requires matched clean/sign-flip runs")
    paired = [{"aggregator": agg, "seed": seed,
               "clean_balanced_accuracy": clean[(agg, seed)]["ensemble_balanced_accuracy"],
               "attacked_balanced_accuracy": attacked[(agg, seed)]["ensemble_balanced_accuracy"],
               "attack_minus_clean": attacked[(agg, seed)]["ensemble_balanced_accuracy"] -
                                      clean[(agg, seed)]["ensemble_balanced_accuracy"]}
              for agg, seed in sorted(clean)]
    write_json(root / "diagnostic_summary.json", {"sender_role_summary": summary, "paired_accuracy": paired})
    lines = ["# What coordinate trimming removes", "",
             "Passive diagnostics accompany matched clean/sign-flip runs. The targeted-sender label "
             "uses the attacked run's sender IDs in both conditions; all senders are honest in the clean run. "
             "The aggregation rule never reads that evaluator label.", "",
             "| Scenario | Sender group | Tie-averaged rejection | Boundary ambiguity | Mean update L2 norm |",
             "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['scenario']} | {row['role']} | {row['tie_averaged_rejection_rate']:.4f} | "
                     f"{row['ambiguous_boundary_rate']:.4f} | {row['update_l2_norm_mean']:.5f} |")
    lines += ["", "Exact rank-based rejection and tie-averaged rejection are both retained. "
              "If identical values straddle a trimming boundary, the exact sender assignment depends "
              "on sorting; the tie-averaged measure divides rejected slots equally among equal values. "
              "Each eight-peer receiver removes half the coordinates overall (two at each end).", "",
              "Rows average sender summaries across the same three seeds. Rounds and receivers are "
              "repeated observations, not independent replicates. Per-sender class proportions, sample "
              "counts, and rejection histories are in per_sender_diagnostics.csv and aggregation_diagnostics.csv.", "",
              "| Aggregator | Seed | Clean balanced accuracy | Attacked balanced accuracy | Attack minus clean |",
              "|---|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['aggregator']} | {row['seed']} | {row['clean_balanced_accuracy']:.5f} | "
                     f"{row['attacked_balanced_accuracy']:.5f} | {row['attack_minus_clean']:.5f} |")
    lines += ["", "A rejected coordinate is not automatically harmful or useful. These measurements "
              "show whose values are removed and how often; they do not prove that minority-label "
              "information caused the clean-accuracy gap. That causal question needs a further intervention "
              "on partitions or training, not a correlation between label skew and rejection."]
    (root / "DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
