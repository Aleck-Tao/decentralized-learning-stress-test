from __future__ import annotations

import itertools
from dataclasses import asdict

import numpy as np

from .data import Records
from .model import ModelState, predict
from .privacy import audit_membership, roc_auc


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups, dtype=int)
    if labels.shape != probabilities.shape or groups.shape != labels.shape:
        raise ValueError("Metric inputs must have equal shapes")
    predictions = probabilities >= 0.5
    accuracy = float(np.mean(predictions == labels))
    recalls: list[float] = []
    for label in (0, 1):
        mask = labels == label
        if np.any(mask):
            recalls.append(float(np.mean(predictions[mask] == label)))
    balanced_accuracy = float(np.mean(recalls))
    auc = roc_auc(labels, probabilities) if len(np.unique(labels)) == 2 else 0.5
    group_tprs: list[float] = []
    for group in (0, 1):
        mask = (groups == group) & (labels == 1)
        if np.any(mask):
            group_tprs.append(float(np.mean(predictions[mask])))
    tpr_gap = float(max(group_tprs) - min(group_tprs)) if len(group_tprs) == 2 else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": float(auc),
        "subgroup_tpr_gap": tpr_gap,
    }


def evaluate_states(
    states: dict[int, ModelState],
    global_test: Records,
    holdouts: dict[int, Records],
    *,
    hidden_dim: int,
) -> dict[str, float]:
    if set(states) != set(holdouts):
        raise ValueError("State and holdout peer IDs differ")
    global_probabilities: dict[int, np.ndarray] = {}
    global_balanced: list[float] = []
    global_auc: list[float] = []
    holdout_balanced: list[float] = []
    for peer_id in sorted(states):
        probabilities, _ = predict(states[peer_id], global_test, hidden_dim)
        global_probabilities[peer_id] = probabilities
        metrics = binary_metrics(global_test.y, probabilities, global_test.groups)
        global_balanced.append(metrics["balanced_accuracy"])
        global_auc.append(metrics["roc_auc"])
        local_probabilities, _ = predict(states[peer_id], holdouts[peer_id], hidden_dim)
        holdout_metrics = binary_metrics(
            holdouts[peer_id].y, local_probabilities, holdouts[peer_id].groups
        )
        holdout_balanced.append(holdout_metrics["balanced_accuracy"])
    stacked = np.stack([global_probabilities[peer_id] for peer_id in sorted(states)])
    ensemble = np.mean(stacked, axis=0)
    ensemble_metrics = binary_metrics(global_test.y, ensemble, global_test.groups)
    disagreements: list[float] = []
    for left, right in itertools.combinations(range(stacked.shape[0]), 2):
        disagreements.append(float(np.mean((stacked[left] >= 0.5) != (stacked[right] >= 0.5))))
    return {
        "mean_peer_global_balanced_accuracy": float(np.mean(global_balanced)),
        "mean_peer_global_roc_auc": float(np.mean(global_auc)),
        "ensemble_balanced_accuracy": ensemble_metrics["balanced_accuracy"],
        "ensemble_roc_auc": ensemble_metrics["roc_auc"],
        "ensemble_subgroup_tpr_gap": ensemble_metrics["subgroup_tpr_gap"],
        "mean_peer_holdout_balanced_accuracy": float(np.mean(holdout_balanced)),
        "worst_peer_holdout_balanced_accuracy": float(np.min(holdout_balanced)),
        "peer_holdout_balanced_accuracy_std": float(np.std(holdout_balanced)),
        "prediction_disagreement_rate": float(np.mean(disagreements)) if disagreements else 0.0,
    }


def evaluate_with_privacy(
    states: dict[int, ModelState],
    train_records: dict[int, Records],
    holdouts: dict[int, Records],
    global_test: Records,
    calibration_nonmembers: Records,
    evaluation_nonmembers: Records,
    *,
    hidden_dim: int,
    seed: int,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = evaluate_states(
        states, global_test, holdouts, hidden_dim=hidden_dim
    )
    membership = audit_membership(
        states,
        train_records,
        calibration_nonmembers,
        evaluation_nonmembers,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    for key, value in asdict(membership).items():
        metrics[f"membership_{key}"] = value
    return metrics

