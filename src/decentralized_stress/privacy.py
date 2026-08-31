from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Records
from .model import ModelState, predict


@dataclass(frozen=True)
class MembershipResult:
    threshold: float
    roc_auc: float
    advantage: float
    tpr_at_1pct_fpr: float
    calibration_members: int
    calibration_nonmembers: int
    evaluation_members: int
    evaluation_nonmembers: int


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be equal one-dimensional arrays")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    tpr = float(np.mean(predictions[positives])) if np.any(positives) else 0.0
    fpr = float(np.mean(predictions[negatives])) if np.any(negatives) else 0.0
    return tpr, fpr


def calibrate_threshold(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate(
        [np.ones(len(member_scores), dtype=int), np.zeros(len(nonmember_scores), dtype=int)]
    )
    if len(member_scores) == 0 or len(nonmember_scores) == 0:
        raise ValueError("Calibration requires members and nonmembers")
    candidates = np.unique(scores)
    best_threshold = float(candidates[0])
    best_advantage = -float("inf")
    for threshold in candidates:
        tpr, fpr = _rates(labels, scores, float(threshold))
        advantage = tpr - fpr
        if advantage > best_advantage:
            best_advantage = advantage
            best_threshold = float(threshold)
    return best_threshold


def membership_from_scores(
    calibration_member_scores: np.ndarray,
    calibration_nonmember_scores: np.ndarray,
    evaluation_member_scores: np.ndarray,
    evaluation_nonmember_scores: np.ndarray,
) -> MembershipResult:
    threshold = calibrate_threshold(calibration_member_scores, calibration_nonmember_scores)
    scores = np.concatenate([evaluation_member_scores, evaluation_nonmember_scores])
    labels = np.concatenate(
        [np.ones(len(evaluation_member_scores), dtype=int), np.zeros(len(evaluation_nonmember_scores), dtype=int)]
    )
    auc = roc_auc(labels, scores)
    tpr, fpr = _rates(labels, scores, threshold)
    candidates = np.unique(scores)
    tpr_at_1pct = 0.0
    for candidate in candidates:
        candidate_tpr, candidate_fpr = _rates(labels, scores, float(candidate))
        if candidate_fpr <= 0.01:
            tpr_at_1pct = max(tpr_at_1pct, candidate_tpr)
    return MembershipResult(
        threshold=threshold,
        roc_auc=float(auc),
        advantage=float(tpr - fpr),
        tpr_at_1pct_fpr=float(tpr_at_1pct),
        calibration_members=len(calibration_member_scores),
        calibration_nonmembers=len(calibration_nonmember_scores),
        evaluation_members=len(evaluation_member_scores),
        evaluation_nonmembers=len(evaluation_nonmember_scores),
    )


def audit_membership(
    states: dict[int, ModelState],
    train_records: dict[int, Records],
    calibration_nonmembers: Records,
    evaluation_nonmembers: Records,
    *,
    hidden_dim: int,
    seed: int,
    maximum_per_peer: int = 64,
) -> MembershipResult:
    if set(states) != set(train_records):
        raise ValueError("Membership audit states and peer datasets differ")
    rng = np.random.default_rng(seed)
    cal_member_scores: list[float] = []
    eval_member_scores: list[float] = []
    cal_nonmember_scores: list[float] = []
    eval_nonmember_scores: list[float] = []
    cal_chunks = np.array_split(rng.permutation(len(calibration_nonmembers)), len(states))
    eval_chunks = np.array_split(rng.permutation(len(evaluation_nonmembers)), len(states))
    member_cal_ids: set[str] = set()
    member_eval_ids: set[str] = set()
    for offset, peer_id in enumerate(sorted(states)):
        peer = train_records[peer_id]
        real_indices = np.flatnonzero(~peer.synthetic)
        order = rng.permutation(real_indices)
        calibration_indices = cal_chunks[offset]
        evaluation_indices = eval_chunks[offset]
        per_group = min(
            maximum_per_peer,
            len(order) // 2,
            len(calibration_indices),
            len(evaluation_indices),
        )
        if per_group < 1:
            raise ValueError("Each peer needs at least two member rows for membership audit")
        cal_member = peer.subset(order[:per_group])
        eval_member = peer.subset(order[per_group : 2 * per_group])
        member_cal_ids.update(cal_member.source_ids)
        member_eval_ids.update(eval_member.source_ids)
        cal_nonmember = calibration_nonmembers.subset(calibration_indices[:per_group])
        eval_nonmember = evaluation_nonmembers.subset(evaluation_indices[:per_group])
        _, cal_member_loss = predict(states[peer_id], cal_member, hidden_dim)
        _, eval_member_loss = predict(states[peer_id], eval_member, hidden_dim)
        _, cal_nonmember_loss = predict(states[peer_id], cal_nonmember, hidden_dim)
        _, eval_nonmember_loss = predict(states[peer_id], eval_nonmember, hidden_dim)
        cal_member_scores.extend((-cal_member_loss).tolist())
        eval_member_scores.extend((-eval_member_loss).tolist())
        cal_nonmember_scores.extend((-cal_nonmember_loss).tolist())
        eval_nonmember_scores.extend((-eval_nonmember_loss).tolist())
    if member_cal_ids.intersection(member_eval_ids):
        raise ValueError("Membership calibration and evaluation member rows overlap")
    nonmember_ids = set(calibration_nonmembers.source_ids) | set(evaluation_nonmembers.source_ids)
    if member_cal_ids.intersection(nonmember_ids) or member_eval_ids.intersection(nonmember_ids):
        raise ValueError("Membership members and nonmembers overlap")
    return membership_from_scores(
        np.asarray(cal_member_scores),
        np.asarray(cal_nonmember_scores),
        np.asarray(eval_member_scores),
        np.asarray(eval_nonmember_scores),
    )
