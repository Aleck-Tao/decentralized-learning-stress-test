from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


NUMERIC_COLUMNS = (0, 2, 4, 10, 11, 12)
CATEGORICAL_COLUMNS = (1, 3, 5, 6, 7, 8, 13)  # sex is evaluation-only
COLUMN_NAMES = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
)
HASH_BUCKETS = 8


@dataclass(frozen=True)
class Records:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    source_ids: tuple[str, ...]
    synthetic: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.source_ids)
        if self.x.ndim != 2:
            raise ValueError("x must be a two-dimensional array")
        if self.x.shape[0] != count or self.y.shape != (count,):
            raise ValueError("Feature, label, and source-id lengths differ")
        if self.groups.shape != (count,) or self.synthetic.shape != (count,):
            raise ValueError("Group or synthetic marker length differs")
        if not np.isfinite(self.x).all():
            raise ValueError("Features contain non-finite values")
        if not set(np.unique(self.y)).issubset({0.0, 1.0}):
            raise ValueError("Labels must be binary")

    def __len__(self) -> int:
        return len(self.source_ids)

    @property
    def feature_count(self) -> int:
        return int(self.x.shape[1])

    def subset(self, indices: Iterable[int] | np.ndarray) -> "Records":
        selected = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices, dtype=int)
        return Records(
            x=self.x[selected].copy(),
            y=self.y[selected].copy(),
            groups=self.groups[selected].copy(),
            source_ids=tuple(self.source_ids[int(index)] for index in selected),
            synthetic=self.synthetic[selected].copy(),
        )


@dataclass(frozen=True)
class DatasetBundle:
    train_pool: Records
    global_test: Records
    attack_calibration_nonmember: Records
    attack_evaluation_nonmember: Records
    feature_names: tuple[str, ...]
    provenance: dict[str, object]


@dataclass(frozen=True)
class PartitionedData:
    train: dict[int, Records]
    holdout: dict[int, Records]
    scarce_peer_ids: tuple[int, ...]
    partition_manifest: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_value(column: int, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in {COLUMN_NAMES[column]}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value in {COLUMN_NAMES[column]}")
    if column == 0:
        return float(np.clip(value / 100.0, 0.0, 1.0))
    if column == 2:
        return float(np.clip(np.log1p(max(value, 0.0)) / np.log1p(2_000_000.0), 0.0, 1.0))
    if column == 4:
        return float(np.clip(value / 16.0, 0.0, 1.0))
    if column == 10:
        return float(np.clip(np.log1p(max(value, 0.0)) / np.log1p(100_000.0), 0.0, 1.0))
    if column == 11:
        return float(np.clip(np.log1p(max(value, 0.0)) / np.log1p(5_000.0), 0.0, 1.0))
    if column == 12:
        return float(np.clip(value / 100.0, 0.0, 1.0))
    raise AssertionError("Unexpected numeric column")


def _categorical_bucket(column: int, value: str) -> int:
    token = f"{COLUMN_NAMES[column]}:{value.strip() or '?'}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") % HASH_BUCKETS


def _encode_rows(text: str, member: str) -> Records:
    features: list[list[float]] = []
    labels: list[float] = []
    groups: list[int] = []
    source_ids: list[str] = []
    reader = csv.reader(io.StringIO(text))
    for line_number, row in enumerate(reader, start=1):
        if not row or (row[0].strip().startswith("|")):
            continue
        row = [value.strip() for value in row]
        if len(row) != 15:
            raise ValueError(f"{member}:{line_number} has {len(row)} columns, expected 15")
        numeric = [_numeric_value(column, row[column]) for column in NUMERIC_COLUMNS]
        categorical = [0.0] * (len(CATEGORICAL_COLUMNS) * HASH_BUCKETS)
        for field_index, column in enumerate(CATEGORICAL_COLUMNS):
            bucket = _categorical_bucket(column, row[column])
            categorical[field_index * HASH_BUCKETS + bucket] = 1.0
        label = row[14].rstrip(".")
        if label not in {"<=50K", ">50K"}:
            raise ValueError(f"{member}:{line_number} has unknown label {row[14]!r}")
        sex = row[9]
        group = 0 if sex == "Female" else 1 if sex == "Male" else -1
        features.append(numeric + categorical)
        labels.append(float(label == ">50K"))
        groups.append(group)
        source_ids.append(f"{member}:{line_number}")
    if not features:
        raise ValueError(f"No records found in {member}")
    return Records(
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.float32),
        groups=np.asarray(groups, dtype=np.int8),
        source_ids=tuple(source_ids),
        synthetic=np.zeros(len(features), dtype=bool),
    )


def _feature_names() -> tuple[str, ...]:
    names = [COLUMN_NAMES[column] for column in NUMERIC_COLUMNS]
    for column in CATEGORICAL_COLUMNS:
        names.extend(f"{COLUMN_NAMES[column]}_hash_{bucket}" for bucket in range(HASH_BUCKETS))
    return tuple(names)


def _split_nonmembers(records: Records, seed: int = 20260924) -> tuple[Records, Records, Records]:
    if len(records) < 15:
        raise ValueError("At least 15 test records are required for strict evaluation splits")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))
    global_end = max(1, int(len(indices) * 0.60))
    calibration_end = max(global_end + 1, int(len(indices) * 0.80))
    return (
        records.subset(indices[:global_end]),
        records.subset(indices[global_end:calibration_end]),
        records.subset(indices[calibration_end:]),
    )


def load_adult(root: Path) -> DatasetBundle:
    root = root.resolve()
    lock_path = root / "data" / "source_lock.json"
    with lock_path.open("r", encoding="utf-8") as stream:
        lock = json.load(stream)
    archive_path = (root / str(lock["archive_path"])).resolve()
    if not archive_path.is_relative_to(root):
        raise ValueError("Adult archive path escapes repository root")
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing {archive_path}. Place the pinned official UCI archive at data/raw/adult.zip."
        )
    actual_hash = sha256_file(archive_path)
    expected_hash = str(lock["archive_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(f"Adult archive SHA-256 mismatch: {actual_hash}")
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        required = set(str(value) for value in lock["required_members"])
        if not required.issubset(members):
            raise ValueError(f"Adult archive missing members: {sorted(required - members)}")
        train_text = archive.read("adult.data").decode("utf-8")
        test_text = archive.read("adult.test").decode("utf-8")
        member_hashes = {
            member: hashlib.sha256(archive.read(member)).hexdigest() for member in sorted(required)
        }
    train = _encode_rows(train_text, "adult.data")
    official_test = _encode_rows(test_text, "adult.test")
    global_test, attack_calibration, attack_evaluation = _split_nonmembers(official_test)
    assert_disjoint(
        {
            "train_pool": train,
            "global_test": global_test,
            "attack_calibration_nonmember": attack_calibration,
            "attack_evaluation_nonmember": attack_evaluation,
        }
    )
    provenance = {
        "kind": "external_real_dataset",
        "dataset_id": lock["dataset_id"],
        "title": lock["title"],
        "doi": lock["doi"],
        "license": lock["license"],
        "source_url": lock["source_url"],
        "archive_path": str(lock["archive_path"]),
        "archive_sha256": actual_hash,
        "archive_bytes": archive_path.stat().st_size,
        "member_sha256": member_hashes,
        "train_rows": len(train),
        "official_test_rows": len(official_test),
        "preprocessing": "fixed numeric transforms plus per-field SHA-256 categorical hashing; sex held out for evaluation only",
    }
    return DatasetBundle(
        train_pool=train,
        global_test=global_test,
        attack_calibration_nonmember=attack_calibration,
        attack_evaluation_nonmember=attack_evaluation,
        feature_names=_feature_names(),
        provenance=provenance,
    )


def make_fixture(seed: int = 17, train_rows: int = 640, test_rows: int = 240) -> DatasetBundle:
    if train_rows < 80 or test_rows < 60:
        raise ValueError("Fixture is too small for strict partitions")
    rng = np.random.default_rng(seed)
    total = train_rows + test_rows
    x = rng.normal(0.0, 1.0, size=(total, 12)).astype(np.float32)
    groups = (x[:, 0] + rng.normal(0.0, 0.8, total) > 0).astype(np.int8)
    logits = 1.4 * x[:, 1] - 1.1 * x[:, 2] + 0.7 * x[:, 3] + 0.25 * groups
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(total) < probabilities).astype(np.float32)
    records = Records(
        x=x,
        y=y,
        groups=groups,
        source_ids=tuple(f"fixture:{index}" for index in range(total)),
        synthetic=np.zeros(total, dtype=bool),
    )
    train = records.subset(np.arange(train_rows))
    official_test = records.subset(np.arange(train_rows, total))
    global_test, attack_calibration, attack_evaluation = _split_nonmembers(official_test, seed + 1)
    assert_disjoint(
        {
            "train_pool": train,
            "global_test": global_test,
            "attack_calibration_nonmember": attack_calibration,
            "attack_evaluation_nonmember": attack_evaluation,
        }
    )
    return DatasetBundle(
        train_pool=train,
        global_test=global_test,
        attack_calibration_nonmember=attack_calibration,
        attack_evaluation_nonmember=attack_evaluation,
        feature_names=tuple(f"fixture_feature_{index}" for index in range(x.shape[1])),
        provenance={
            "kind": "generated_test_fixture",
            "generator": "decentralized_stress.data.make_fixture",
            "seed": seed,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "claim_boundary": "Unit/smoke fixture only; not UCI Adult and not public-administration data",
        },
    )


def assert_disjoint(named_records: dict[str, Records]) -> None:
    seen: dict[str, str] = {}
    for name, records in named_records.items():
        if len(set(records.source_ids)) != len(records.source_ids):
            raise ValueError(f"Duplicate source IDs inside {name}")
        for source_id in records.source_ids:
            if source_id in seen:
                raise ValueError(f"Source row {source_id} occurs in both {seen[source_id]} and {name}")
            seen[source_id] = name


def partition_clients(
    records: Records,
    clients: int,
    *,
    seed: int,
    alpha: float | None,
    scarcity_fraction: float,
    scarce_client_fraction: float,
) -> PartitionedData:
    if clients < 2 or len(records) < clients * 12:
        raise ValueError("Insufficient records for client partitioning")
    rng = np.random.default_rng(seed)
    assignments: list[list[int]] | None = None
    for _ in range(100):
        candidate = [[] for _ in range(clients)]
        for label in (0.0, 1.0):
            indices = np.flatnonzero(records.y == label)
            rng.shuffle(indices)
            if alpha is None:
                parts = np.array_split(indices, clients)
            else:
                proportions = rng.dirichlet(np.full(clients, alpha))
                counts = rng.multinomial(len(indices), proportions)
                boundaries = np.cumsum(counts)[:-1]
                parts = np.split(indices, boundaries)
            for client_id, part in enumerate(parts):
                candidate[client_id].extend(int(index) for index in part)
        if min(len(part) for part in candidate) >= 10:
            assignments = candidate
            break
    if assignments is None:
        raise ValueError("Could not construct non-empty client partitions; increase alpha or data size")

    scarce_count = (
        0
        if scarce_client_fraction <= 0
        else max(1, int(round(clients * scarce_client_fraction)))
    )
    scarce_ids = tuple(
        sorted(int(value) for value in rng.choice(clients, size=scarce_count, replace=False))
    )
    train: dict[int, Records] = {}
    holdout: dict[int, Records] = {}
    manifest_clients: list[dict[str, object]] = []
    for client_id, values in enumerate(assignments):
        values_array = np.asarray(values, dtype=int)
        rng.shuffle(values_array)
        holdout_count = max(2, int(round(len(values_array) * 0.20)))
        holdout_indices = values_array[:holdout_count]
        train_indices = values_array[holdout_count:]
        if client_id in scarce_ids:
            retained = max(6, int(round(len(train_indices) * scarcity_fraction)))
            train_indices = train_indices[:retained]
        train[client_id] = records.subset(train_indices)
        holdout[client_id] = records.subset(holdout_indices)
        train_label_counts = {
            "0": int(np.count_nonzero(train[client_id].y == 0.0)),
            "1": int(np.count_nonzero(train[client_id].y == 1.0)),
        }
        holdout_label_counts = {
            "0": int(np.count_nonzero(holdout[client_id].y == 0.0)),
            "1": int(np.count_nonzero(holdout[client_id].y == 1.0)),
        }
        manifest_clients.append(
            {
                "peer_id": client_id,
                "train_rows": len(train[client_id]),
                "holdout_rows": len(holdout[client_id]),
                "train_label_counts": train_label_counts,
                "holdout_label_counts": holdout_label_counts,
                "scarce": client_id in scarce_ids,
                "train_source_ids_sha256": hashlib.sha256(
                    "\n".join(train[client_id].source_ids).encode("utf-8")
                ).hexdigest(),
                "holdout_source_ids_sha256": hashlib.sha256(
                    "\n".join(holdout[client_id].source_ids).encode("utf-8")
                ).hexdigest(),
            }
        )
    named = {f"peer_{key}_train": value for key, value in train.items()}
    named.update({f"peer_{key}_holdout": value for key, value in holdout.items()})
    assert_disjoint(named)
    return PartitionedData(
        train=train,
        holdout=holdout,
        scarce_peer_ids=scarce_ids,
        partition_manifest={
            "seed": seed,
            "client_count": clients,
            "mode": "iid" if alpha is None else "dirichlet_label_skew",
            "dirichlet_alpha": alpha,
            "scarcity_fraction": scarcity_fraction,
            "scarce_peer_ids": list(scarce_ids),
            "clients": manifest_clients,
            "boundary": "These are simulated peer partitions, not observed organisational boundaries.",
        },
    )


def concatenate_records(first: Records, second: Records) -> Records:
    if first.feature_count != second.feature_count:
        raise ValueError("Cannot concatenate records with different feature dimensions")
    return Records(
        x=np.concatenate([first.x, second.x], axis=0),
        y=np.concatenate([first.y, second.y], axis=0),
        groups=np.concatenate([first.groups, second.groups], axis=0),
        source_ids=first.source_ids + second.source_ids,
        synthetic=np.concatenate([first.synthetic, second.synthetic], axis=0),
    )


def augment_local_non_private(
    records: Records,
    *,
    peer_id: int,
    ratio: float,
    seed: int,
    poison_labels: bool = False,
) -> tuple[Records, dict[str, object]]:
    if ratio <= 0:
        return records, {"generated_rows": 0, "non_dp": True, "poisoned": False}
    if len(records) < 2:
        raise ValueError("At least two local rows are required for augmentation")
    rng = np.random.default_rng(seed)
    count = max(1, int(round(len(records) * ratio)))
    parent_indices = rng.integers(0, len(records), size=count)
    x = records.x[parent_indices].copy()
    numeric_count = min(6, x.shape[1])
    local_scale = np.std(records.x[:, :numeric_count], axis=0)
    jitter = rng.normal(0.0, np.maximum(local_scale, 1e-3) * 0.02, size=(count, numeric_count))
    x[:, :numeric_count] = np.clip(x[:, :numeric_count] + jitter, -4.0, 4.0)
    y = records.y[parent_indices].copy()
    if poison_labels:
        y = 1.0 - y
    synthetic = Records(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        groups=records.groups[parent_indices].copy(),
        source_ids=tuple(f"synthetic:peer-{peer_id}:seed-{seed}:row-{index}" for index in range(count)),
        synthetic=np.ones(count, dtype=bool),
    )
    parent_hash = hashlib.sha256(
        "\n".join(records.source_ids[int(index)] for index in parent_indices).encode("utf-8")
    ).hexdigest()
    manifest = {
        "peer_id": peer_id,
        "generator": "class-preserving local bootstrap with bounded numeric jitter",
        "generated_rows": count,
        "ratio": ratio,
        "seed": seed,
        "parent_source_ids_sha256": parent_hash,
        "poisoned": poison_labels,
        "non_dp": True,
        "privacy_claim": "none",
    }
    return concatenate_records(records, synthetic), manifest
