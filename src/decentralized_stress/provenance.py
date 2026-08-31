from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def source_paths(root: Path) -> list[Path]:
    paths = [root / "pyproject.toml", root / "data" / "source_lock.json"]
    paths.extend(sorted((root / "src" / "decentralized_stress").glob("*.py")))
    paths.extend(sorted((root / "configs").glob("*.json")))
    return [path for path in paths if path.is_file()]


def source_tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    root = root.resolve()
    entries = []
    for path in source_paths(root):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), entries


def build_file_manifest(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted((value.resolve() for value in paths), key=lambda value: value.as_posix()):
        if not path.is_relative_to(root):
            raise ValueError(f"Manifest path escapes root: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return entries


def write_result_manifest(root: Path, result_dir: Path) -> Path:
    root = root.resolve()
    result_dir = result_dir.resolve()
    if not result_dir.is_relative_to(root / "results"):
        raise ValueError("Result directory must remain under results/")
    manifest_path = result_dir / "result_manifest.json"
    files = [path for path in result_dir.rglob("*") if path.is_file() and path != manifest_path]
    payload = {
        "schema_version": "1.0",
        "result_directory": result_dir.relative_to(root).as_posix(),
        "result_count": len(files),
        "results": build_file_manifest(files, root),
    }
    write_json(manifest_path, payload)
    return manifest_path


def verify_result_manifest(root: Path, result_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    result_dir = result_dir.resolve()
    manifest_path = result_dir / "result_manifest.json"
    manifest = read_json(manifest_path)
    entries = manifest.get("results", [])
    if int(manifest.get("result_count", -1)) != len(entries):
        raise ValueError("result_count does not match result entries")
    listed: set[str] = set()
    for entry in entries:
        relative = str(entry["path"])
        if relative in listed:
            raise ValueError(f"Duplicate result path: {relative}")
        listed.add(relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(result_dir):
            raise ValueError(f"Result path escapes output directory: {relative}")
        if not path.is_file():
            raise ValueError(f"Missing result file: {relative}")
        if sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"SHA-256 mismatch: {relative}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Byte-size mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in result_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if listed != actual:
        raise ValueError(
            f"Result manifest coverage mismatch: missing={sorted(actual-listed)}, unexpected={sorted(listed-actual)}"
        )
    run_manifest = read_json(result_dir / "run_manifest.json")
    current_tree, _ = source_tree_digest(root)
    if str(run_manifest.get("source_tree_sha256")) != current_tree:
        raise ValueError("Source tree differs from the benchmark run")
    config_path = (root / str(run_manifest["config_path"])).resolve()
    if not config_path.is_relative_to(root):
        raise ValueError("Run-manifest configuration path escapes repository root")
    if not config_path.is_file() or sha256_file(config_path) != str(run_manifest["config_sha256"]):
        raise ValueError("Benchmark configuration differs from the run manifest")
    for prefix in ("dataset_source", "partition", "synthetic"):
        path = (root / str(run_manifest[f"{prefix}_manifest_path"])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{prefix} manifest path is invalid")
        if sha256_file(path) != str(run_manifest[f"{prefix}_manifest_sha256"]):
            raise ValueError(f"{prefix} manifest differs from the benchmark run")
    source_manifest = read_json(
        (root / str(run_manifest["dataset_source_manifest_path"])).resolve()
    )
    if source_manifest.get("kind") == "external_real_dataset":
        archive_path = (root / str(source_manifest["archive_path"])).resolve()
        if not archive_path.is_relative_to(root) or not archive_path.is_file():
            raise ValueError("Pinned external dataset archive is missing")
        if sha256_file(archive_path) != str(source_manifest["archive_sha256"]):
            raise ValueError("Pinned external dataset archive differs from the benchmark run")
    return {
        "valid": True,
        "results_verified": len(entries),
        "source_tree_sha256": current_tree,
        "result_manifest_sha256": sha256_file(manifest_path),
    }
