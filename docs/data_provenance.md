# Data provenance

## External source

`data/source_lock.json` pins the UCI Adult DOI, official archive URL, CC BY 4.0 licence, expected archive path, SHA-256, and required ZIP members. The loader hashes the archive before opening it and separately records hashes for `adult.data`, `adult.names`, and `adult.test`.

No download command silently replaces the archive. A changed upstream file requires a reviewed source-lock change.

## Transformation chain

```text
official Adult ZIP + archive SHA-256
→ member hashes and row parser
→ fixed numeric transforms + per-field category hashing
→ immutable train / global-test / MIA-calibration / MIA-evaluation split
→ peer train/holdout source-ID hashes and per-class counts
→ local synthetic output and parent-ID-set hashes
→ topology, threat model, optimiser and experiment config
→ round and run metrics
→ summary, report, run manifest and result manifest
```

Generated partition manifests explicitly call the peer boundaries simulated. For every peer they record row counts, ordered train and holdout source-ID hashes, and separate `0`/`1` label counts for both splits. The counts make single-class local holdouts directly auditable; the hashes bind the ordered source-row assignments without publishing duplicate row content. Synthetic manifests label every generator entry non-DP and record whether its labels were poisoned.

## Result binding

`run_manifest.json` records hashes for the package/config source tree, selected benchmark configuration, source manifest, partition manifest, and synthetic manifest, together with Python, NumPy, PyTorch, platform, byte-order, CPU-device, and thread-count metadata. `result_manifest.json` records the exact result-file set, byte sizes, and SHA-256 values. `dlstress verify` checks both output integrity and whether the current source/config still match the recorded run.

Hashes detect change; they are not signatures, authorship proof, data realism, or privacy evidence.

## Reproducibility boundary

Partition, augmentation, attack, and optimiser seeds are versioned for controlled research repetition. PyTorch/BLAS floating-point output may differ slightly across systems. The repository therefore does not call cross-platform metric files byte deterministic; tests assert split integrity, finite values, protocol invariants, and numeric tolerances.
