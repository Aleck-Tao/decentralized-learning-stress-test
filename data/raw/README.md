# Raw data boundary

Place the official UCI Adult archive at `data/raw/adult.zip`. The loader rejects an archive whose SHA-256 differs from the value pinned in `data/source_lock.json`, or which lacks `adult.data`, `adult.names`, or `adult.test`.

The archive is external evidence under CC BY 4.0, not authored by this repository. Generated partitions and synthetic rows must never be placed in this directory.

