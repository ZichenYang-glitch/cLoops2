#!/usr/bin/env python
"""Reproduce the pre-v0.2 samplePETs algorithm on synthetic PETs.

This script deliberately contains the small historical algorithm instead of
calling the current implementation.  It records the baseline defects after
the source has been fixed: the ratio denominator is global ``Unique PETs``,
only ``data.cis`` is traversed, every file is independently rounded, and
``take >= rows`` selects with replacement.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def write_fixture(directory):
    directory.mkdir(parents=True)
    cis = np.asarray([[100 + i, 200 + i] for i in range(4)], dtype=np.int64)
    trans = np.asarray([[1000 + i, 2000 + i] for i in range(6)],
                       dtype=np.int64)
    cis_path = directory / "chr1-chr1.ixy"
    trans_path = directory / "chr1-chr2.ixy"
    joblib.dump(cis, str(cis_path))
    joblib.dump(trans, str(trans_path))
    meta = {
        "Unique PETs": 10,
        "data": {
            "cis": {"chr1-chr1": {"ixy": str(cis_path.resolve())}},
            "trans": {"chr1-chr2": {"ixy": str(trans_path.resolve())}},
        },
    }
    (directory / "petMeta.json").write_text(json.dumps(meta))
    return meta


def legacy_sample(meta, output, target, seed):
    output.mkdir()
    ratio = float(target) / meta["Unique PETs"]
    np.random.seed(seed)
    written = {}
    for key, entry in meta["data"]["cis"].items():
        matrix = joblib.load(entry["ixy"])
        take = int(round(matrix.shape[0] * ratio))
        replace = take >= matrix.shape[0]
        indexes = np.random.choice(
            matrix.shape[0], size=take, replace=replace)
        sampled = matrix[indexes, :]
        path = output / (key + ".ixy")
        joblib.dump(sampled, str(path))
        written[key] = sampled
    return ratio, written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="tmp_samplepets_legacy_repro")
    parser.add_argument("--tot", type=int, default=9)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if root.exists():
        raise SystemExit(
            "refusing to overwrite existing directory %s; choose --root" %
            root)
    meta = write_fixture(root / "input")
    ratio, written = legacy_sample(
        meta, root / "legacy-output", args.tot, args.seed)
    rows = sum(matrix.shape[0] for matrix in written.values())
    distinct = sum(np.unique(matrix, axis=0).shape[0]
                   for matrix in written.values())
    print(json.dumps({
        "requested PETs": args.tot,
        "legacy ratio": ratio,
        "legacy output rows": rows,
        "legacy distinct rows": distinct,
        "legacy output files": [key + ".ixy" for key in sorted(written)],
        "trans output missing": True,
        "target mismatch": rows != args.tot,
        "replacement path used": any(
            int(round(matrix.shape[0] * ratio)) >= matrix.shape[0]
            for matrix in [joblib.load(
                meta["data"]["cis"][key]["ixy"])
                for key in meta["data"]["cis"]]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
