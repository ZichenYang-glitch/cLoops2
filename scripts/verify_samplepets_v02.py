#!/usr/bin/env python
"""Create a tiny cLoops2 dataset and verify samplePETs v0.2 output.

Run this script from the repository root.  It is intentionally small enough to
use for a before/after check without downloading external data.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from cLoops2.filter import samplePETs


def _write_fixture(input_dir):
    input_dir.mkdir(parents=True)
    cis = np.array(
        [[101, 201], [102, 202], [103, 203], [104, 204]], dtype=np.int64
    )
    trans = np.array(
        [
            [1001, 2001],
            [1002, 2002],
            [1003, 2003],
            [1004, 2004],
            [1005, 2005],
            [1006, 2006],
        ],
        dtype=np.int64,
    )
    cis_path = input_dir / "chr1-chr1.ixy"
    trans_path = input_dir / "chr1-chr2.ixy"
    joblib.dump(cis, cis_path)
    joblib.dump(trans, trans_path)
    meta = {
        "Unique PETs": 10,
        "Unique Cis PETs": 4,
        "Unique Trans PETs": 6,
        "Total PETs": 10,
        "Total Cis PETs": 4,
        "Total Trans PETs": 6,
        "Retention": {
            "retain trans": True,
            "retained categories": ["cis", "trans"],
            "chromosome whitelist": ["chr1", "chr2"],
            "cut": 0,
            "mcut": -1,
        },
        "data": {
            "cis": {"chr1-chr1": {"ixy": str(cis_path.resolve())}},
            "trans": {"chr1-chr2": {"ixy": str(trans_path.resolve())}},
        },
    }
    (input_dir / "petMeta.json").write_text(json.dumps(meta))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="tmp_samplepets_repro")
    parser.add_argument("--tot", type=int, default=3)
    parser.add_argument("--mode", choices=("cis", "trans", "all"),
                        default="trans")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if root.exists():
        raise SystemExit(
            "refusing to overwrite existing directory %s; choose --root" %
            root)
    input_dir = root / "input"
    output_dir = root / "output"
    _write_fixture(input_dir)
    samplePETs(str(input_dir), str(output_dir), args.tot, cpu=1,
               mode=args.mode, seed=args.seed)

    output_meta = json.loads((output_dir / "petMeta.json").read_text())
    output_arrays = [joblib.load(path) for path in sorted(output_dir.glob("*.ixy"))]
    output_rows = sum(array.shape[0] for array in output_arrays)
    distinct_rows = sum(
        np.unique(array, axis=0).shape[0] for array in output_arrays
    )
    summary = {
        "requested PETs": args.tot,
        "mode": args.mode,
        "seed": args.seed,
        "output Unique PETs": output_meta["Unique PETs"],
        "output rows": output_rows,
        "distinct coordinate rows": distinct_rows,
        "cis entries": sorted(output_meta["data"]["cis"]),
        "trans entries": sorted(output_meta["data"]["trans"]),
        "ixy files": sorted(path.name for path in output_dir.glob("*.ixy")),
        "Sampling": output_meta["Sampling"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
