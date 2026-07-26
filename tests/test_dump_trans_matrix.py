import json
import logging
import gzip
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cLoops2.dump import (_iter_bidirectional_intervals,
                          _select_dump_records, ixy2bed, ixy2bedpe,
                          ixy2transmat)
from cLoops2.plot import plotTransMatrix


class DumpTransMatrixTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.predir = self.root / "input"
        self.predir.mkdir()
        self.xy = np.asarray([
            [0, 20],
            [0, 20],
            [20, 0],
            [30, 30],
        ], dtype=np.int64)
        path = self.predir / "chr1-chr2.ixy"
        joblib.dump(self.xy, str(path))
        (self.predir / "petMeta.json").write_text(json.dumps({
            "Unique PETs": 4,
            "data": {
                "cis": {},
                "trans": {
                    "chr1-chr2": {
                        "ixy": str(path),
                        "chromX": "chr1",
                        "chromY": "chr2",
                        "record_id": "trans:chr1-chr2",
                    },
                },
            },
        }))
        self.logger = logging.getLogger("dump-trans-matrix-test")
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        self.temporary.cleanup()

    def _dump(self, name, **kwargs):
        prefix = str(self.root / name)
        path = ixy2transmat(
            str(self.predir),
            prefix,
            self.logger,
            chrom="chr1-chr2",
            x_start=0,
            x_end=30,
            y_start=0,
            y_end=30,
            x_res=10,
            y_res=15,
            **kwargs
        )
        return Path(path)

    def test_dense_observed_is_rectangular_without_mirroring(self):
        path = self._dump("observed", method="obs")
        matrix = pd.read_csv(path, sep="\t", index_col=0).values
        self.assertEqual(matrix.shape, (4, 3))
        self.assertEqual(int(matrix.sum()), 4)
        self.assertEqual(int(matrix[0, 1]), 2)
        self.assertEqual(int(matrix[2, 0]), 1)
        self.assertEqual(int(matrix[1, 0]), 0)

    def test_pair_and_window_oe_are_explicit_distinct_outputs(self):
        pair = pd.read_csv(self._dump("pair", method="pair_oe"),
                           sep="\t", index_col=0).values
        window = pd.read_csv(self._dump("window", method="window_oe"),
                             sep="\t", index_col=0).values
        self.assertEqual(pair.shape, window.shape)
        self.assertTrue(np.isfinite(pair).all())
        self.assertTrue(np.isfinite(window).all())

    def test_sparse_coo_preserves_only_observed_cells(self):
        path = self._dump("sparse", method="obs", sparse=True)
        table = pd.read_csv(path, sep="\t")
        self.assertEqual(int(table["count"].sum()), 4)
        self.assertEqual(table.shape[0], 3)
        self.assertEqual(set(table["chromX"]), {"chr1"})
        self.assertEqual(set(table["chromY"]), {"chr2"})

    def test_dense_guard_fails_before_large_allocation(self):
        with self.assertRaisesRegex(ValueError, "exceeding the limit"):
            self._dump("guard", method="obs", max_dense_cells=11)

    def test_rectangular_plot_uses_two_axes(self):
        prefix = str(self.root / "plot")
        matrix = plotTransMatrix(
            str(self.predir / "chr1-chr2.ixy"),
            prefix,
            x_start=0,
            x_end=30,
            y_start=0,
            y_end=30,
            x_res=10,
            y_res=15,
            method="obs",
            max_dense_cells=100,
        )
        self.assertEqual(matrix.shape, (4, 3))
        self.assertEqual(int(matrix.sum()), 4)
        self.assertTrue(Path(prefix + "_trans_matrix.pdf").is_file())

    def test_bedpe_trans_mode_preserves_both_chromosome_axes(self):
        prefix = str(self.root / "bedpe")
        ixy2bedpe(
            str(self.predir), prefix, self.logger, mode="trans", cut=100,
            ext=1)
        with gzip.open(prefix + "_PETs.bedpe.gz", "rt") as handle:
            rows = [line.rstrip().split("\t") for line in handle]
        self.assertEqual(len(rows), 4)
        self.assertEqual({row[0] for row in rows}, {"chr1"})
        self.assertEqual({row[3] for row in rows}, {"chr2"})

    def test_bed_trans_mode_does_not_assign_y_end_to_chrom_x(self):
        prefix = str(self.root / "bed")
        ixy2bed(str(self.predir), prefix, self.logger, mode="trans", ext=0)
        with gzip.open(prefix + "_reads.bed.gz", "rt") as handle:
            rows = [line.rstrip().split("\t") for line in handle]
        by_chrom = {}
        for chrom, start, end in rows:
            by_chrom.setdefault(chrom, set()).add(int(start))
        self.assertEqual(by_chrom["chr1"], {0, 20, 30})
        self.assertEqual(by_chrom["chr2"], {0, 20, 30})

    def test_browser_reverse_record_swaps_chromosomes_and_coordinates(self):
        records = _select_dump_records(str(self.predir), "trans")
        rows = list(_iter_bidirectional_intervals(records, ext=1))
        self.assertEqual(rows[0], ("chr1", 0, 1, "chr2", 19, 21))
        self.assertEqual(rows[1], ("chr2", 19, 21, "chr1", 0, 1))


if __name__ == "__main__":
    unittest.main()
