import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.callTransLoops import runTransDBSCANLoops


class TransCandidateDiscoveryTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, name, rows):
        path = self.root / name
        joblib.dump(np.asarray(rows, dtype=np.int64).reshape((-1, 2)),
                    str(path))
        return path

    def test_cluster_coordinates_are_axis_specific_and_deterministic(self):
        path = self._write("chr1-chr2.ixy", [
            [100, 1000],
            [105, 1005],
            [110, 1010],
            [115, 1015],
            [120, 1020],
            [5000, 8000],
        ])
        key, loops = runTransDBSCANLoops(str(path), eps=50, minPts=3)
        self.assertEqual(key, "chr1-chr2")
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        self.assertEqual((loop.chromX, loop.x_start, loop.x_end),
                         ("chr1", 100, 120))
        self.assertEqual((loop.chromY, loop.y_start, loop.y_end),
                         ("chr2", 1000, 1020))
        self.assertEqual(loop.rab, 5)
        _, repeated = runTransDBSCANLoops(str(path), eps=50, minPts=3)
        self.assertEqual(
            [(x.x_start, x.x_end, x.y_start, x.y_end) for x in loops],
            [(x.x_start, x.x_end, x.y_start, x.y_end) for x in repeated],
        )

    def test_empty_and_too_small_pairs_are_skipped_before_dbscan(self):
        empty = self._write("chr1-chr3.ixy", [])
        small = self._write("chr2-chr3.ixy", [[1, 10], [2, 11]])
        self.assertEqual(runTransDBSCANLoops(str(empty), 10, 3)[1], [])
        self.assertEqual(runTransDBSCANLoops(str(small), 10, 3)[1], [])

    def test_cis_input_and_invalid_parameters_are_rejected(self):
        cis = self._write("chr1-chr1.ixy", [[1, 10], [2, 11], [3, 12]])
        with self.assertRaisesRegex(ValueError, "trans"):
            runTransDBSCANLoops(str(cis), 10, 3)
        trans = self._write("chr1-chr2.ixy", [[1, 10], [2, 11], [3, 12]])
        with self.assertRaisesRegex(ValueError, "positive"):
            runTransDBSCANLoops(str(trans), 0, 3)


if __name__ == "__main__":
    unittest.main()
