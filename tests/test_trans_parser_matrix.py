import os
import tempfile
import unittest

import numpy as np

from cLoops2.cmat import getTransObsMat, getTransObsMatCOO
from cLoops2.io import parseTxt2Loops


class ParseTransLoopsTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.loop_file = os.path.join(self.tmpdir.name, "loops.txt")
        with open(self.loop_file, "w") as handle:
            handle.write(
                "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\t"
                "distance\n")
            handle.write("cis-near\tchr1\t10\t20\tchr1\t60\t70\t50\n")
            handle.write("cis-far\tchr1\t10\t20\tchr1\t510\t520\t500\n")
            handle.write("trans\tchr1\t90\t110\tchr2\t90\t110\t-1\n")
            handle.write("trans-na\tchr1\t200\t220\tchr3\t300\t320\tNA\n")

    def tearDown(self):
        self.tmpdir.cleanup()

    @staticmethod
    def _ids(loops):
        return {loop.id for values in loops.values() for loop in values}

    def test_default_remains_cis_and_applies_distance_cut(self):
        loops = parseTxt2Loops(self.loop_file, cut=100)
        self.assertEqual(self._ids(loops), {"cis-far"})

    def test_cis_maximum_distance(self):
        loops = parseTxt2Loops(self.loop_file, cut=0, mcut=100, mode="cis")
        self.assertEqual(self._ids(loops), {"cis-near"})

    def test_trans_ignores_cis_distance_filters(self):
        loops = parseTxt2Loops(self.loop_file,
                               cut=1000000,
                               mcut=1,
                               mode="trans")
        self.assertEqual(self._ids(loops), {"trans", "trans-na"})
        parsed = [loop for values in loops.values() for loop in values]
        self.assertTrue(all(not loop.cis for loop in parsed))
        self.assertTrue(all(loop.distance == -1 for loop in parsed))

    def test_all_routes_each_category_with_cis_only_cut(self):
        loops = parseTxt2Loops(self.loop_file, cut=100, mode="all")
        self.assertEqual(self._ids(loops), {"cis-far", "trans", "trans-na"})

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            parseTxt2Loops(self.loop_file, mode="invalid")

    def test_versioned_trans_result_prefix_is_accepted(self):
        versioned = os.path.join(self.tmpdir.name, "versioned.tsv")
        with open(versioned, "w") as handle:
            handle.write("#cLoops2-trans-loop-result\t2\n")
            handle.write(
                "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\t"
                "distance(bp)\textra\n")
            handle.write("v2\tchr1\t10\t20\tchr2\t30\t40\t-1\tNA\n")
        loops = parseTxt2Loops(versioned, mode="trans")
        self.assertEqual(self._ids(loops), {"v2"})


class RectangularTransMatrixTest(unittest.TestCase):

    def setUp(self):
        self.xy = np.asarray([
            [0, 20],
            [0, 20],
            [20, 0],
            [30, 30],
            [31, 5],
            [-1, 5],
        ], dtype=np.int64)

    def test_dense_matrix_is_rectangular_and_not_mirrored(self):
        matrix = getTransObsMat(self.xy, 0, 30, 0, 30, 10, 15)
        self.assertEqual(matrix.shape, (4, 3))
        self.assertEqual(int(matrix.sum()), 4)
        self.assertEqual(int(matrix[0, 1]), 2)
        self.assertEqual(int(matrix[2, 0]), 1)
        self.assertEqual(int(matrix[3, 2]), 1)
        self.assertEqual(int(matrix[1, 0]), 0)

    def test_equal_numeric_bins_are_not_doubled(self):
        matrix = getTransObsMat(np.asarray([[10, 10]], dtype=np.int64),
                                0, 20, 0, 20, 10)
        self.assertEqual(int(matrix.sum()), 1)
        self.assertEqual(int(matrix[1, 1]), 1)

    def test_sparse_and_dense_results_match(self):
        sparse = getTransObsMatCOO(self.xy, 0, 30, 0, 30, 10, 15)
        dense = getTransObsMat(self.xy, 0, 30, 0, 30, 10, 15)
        np.testing.assert_array_equal(sparse.toarray(), dense)

    def test_dense_limit_is_checked_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "exceeding the limit"):
            getTransObsMat(np.empty((0, 2), dtype=np.int64),
                           0,
                           999,
                           0,
                           999,
                           1,
                           max_dense_cells=999999)

    def test_invalid_matrix_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            getTransObsMat(np.asarray([1, 2], dtype=np.int64),
                           0, 10, 0, 10, 10)


if __name__ == "__main__":
    unittest.main()
