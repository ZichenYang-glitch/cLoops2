import json
import logging
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.io import combineDirs, combineIxys
from cLoops2.filter import samplePETs
from cLoops2.metadata import (CAP_FORMAL_INFERENCE,
                              build_library_context)


class CombineKeepTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.first = self.root / "a" / "chr1-chr2.ixy"
        self.second = self.root / "b" / "chr1-chr2.ixy"
        self.first.parent.mkdir()
        self.second.parent.mkdir()
        joblib.dump(np.asarray([[1, 10], [1, 10], [2, 20]],
                              dtype=np.int64), str(self.first))
        joblib.dump(np.asarray([[1, 10], [3, 30]], dtype=np.int64),
                    str(self.second))

    def tearDown(self):
        self.temporary.cleanup()

    def _combine(self, keep):
        outdir = self.root / ("keep-%s" % keep)
        outdir.mkdir()
        combineIxys("chr1-chr2", [str(self.first), str(self.second)],
                    str(outdir), keep=keep)
        return joblib.load(str(outdir / "chr1-chr2.ixy"))

    def test_keep_zero_really_preserves_all_coordinate_multiplicity(self):
        output = self._combine(0)
        self.assertEqual(output.shape, (5, 2))
        self.assertEqual(np.count_nonzero(np.all(output == [1, 10], axis=1)),
                         3)

    def test_positive_keep_caps_each_coordinate(self):
        self.assertEqual(self._combine(1).shape[0], 3)
        self.assertEqual(self._combine(2).shape[0], 4)

    def test_negative_keep_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._combine(-1)

    def test_trans_only_directory_combine_uses_the_trans_source_path(self):
        inputs = []
        expected = []
        for i in range(2):
            directory = self.root / ("trans-input-%s" % i)
            directory.mkdir()
            matrix = np.asarray([[10 + i, 100 + i]], dtype=np.int64)
            expected.append(matrix[0])
            path = directory / "chr1-chr2.ixy"
            joblib.dump(matrix, str(path))
            (directory / "petMeta.json").write_text(json.dumps({
                "Unique PETs": 1,
                "Unique Cis PETs": 0,
                "Unique Trans PETs": 1,
                "Total PETs": 1,
                "Total Cis PETs": 0,
                "Total Trans PETs": 1,
                "Retention": {
                    "retain trans": True,
                    "retained categories": ["cis", "trans"],
                    "chromosome whitelist": ["chr1", "chr2"],
                    "cut": 0,
                    "mcut": -1,
                },
                "data": {
                    "cis": {},
                    "trans": {"chr1-chr2": {"ixy": str(path)}},
                },
            }))
            inputs.append(str(directory))
        output = self.root / "combined-trans"
        logger = logging.getLogger("combine-trans-only-test")
        logger.addHandler(logging.NullHandler())
        combineDirs(inputs, str(output), logger, keep=0, cpu=1)
        observed = joblib.load(str(output / "chr1-chr2.ixy"))
        np.testing.assert_array_equal(
            observed, np.asarray(expected, dtype=np.int64))
        meta = json.loads((output / "petMeta.json").read_text())
        self.assertEqual(meta["Transformation"]["validity"], "valid")
        context = build_library_context(
            meta, actual_counts={"trans:chr1-chr2": 2})
        self.assertEqual(context.validity, "valid", context.reasons)
        self.assertNotIn(CAP_FORMAL_INFERENCE, context.capabilities)

        sampled = self.root / "combined-trans-sampled"
        child = samplePETs(str(output), str(sampled), 1,
                           mode="trans", seed=9)
        self.assertFalse(child["replacement_ever"])
        self.assertFalse(child["formal_inference_eligible"])
        self.assertEqual(child["parent_source_kind"], "all_sample")
        child_context = build_library_context(
            json.loads((sampled / "petMeta.json").read_text()),
            actual_counts={"trans:chr1-chr2": 1})
        self.assertNotIn(CAP_FORMAL_INFERENCE,
                         child_context.capabilities)

    def test_combine_projections_does_not_invent_global_depth(self):
        projections = []
        for i in range(2):
            source = self.root / ("mixed-source-%s" % i)
            source.mkdir()
            cis_path = source / "chr1-chr1.ixy"
            trans_path = source / "chr1-chr2.ixy"
            joblib.dump(np.asarray([[i, 10 + i]], dtype=np.int64),
                        str(cis_path))
            joblib.dump(np.asarray([[20 + i, 30 + i]], dtype=np.int64),
                        str(trans_path))
            (source / "petMeta.json").write_text(json.dumps({
                "Unique PETs": 2,
                "Unique Cis PETs": 1,
                "Unique Trans PETs": 1,
                "Total PETs": 2,
                "Total Cis PETs": 1,
                "Total Trans PETs": 1,
                "Retention": {
                    "retain trans": True,
                    "retained categories": ["cis", "trans"],
                    "chromosome whitelist": ["chr1", "chr2"],
                    "cut": 0,
                    "mcut": -1,
                },
                "data": {
                    "cis": {"chr1-chr1": {"ixy": str(cis_path)}},
                    "trans": {"chr1-chr2": {"ixy": str(trans_path)}},
                },
            }))
            projection = self.root / ("projection-%s" % i)
            samplePETs(str(source), str(projection), 2,
                       mode="trans", seed=100 + i)
            projections.append(str(projection))

        output = self.root / "combined-projections"
        logger = logging.getLogger("combine-projection-test")
        logger.addHandler(logging.NullHandler())
        combineDirs(projections, str(output), logger, keep=0, cpu=1)
        meta = json.loads((output / "petMeta.json").read_text())
        self.assertEqual(meta["Transformation"]["validity"],
                         "unknown_ancestry")
        self.assertFalse(
            meta["Transformation"]["parents_fully_materialized"])
        context = build_library_context(
            meta, actual_counts={"trans:chr1-chr2": 2})
        self.assertEqual(context.validity, "unknown")
        self.assertIsNone(context.logical_total)
        self.assertNotIn(CAP_FORMAL_INFERENCE, context.capabilities)


if __name__ == "__main__":
    unittest.main()
