import json
import logging
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.filter import samplePETs
from cLoops2.quant import quantLoops
from cLoops2.trans_stats import read_trans_loop_results


class QuantTransTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        self.source = source
        cis = np.asarray([[i, 100 + i] for i in range(4)], dtype=np.int64)
        trans = np.asarray([[200 + i, 300 + i] for i in range(6)],
                           dtype=np.int64)
        cis_path = source / "chr1-chr1.ixy"
        trans_path = source / "chr1-chr2.ixy"
        joblib.dump(cis, str(cis_path))
        joblib.dump(trans, str(trans_path))
        (source / "petMeta.json").write_text(json.dumps({
            "Unique PETs": 10,
            "Unique Cis PETs": 4,
            "Unique Trans PETs": 6,
            "Total PETs": 10,
            "Total Cis PETs": 4,
            "Total Trans PETs": 6,
            "Retention": {
                "retain trans": True,
                "retained categories": ["cis", "trans"],
            },
            "data": {
                "cis": {"chr1-chr1": {"ixy": str(cis_path)}},
                "trans": {"chr1-chr2": {"ixy": str(trans_path)}},
            },
        }))
        self.sampled = self.root / "sampled-trans"
        samplePETs(str(source), str(self.sampled), 8,
                   mode="trans", seed=123)
        self.logger = logging.getLogger("quant-trans-test")
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        self.temporary.cleanup()

    def test_projection_uses_logical_depth_not_physical_trans_rows(self):
        candidate = self.root / "candidate.tsv"
        candidate.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\tdistance\n"
            "all-trans\tchr1\t0\t1000\tchr2\t0\t1000\t-1\n")
        prefix = str(self.root / "quantified")
        results = quantLoops(
            str(self.sampled),
            str(candidate),
            prefix,
            self.logger,
            mode="trans",
            transLocalPad=100,
            transTestScope="global",
        )
        self.assertEqual(len(results), 1)
        result = results[0]
        physical = json.loads(
            (self.sampled / "petMeta.json").read_text())["Unique PETs"]
        self.assertEqual(result.pets, physical)
        self.assertEqual(result.library_depth, 8)
        self.assertAlmostEqual(result.library_rpm,
                               float(physical) / 8 * 1e6)
        self.assertEqual(result.inference_mode, "descriptive_quant")
        self.assertIsNone(result.bh_adjusted_p)
        self.assertIsNone(result.significant)
        loaded = read_trans_loop_results(
            prefix + "_trans_quantified_loops.txt")
        self.assertEqual(loaded, results)

    def test_all_mode_splits_mixed_cis_and_trans_rows(self):
        candidate = self.root / "mixed.tsv"
        candidate.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\tdistance\n"
            "cis-loop\tchr1\t0\t20\tchr1\t90\t130\t100\n"
            "trans-loop\tchr1\t0\t1000\tchr2\t0\t1000\t-1\n")
        prefix = str(self.root / "mixed-quant")
        results = quantLoops(
            str(self.source), str(candidate), prefix, self.logger,
            mode="all", offp=True, transLocalPad=100,
            transTestScope="global",
        )
        self.assertEqual([result.loop_id for result in results],
                         ["trans-loop"])
        self.assertTrue(Path(prefix + "_loops.txt").is_file())
        self.assertTrue(
            Path(prefix + "_trans_quantified_loops.txt").is_file())


if __name__ == "__main__":
    unittest.main()
