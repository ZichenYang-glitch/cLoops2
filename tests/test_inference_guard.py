import json
import logging
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.callCisLoops import callCisLoops
from cLoops2.callTransLoops import callTransLoops
from cLoops2.filter import samplePETs
from cLoops2.quant import quantLoops, quantPeaks


class ReplacementInferenceGuardTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        cis = np.asarray([[10 + i, 100 + i] for i in range(4)],
                         dtype=np.int64)
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
        self.upsampled = self.root / "upsampled"
        samplePETs(str(source), str(self.upsampled), 15,
                   mode="all", seed=3)
        self.logger = logging.getLogger("replacement-guard-test")
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        self.temporary.cleanup()

    def test_cis_loop_caller_rejects_replacement_before_clustering(self):
        with self.assertRaisesRegex(ValueError, "replacement"):
            callCisLoops(
                str(self.upsampled),
                str(self.root / "cis-call"),
                self.logger,
                eps=[10],
                minPts=[2],
                cpu=1,
            )

    def test_fixed_trans_inference_rejects_replacement(self):
        candidate = self.root / "trans.tsv"
        candidate.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\tdistance\n"
            "t\tchr1\t0\t1000\tchr2\t0\t1000\t-1\n")
        with self.assertRaisesRegex(ValueError, "replacement"):
            callTransLoops(
                str(self.upsampled),
                str(self.root / "trans-call"),
                self.logger,
                eps=[10],
                minPts=[2],
                cpu=1,
                candidate_file=str(candidate),
                candidate_source="independent fixture",
                test_scope="global",
            )

    def test_split_validation_inference_rejects_replacement(self):
        with self.assertRaisesRegex(ValueError, "formal_inference|replacement"):
            callTransLoops(
                str(self.upsampled),
                str(self.root / "trans-split-call"),
                self.logger,
                eps=[10],
                minPts=[2],
                cpu=1,
                split_validation=True,
                split_seed=17,
                validation_fraction=0.5,
                test_scope="global",
            )

    def test_cis_quant_requires_offp_for_replacement(self):
        candidate = self.root / "cis.tsv"
        candidate.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\tdistance\n"
            "c\tchr1\t0\t50\tchr1\t80\t150\t80\n")
        with self.assertRaisesRegex(ValueError, "-offp"):
            quantLoops(
                str(self.upsampled),
                str(candidate),
                str(self.root / "quant-invalid"),
                self.logger,
                mode="cis",
                offp=False,
            )
        quantLoops(
            str(self.upsampled),
            str(candidate),
            str(self.root / "quant-descriptive"),
            self.logger,
            mode="cis",
            offp=True,
        )
        self.assertTrue((self.root / "quant-descriptive_loops.txt").is_file())

    def test_peak_quantification_rejects_replacement_pvalues(self):
        peaks = self.root / "peaks.bed"
        peaks.write_text("chr1\t0\t200\n")
        with self.assertRaisesRegex(ValueError, "replacement"):
            quantPeaks(
                str(self.upsampled), str(peaks),
                str(self.root / "peak-quant"), self.logger, cpu=1)


if __name__ == "__main__":
    unittest.main()
