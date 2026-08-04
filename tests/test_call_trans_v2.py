import json
import logging
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.callTransLoops import callTransLoops, _split_record_indices
from cLoops2.ds import TransContactIndex
from cLoops2.filter import samplePETs
from cLoops2.trans_stats import adjust_pvalues, read_trans_loop_results


class CallTransV2IntegrationTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.predir = self.root / "input"
        self.predir.mkdir()
        cis = np.asarray([[i, i + 100] for i in range(4)], dtype=np.int64)
        trans = np.asarray([
            [100, 1000],
            [105, 1005],
            [110, 1010],
            [115, 1015],
            [120, 1020],
            [1000, 5000],
            [2000, 6000],
            [3000, 7000],
            [4000, 8000],
            [5000, 9000],
        ], dtype=np.int64)
        cis_path = self.predir / "chr1-chr1.ixy"
        trans_path = self.predir / "chr1-chr2.ixy"
        joblib.dump(cis, str(cis_path))
        joblib.dump(trans, str(trans_path))
        meta = {
            "Unique PETs": 14,
            "Unique Cis PETs": 4,
            "Unique Trans PETs": 10,
            "Total PETs": 14,
            "Total Cis PETs": 4,
            "Total Trans PETs": 10,
            "Retention": {
                "retain trans": True,
                "retained categories": ["cis", "trans"],
            },
            "data": {
                "cis": {
                    "chr1-chr1": {
                        "ixy": str(cis_path),
                        "chromX": "chr1",
                        "chromY": "chr1",
                        "record_id": "cis:chr1-chr1",
                    },
                },
                "trans": {
                    "chr1-chr2": {
                        "ixy": str(trans_path),
                        "chromX": "chr1",
                        "chromY": "chr2",
                        "record_id": "trans:chr1-chr2",
                    },
                },
            },
        }
        (self.predir / "petMeta.json").write_text(json.dumps(meta))
        self.logger = logging.getLogger("call-trans-v2-test")
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        self.temporary.cleanup()

    def test_de_novo_is_explicitly_exploratory(self):
        prefix = str(self.root / "explore")
        results = callTransLoops(
            str(self.predir),
            prefix,
            self.logger,
            eps=[50],
            minPts=[3],
            cpu=1,
            local_pad=100,
            test_scope="global",
        )
        output = Path(prefix + "_trans_candidates.txt")
        self.assertTrue(output.is_file())
        self.assertFalse(Path(prefix + "_trans_loops.txt").exists())
        self.assertGreaterEqual(len(results), 1)
        loaded = read_trans_loop_results(str(output))
        self.assertEqual(len(loaded), len(results))
        for result in loaded:
            self.assertEqual(result.inference_mode, "exploratory_de_novo")
            self.assertFalse(result.inferential_validity)
            self.assertIsNone(result.bh_adjusted_p)
            self.assertIsNone(result.by_adjusted_p)
            self.assertIsNone(result.significant)
        cluster = min(loaded, key=lambda result: result.x_start)
        self.assertEqual((cluster.chrom_x, cluster.x_start, cluster.x_end),
                         ("chr1", 100, 120))
        self.assertEqual((cluster.chrom_y, cluster.y_start, cluster.y_end),
                         ("chr2", 1000, 1020))

    def test_independent_fixed_candidates_get_formal_family_correction(self):
        candidate_file = self.root / "fixed.tsv"
        candidate_file.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\tdistance\n"
            "cluster\tchr1\t90\t130\tchr2\t990\t1030\t-1\n"
            # Reversed axes exercise canonicalization; this rectangle has no
            # observed PETs but must remain in the correction family.
            "zero\tchr2\t20000\t20100\tchr1\t10000\t10100\t-1\n")
        prefix = str(self.root / "formal")
        results = callTransLoops(
            str(self.predir),
            prefix,
            self.logger,
            eps=[50],
            minPts=[3],
            cpu=1,
            candidate_file=str(candidate_file),
            candidate_source="independent synthetic preregistration",
            local_pad=100,
            test_scope="global",
            adjustment="BH",
            alpha=0.05,
        )
        output = Path(prefix + "_trans_loops.txt")
        self.assertTrue(output.is_file())
        self.assertEqual(len(results), 2)
        by_id = {result.loop_id: result for result in results}
        self.assertEqual(set(by_id), {"cluster", "zero"})
        self.assertEqual(by_id["zero"].pets, 0)
        self.assertEqual(by_id["zero"].chrom_x, "chr1")
        self.assertEqual(by_id["zero"].chrom_y, "chr2")
        for result in results:
            self.assertEqual(result.inference_mode, "formal_fixed")
            self.assertTrue(result.inferential_validity)
            self.assertIsNotNone(result.bh_adjusted_p)
            self.assertIsNotNone(result.by_adjusted_p)
            self.assertIsInstance(result.significant, bool)

    def test_manifest_zero_pair_stays_in_fixed_candidate_family(self):
        projection = self.root / "zero-projection"
        samplePETs(str(self.predir), str(projection), 0,
                   mode="trans", seed=19)
        candidate_file = self.root / "zero-fixed.tsv"
        candidate_file.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\tdistance\n"
            "known-zero\tchr1\t1\t10\tchr2\t20\t30\t-1\n")
        prefix = str(self.root / "zero-formal")
        results = callTransLoops(
            str(projection), prefix, self.logger,
            eps=[], minPts=[], candidate_file=str(candidate_file),
            candidate_source="independent zero-pair fixture",
            local_pad=100, test_scope="global",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].pair_pets, 0)
        self.assertEqual(results[0].pets, 0)
        self.assertEqual(results[0].primary_pvalue, 1.0)
        self.assertEqual(results[0].bh_adjusted_p, 1.0)

    def test_legacy_axis_mixed_significance_api_hard_fails(self):
        from cLoops2.callTransLoops import estLoopSig, markSigLoops
        with self.assertRaisesRegex(RuntimeError, "axis-mixed"):
            estLoopSig("chr1-chr2", [], "unused.ixy")
        with self.assertRaisesRegex(RuntimeError, "axis-mixed"):
            markSigLoops("chr1-chr2", [])

    def test_record_split_is_deterministic_disjoint_and_record_specific(self):
        discovery_a, validation_a = _split_record_indices(
            50, 123, "trans:chr1-chr2", 0.4)
        discovery_b, validation_b = _split_record_indices(
            50, 123, "trans:chr1-chr2", 0.4)
        discovery_other, validation_other = _split_record_indices(
            50, 123, "trans:chr1-chr3", 0.4)
        np.testing.assert_array_equal(discovery_a, discovery_b)
        np.testing.assert_array_equal(validation_a, validation_b)
        self.assertEqual(len(np.intersect1d(discovery_a, validation_a)), 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate((discovery_a, validation_a))),
            np.arange(50),
        )
        self.assertFalse(np.array_equal(validation_a, validation_other))
        self.assertFalse(np.array_equal(discovery_a, discovery_other))

    def test_split_validation_requires_seed_and_excludes_fixed_file(self):
        with self.assertRaisesRegex(ValueError, "split_seed is required"):
            callTransLoops(
                str(self.predir), str(self.root / "missing-seed"), self.logger,
                eps=[50], minPts=[3], split_validation=True,
                test_scope="global")

        candidate_file = self.root / "split-conflict.tsv"
        candidate_file.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\n"
            "candidate\tchr1\t90\t130\tchr2\t990\t1030\n")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            callTransLoops(
                str(self.predir), str(self.root / "split-conflict"), self.logger,
                eps=[50], minPts=[3], split_validation=True, split_seed=1,
                candidate_file=str(candidate_file),
                candidate_source="independent fixture",
                test_scope="global")

    def test_split_validation_is_formal_reproducible_and_validation_only(self):
        cluster = np.asarray(
            [[100 + i * 3, 1000 + i * 3] for i in range(30)],
            dtype=np.int64)
        background = np.asarray(
            [[5000 + i * 1000, 20000 + i * 1300] for i in range(10)],
            dtype=np.int64)
        trans = np.vstack((cluster, background))
        trans_path = self.predir / "chr1-chr2.ixy"
        joblib.dump(trans, str(trans_path))
        meta_path = self.predir / "petMeta.json"
        meta = json.loads(meta_path.read_text())
        meta["Unique PETs"] = 44
        meta["Unique Trans PETs"] = 40
        meta["Total PETs"] = 44
        meta["Total Trans PETs"] = 40
        meta_path.write_text(json.dumps(meta))

        first_prefix = str(self.root / "split-first")
        second_prefix = str(self.root / "split-second")
        kwargs = dict(
            eps=[25],
            minPts=[4],
            split_validation=True,
            split_seed=37,
            validation_fraction=0.5,
            local_pad=100,
            test_scope="global",
            adjustment="BH",
            alpha=0.05,
        )
        first = callTransLoops(
            str(self.predir), first_prefix, self.logger, cpu=1, **kwargs)
        second = callTransLoops(
            str(self.predir), second_prefix, self.logger, cpu=2, **kwargs)

        self.assertGreaterEqual(len(first), 1)
        self.assertEqual(first, second)
        self.assertTrue(Path(first_prefix + "_trans_loops.txt").is_file())
        self.assertFalse(Path(first_prefix + "_trans_candidates.txt").exists())
        provenance_path = Path(first_prefix + "_trans_split.json")
        self.assertTrue(provenance_path.is_file())
        provenance = json.loads(provenance_path.read_text())
        second_provenance = json.loads(
            Path(second_prefix + "_trans_split.json").read_text())
        self.assertEqual(provenance, second_provenance)
        self.assertEqual(provenance["split_seed"], 37)
        self.assertEqual(provenance["validation_fraction"], 0.5)
        self.assertEqual(provenance["candidate_count"], len(first))
        split = provenance["record_splits"][0]
        self.assertEqual(split["source_rows"], 40)
        self.assertEqual(split["discovery_rows"], 20)
        self.assertEqual(split["validation_rows"], 20)
        self.assertNotEqual(split["discovery_index_sha256"],
                            split["validation_index_sha256"])

        _, validation_ids = _split_record_indices(
            40, 37, "trans:chr1-chr2", 0.5)
        validation_index = TransContactIndex.from_matrix(trans[validation_ids])
        expected_bh = adjust_pvalues(
            [result.primary_pvalue for result in first], "BH")
        for result, expected_q in zip(first, expected_bh):
            self.assertEqual(result.inference_mode,
                             "formal_split_validation")
            self.assertTrue(result.selection_adjusted)
            self.assertTrue(result.inferential_validity)
            self.assertEqual(result.pair_pets, 20)
            self.assertEqual(
                result.pets,
                validation_index.count_rect(
                    result.x_start, result.x_end,
                    result.y_start, result.y_end),
            )
            self.assertAlmostEqual(result.bh_adjusted_p, expected_q)
            self.assertIn("seed=37", result.candidate_source)


if __name__ == "__main__":
    unittest.main()
