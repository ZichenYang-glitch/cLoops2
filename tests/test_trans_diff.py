import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.filter import samplePETs
from cLoops2.trans_diff import call_trans_diff


class TransDifferentialTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidates = self.root / "candidates.tsv"
        self.candidates.write_text(
            "loopId\tchromX\tstartX\tendX\tchromY\tstartY\tendY\n"
            "signal\tchr1\t90\t150\tchr2\t990\t1050\n"
            "zero\tchr1\t5000\t5100\tchr2\t6000\t6100\n")

    def tearDown(self):
        self.temporary.cleanup()

    def _sample(self, name, loop_count, total=30):
        directory = self.root / name
        directory.mkdir()
        signal = np.asarray(
            [[100 + i, 1000 + i] for i in range(loop_count)],
            dtype=np.int64)
        background = np.asarray(
            [[10000 + i, 20000 + i]
             for i in range(total - loop_count)], dtype=np.int64)
        matrix = np.concatenate([signal, background], axis=0)
        path = directory / "chr1-chr2.ixy"
        joblib.dump(matrix, str(path))
        meta = {
            "Unique PETs": total,
            "Unique Cis PETs": 0,
            "Unique Trans PETs": total,
            "Total PETs": total,
            "Total Cis PETs": 0,
            "Total Trans PETs": total,
            "Retention": {
                "retain trans": True,
                "retained categories": ["cis", "trans"],
                "chromosome whitelist": ["chr1", "chr2"],
                "cut": 0,
                "mcut": -1,
                "mapq": 1,
                "format": "bedpe",
            },
            "data": {
                "cis": {},
                "trans": {"chr1-chr2": {
                    "ixy": str(path), "chromX": "chr1", "chromY": "chr2",
                    "record_id": "trans:chr1-chr2"}},
            },
        }
        (directory / "petMeta.json").write_text(json.dumps(meta))
        return directory

    def _sheet(self, name, rows):
        path = self.root / name
        with path.open("w") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample", "condition", "directory"])
            writer.writerows(rows)
        return path

    def test_single_sample_exact_is_explicitly_not_biological(self):
        ref, con = self._sample("ref", 2), self._sample("con", 12)
        sheet = self._sheet("exact.tsv", [
            ["r1", "control", ref], ["c1", "treated", con]])
        results, paths = call_trans_diff(
            str(sheet), str(self.candidates), str(self.root / "exact"),
            "external fixed fixture", method="auto", reference="control",
            contrast="treated")
        signal = {result.loop_id: result for result in results}["signal"]
        self.assertEqual(signal.method, "exact_rate")
        self.assertFalse(signal.biological_inference)
        self.assertIsNone(signal.significant)
        self.assertGreater(signal.log2_fold_change, 1)
        self.assertIsNotNone(signal.bh_adjusted_p)
        self.assertTrue(Path(paths["counts"]).is_file())
        self.assertTrue(Path(paths["pair_counts"]).is_file())
        self.assertTrue(Path(paths["reads_a"]).is_file())
        self.assertTrue(Path(paths["reads_b"]).is_file())
        self.assertTrue(Path(paths["edger_script"]).is_file())

    def test_replicated_nb_uses_raw_sample_counts_and_offsets(self):
        rows = []
        for index in range(3):
            rows.append(["r%s" % index, "control",
                         self._sample("r%s" % index, 1)])
        for index in range(3):
            rows.append(["c%s" % index, "treated",
                         self._sample("c%s" % index, 20)])
        sheet = self._sheet("nb.tsv", rows)
        results, paths = call_trans_diff(
            str(sheet), str(self.candidates), str(self.root / "nb"),
            "independent discovery cohort", method="auto",
            reference="control", contrast="treated", prior_df=10)
        signal = {result.loop_id: result for result in results}["signal"]
        self.assertEqual(signal.method, "nb_glm_wald")
        self.assertEqual(signal.model_status, "nb_glm")
        self.assertTrue(signal.biological_inference)
        self.assertEqual(signal.reference_total, 3)
        self.assertEqual(signal.contrast_total, 60)
        self.assertGreater(signal.log2_fold_change, 3)
        self.assertLess(signal.pvalue, 0.05)
        self.assertIn("y$offset", Path(paths["edger_script"]).read_text())
        self.assertIn(
            'levels=c("control", "treated")',
            Path(paths["edger_script"]).read_text())

    def test_pair_and_marginal_exposures_are_exported_explicitly(self):
        ref, con = self._sample("pair-ref", 2), self._sample("pair-con", 4)
        sheet = self._sheet("pair.tsv", [
            ["r", "A", ref], ["c", "B", con]])
        _, pair_paths = call_trans_diff(
            str(sheet), str(self.candidates), str(self.root / "pair"),
            "external", method="export", offset_scope="pair")
        lines = Path(pair_paths["exposures"]).read_text().splitlines()
        self.assertEqual(lines[1].split("\t")[-2:], ["30.0", "30.0"])
        _, marginal_paths = call_trans_diff(
            str(sheet), str(self.candidates), str(self.root / "marginal"),
            "external", method="export", offset_scope="marginal")
        marginal = Path(marginal_paths["exposures"]).read_text().splitlines()
        self.assertNotEqual(marginal[1].split("\t")[-2:], ["30.0", "30.0"])

    def test_replacement_sample_is_rejected(self):
        source = self._sample("source", 3, total=10)
        replaced = self.root / "replaced"
        samplePETs(str(source), str(replaced), 20,
                   mode="all", seed=3)
        control = self._sample("plain", 3, total=10)
        sheet = self._sheet("replacement.tsv", [
            ["a", "A", control], ["b", "B", replaced]])
        with self.assertRaisesRegex(ValueError, "replacement"):
            call_trans_diff(
                str(sheet), str(self.candidates), str(self.root / "bad"),
                "external", method="exact")

    def test_cli_trans_mode_runs_without_legacy_cis_arguments(self):
        ref, con = self._sample("cli-ref", 2), self._sample("cli-con", 12)
        sheet = self._sheet("cli.tsv", [
            ["r1", "control", ref], ["c1", "treated", con]])
        prefix = self.root / "cli-result"
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["MPLCONFIGDIR"] = str(self.root / "mpl")
        result = subprocess.run(
            [sys.executable, "-m", "cLoops2.cLoops2", "callDiffLoops",
             "-mode", "trans", "-samples", str(sheet),
             "-trans_candidates", str(self.candidates),
             "-trans_candidate_source", "independent CLI fixture",
             "-reference", "control", "-contrast", "treated",
             "-o", str(prefix)],
            cwd=str(self.root), env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(str(prefix) + "_trans_dloops.tsv").is_file())
        self.assertTrue(Path(str(prefix) + "_trans_counts.tsv").is_file())

    def test_existing_output_is_never_silently_overwritten(self):
        ref, con = self._sample("overwrite-ref", 2), self._sample(
            "overwrite-con", 3)
        sheet = self._sheet("overwrite.tsv", [
            ["r", "A", ref], ["c", "B", con]])
        prefix = self.root / "existing"
        sentinel = Path(str(prefix) + "_trans_counts.tsv")
        sentinel.write_text("do-not-overwrite\n")
        with self.assertRaisesRegex(FileExistsError, "output exists"):
            call_trans_diff(
                str(sheet), str(self.candidates), str(prefix),
                "external", method="exact")
        self.assertEqual(sentinel.read_text(), "do-not-overwrite\n")


if __name__ == "__main__":
    unittest.main()
