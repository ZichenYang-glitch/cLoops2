import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]


class CliModeTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = os.environ.copy()
        self.environment["PYTHONPATH"] = str(REPOSITORY)
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self.environment["MPLCONFIGDIR"] = str(self.root / "matplotlib")

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "cLoops2.cLoops2"] + list(arguments),
            cwd=str(self.root),
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def _input(self):
        directory = self.root / "input"
        directory.mkdir()
        cis = np.asarray([[i, i + 100] for i in range(4)], dtype=np.int64)
        trans = np.asarray([[i + 200, i + 300] for i in range(6)],
                           dtype=np.int64)
        cis_path = directory / "chr1-chr1.ixy"
        trans_path = directory / "chr1-chr2.ixy"
        joblib.dump(cis, str(cis_path))
        joblib.dump(trans, str(trans_path))
        (directory / "petMeta.json").write_text(json.dumps({
            "Unique PETs": 10,
            "Total PETs": 10,
            "Total Cis PETs": 4,
            "Total Trans PETs": 6,
            "data": {
                "cis": {"chr1-chr1": {"ixy": str(cis_path)}},
                "trans": {"chr1-chr2": {"ixy": str(trans_path)}},
            },
        }))
        return directory

    def test_sample_help_exposes_mode_and_seed(self):
        result = self._run("samplePETs", "-h")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-mode {cis,trans,all}", result.stdout)
        self.assertIn("-seed", result.stdout)
        self.assertIn("automatically performs sampling", result.stdout)

    def test_callloops_help_exposes_mode_and_legacy_alias(self):
        result = self._run("callLoops", "-h")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-mode {cis,trans,all}", result.stdout)
        self.assertIn("-trans", result.stdout)
        self.assertIn("alias for -mode all", result.stdout)
        self.assertIn("-trans_candidates", result.stdout)

    def test_downstream_help_exposes_trans_quant_plot_and_dump(self):
        quant = self._run("quant", "-h")
        self.assertEqual(quant.returncode, 0, quant.stderr)
        self.assertIn("-mode {cis,trans,all}", quant.stdout)
        plot = self._run("plot", "-h")
        self.assertEqual(plot.returncode, 0, plot.stderr)
        self.assertIn("-trans_method", plot.stdout)
        self.assertIn("-y_start", plot.stdout)
        dump = self._run("dump", "-h")
        self.assertEqual(dump.returncode, 0, dump.stderr)
        self.assertIn("-trans_mat", dump.stdout)
        self.assertIn("-trans_mat_method", dump.stdout)
        filtered = self._run("filterPETs", "-h")
        self.assertEqual(filtered.returncode, 0, filtered.stderr)
        self.assertIn("-mode {cis,trans,all}", filtered.stdout)
        annotated = self._run("anaLoops", "-h")
        self.assertEqual(annotated.returncode, 0, annotated.stderr)
        self.assertIn("-mode {cis,trans,all}", annotated.stdout)

    def test_diff_and_split_help_exposes_statistical_modes(self):
        loops = self._run("callLoops", "-h")
        self.assertEqual(loops.returncode, 0, loops.stderr)
        self.assertIn("-trans_split", loops.stdout)
        self.assertIn("-trans_split_seed", loops.stdout)
        diff = self._run("callDiffLoops", "-h")
        self.assertEqual(diff.returncode, 0, diff.stderr)
        self.assertIn("-mode {cis,trans}", diff.stdout)
        self.assertIn("-samples", diff.stdout)
        self.assertIn("-trans_offset {global,pair,marginal}", diff.stdout)
        for command in ("transAgg", "transViewpoint", "transMontage"):
            result = self._run(command, "-h")
            self.assertEqual(result.returncode, 0, result.stderr)
        aggregate = self._run("transAgg", "-h")
        self.assertIn("-method {obs,pair_oe,window_oe}", aggregate.stdout)

    def test_sample_cli_writes_real_trans_projection(self):
        source = self._input()
        output = self.root / "trans-output"
        result = self._run(
            "samplePETs",
            "-d", str(source),
            "-o", str(output),
            "-tot", "8",
            "-mode", "trans",
            "-seed", "123",
            "-p", "1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads((output / "petMeta.json").read_text())
        self.assertTrue(meta["data"]["trans"])
        self.assertFalse(meta["data"]["cis"])
        self.assertEqual(meta["Sampling"]["target_logical_total"], 8)
        self.assertEqual(meta["Sampling"]["emit"], "trans")
        self.assertEqual(meta["Unique PETs"],
                         meta["Sampling"]["written_by_category"]["trans"])

    def test_negative_target_is_an_argparse_error(self):
        result = self._run("samplePETs", "-tot", "-1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-negative", result.stderr)

    def test_callloops_trans_mode_writes_exploratory_candidates_only(self):
        source = self._input()
        prefix = self.root / "called"
        result = self._run(
            "callLoops",
            "-d", str(source),
            "-o", str(prefix),
            "-eps", "50",
            "-minPts", "3",
            "-mode", "trans",
            "-p", "1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(str(prefix) + "_trans_candidates.txt").is_file())
        self.assertFalse(Path(str(prefix) + "_trans_loops.txt").exists())

    def test_fixed_trans_only_does_not_require_dbscan_parameters(self):
        source = self._input()
        candidates = self.root / "fixed.tsv"
        candidates.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\tdistance\n"
            "fixed\tchr1\t0\t1000\tchr2\t0\t1000\t-1\n")
        prefix = self.root / "formal"
        result = self._run(
            "callLoops", "-d", str(source), "-o", str(prefix),
            "-mode", "trans", "-trans_candidates", str(candidates),
            "-trans_candidate_source", "independent CLI fixture",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(str(prefix) + "_trans_loops.txt").is_file())

    def test_trans_candidate_options_are_rejected_in_cis_mode(self):
        source = self._input()
        candidates = self.root / "fixed-cis-mode.tsv"
        candidates.write_text(
            "loopId\tchrA\tstartA\tendA\tchrB\tstartB\tendB\tdistance\n"
            "fixed\tchr1\t0\t10\tchr2\t0\t10\t-1\n")
        result = self._run(
            "callLoops", "-d", str(source), "-o", str(self.root / "bad"),
            "-mode", "cis", "-trans_candidates", str(candidates),
            "-trans_candidate_source", "independent fixture",
            "-eps", "10", "-minPts", "2",
        )
        self.assertEqual(result.returncode, 2)

    def test_trans_split_requires_a_seed(self):
        source = self._input()
        result = self._run(
            "callLoops", "-d", str(source), "-o", str(self.root / "bad-split"),
            "-mode", "trans", "-eps", "50", "-minPts", "3",
            "-trans_split",
        )
        self.assertEqual(result.returncode, 2)

    def test_directional_trans_summary_commands_write_outputs(self):
        source = self._input()
        loops = self.root / "trans-features.tsv"
        loops.write_text(
            "loopId\tchromX\tstartX\tendX\tchromY\tstartY\tendY\tdistance\n"
            "feature\tchr1\t200\t205\tchr2\t300\t305\t-1\n")
        aggregate_prefix = self.root / "aggregate"
        aggregate = self._run(
            "transAgg", "-d", str(source), "-loops", str(loops),
            "-x_bs", "10", "-x_flank_bins", "0", "-y_flank_bins", "0",
            "-o", str(aggregate_prefix))
        self.assertEqual(aggregate.returncode, 0, aggregate.stderr)
        self.assertTrue(Path(str(aggregate_prefix) + "_trans_agg.json").is_file())

        viewpoint_prefix = self.root / "viewpoint"
        viewpoint = self._run(
            "transViewpoint", "-d", str(source), "-chromX", "chr1",
            "-chromY", "chr2", "-anchor_axis", "x",
            "-anchor_start", "200", "-anchor_end", "205",
            "-target_start", "300", "-target_end", "310", "-bs", "5",
            "-o", str(viewpoint_prefix))
        self.assertEqual(viewpoint.returncode, 0, viewpoint.stderr)
        self.assertTrue(Path(str(viewpoint_prefix) +
                             "_trans_viewpoint.tsv").is_file())

        x_regions, y_regions = self.root / "x.bed", self.root / "y.bed"
        x_regions.write_text("chr1\t195\t215\tx\n")
        y_regions.write_text("chr2\t295\t315\ty\n")
        montage_prefix = self.root / "montage"
        montage = self._run(
            "transMontage", "-d", str(source), "-chromX", "chr1",
            "-chromY", "chr2", "-x_regions", str(x_regions),
            "-y_regions", str(y_regions), "-o", str(montage_prefix))
        self.assertEqual(montage.returncode, 0, montage.stderr)
        self.assertTrue(Path(str(montage_prefix) +
                             "_trans_montage.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
