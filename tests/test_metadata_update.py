import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.filter import samplePETs
from cLoops2.io import writeNewJson
from cLoops2.metadata import build_library_context
from cLoops2.metadata import CAP_FORMAL_INFERENCE


class MetadataUpdateTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        cis = np.asarray([[i, i + 100] for i in range(5)], dtype=np.int64)
        trans = np.asarray([[i + 200, i + 300] for i in range(5)],
                           dtype=np.int64)
        cis_path = self.source / "chr1-chr1.ixy"
        trans_path = self.source / "chr1-chr2.ixy"
        joblib.dump(cis, str(cis_path))
        joblib.dump(trans, str(trans_path))
        meta = {
            "Unique PETs": 10,
            "Total PETs": 10,
            "Total Cis PETs": 5,
            "Total Trans PETs": 5,
            "Retention": {
                "retain trans": True,
                "retained categories": ["cis", "trans"],
            },
            "data": {
                "cis": {
                    "chr1-chr1": {"ixy": str(cis_path.resolve())},
                },
                "trans": {
                    "chr1-chr2": {"ixy": str(trans_path.resolve())},
                },
            },
        }
        (self.source / "petMeta.json").write_text(json.dumps(meta))

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _meta(directory):
        return json.loads((directory / "petMeta.json").read_text())

    @staticmethod
    def _actual_counts(directory):
        meta = MetadataUpdateTest._meta(directory)
        counts = {}
        for category in ("cis", "trans"):
            for key, entry in meta["data"][category].items():
                counts["%s:%s" % (category, key)] = int(
                    joblib.load(entry["ixy"], mmap_mode="r").shape[0])
        return counts

    def test_update_preserves_valid_sampling_after_directory_move(self):
        output = self.root / "sampled"
        samplePETs(str(self.source), str(output), 8,
                   mode="all", seed=11)
        moved = self.root / "moved"
        shutil.move(str(output), str(moved))
        writeNewJson(str(moved))
        meta = self._meta(moved)
        self.assertIn("Sampling", meta)
        self.assertIn("Retention", meta)
        for category in ("cis", "trans"):
            for entry in meta["data"][category].values():
                self.assertEqual(Path(entry["ixy"]).parent, moved)
                self.assertIn("chromX", entry)
                self.assertIn("chromY", entry)
        context = build_library_context(meta,
                                        actual_counts=self._actual_counts(moved))
        self.assertEqual(context.validity, "valid", context.reasons)
        self.assertEqual(context.logical_total, 8)

    def test_update_keeps_audit_block_but_invalidates_changed_counts(self):
        output = self.root / "sampled-tampered"
        samplePETs(str(self.source), str(output), 8,
                   mode="all", seed=12)
        path = sorted(output.glob("*.ixy"))[0]
        matrix = joblib.load(str(path))
        joblib.dump(matrix[:-1], str(path))
        writeNewJson(str(output))
        meta = self._meta(output)
        self.assertIn("Sampling", meta)
        context = build_library_context(
            meta, actual_counts=self._actual_counts(output))
        self.assertEqual(context.validity, "invalid")
        self.assertTrue(any("physical" in reason or "written" in reason
                            for reason in context.reasons))

    def test_update_cannot_recertify_a_duplicated_pre_root(self):
        cis_path = self.source / "chr1-chr1.ixy"
        matrix = joblib.load(str(cis_path))
        joblib.dump(np.concatenate([matrix, matrix], axis=0), str(cis_path))
        writeNewJson(str(self.source))
        meta = self._meta(self.source)
        self.assertEqual(meta["Transformation"]["operation"],
                         "external_update")
        self.assertIsNone(meta["Transformation"]["replacement_ever"])
        context = build_library_context(
            meta, actual_counts=self._actual_counts(self.source))
        self.assertEqual(context.validity, "unknown")
        self.assertIsNone(context.logical_total)
        self.assertNotIn(CAP_FORMAL_INFERENCE, context.capabilities)

    def test_same_aggregate_per_pair_tamper_invalidates_sampling(self):
        source = self.root / "two-cis-source"
        source.mkdir()
        data = {"cis": {}, "trans": {}}
        for chrom, offset in (("chr1", 0), ("chr2", 1000)):
            key = "%s-%s" % (chrom, chrom)
            path = source / (key + ".ixy")
            joblib.dump(np.asarray(
                [[offset + i, offset + 100 + i] for i in range(3)],
                dtype=np.int64), str(path))
            data["cis"][key] = {"ixy": str(path)}
        (source / "petMeta.json").write_text(json.dumps({
            "Unique PETs": 6,
            "Unique Cis PETs": 6,
            "Unique Trans PETs": 0,
            "Total PETs": 6,
            "Total Cis PETs": 6,
            "Total Trans PETs": 0,
            "Retention": {
                "retain trans": False,
                "retained categories": ["cis"],
            },
            "data": data,
        }))
        sampled = self.root / "two-cis-sampled"
        samplePETs(str(source), str(sampled), 6,
                   mode="all", seed=51)
        first = sampled / "chr1-chr1.ixy"
        second = sampled / "chr2-chr2.ixy"
        first_matrix = joblib.load(str(first))
        second_matrix = joblib.load(str(second))
        moved = first_matrix[-1:, :]
        joblib.dump(first_matrix[:-1, :], str(first))
        joblib.dump(np.concatenate([second_matrix, moved], axis=0), str(second))

        writeNewJson(str(sampled))
        meta = self._meta(sampled)
        self.assertEqual(meta["Unique PETs"], 6)
        self.assertEqual(meta["Unique Cis PETs"], 6)
        self.assertEqual(meta["Sampling"]["validity"], "invalid")
        context_without_external_counts = build_library_context(meta)
        self.assertEqual(context_without_external_counts.validity, "invalid")
        self.assertNotIn(CAP_FORMAL_INFERENCE,
                         context_without_external_counts.capabilities)


if __name__ == "__main__":
    unittest.main()
