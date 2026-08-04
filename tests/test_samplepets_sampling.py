#!/usr/bin/env python
"""Integration tests for the v0.2 whole-library sampling contract."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.filter import samplePETs
from cLoops2.metadata import build_library_context


class SamplePETsTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cloops2-sampling-")
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.arrays = self._write_fixture(self.input_dir)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_fixture(directory, unique_override=None, compress=0):
        directory.mkdir(parents=True)
        arrays = {
            "chr1-chr1": np.array(
                [[100 + i, 1000 + i] for i in range(6)], dtype=np.int64),
            "chr2-chr2": np.array(
                [[2100 + i, 3100 + i] for i in range(4)], dtype=np.int64),
            "chr1-chr2": np.array(
                [[4100 + i, 5100 + i] for i in range(7)], dtype=np.int64),
            "chr1-chr3": np.array(
                [[6100 + i, 7100 + i] for i in range(3)], dtype=np.int64),
        }
        data = {"cis": {}, "trans": {}}
        for key, mat in arrays.items():
            path = directory / (key + ".ixy")
            joblib.dump(mat, str(path), compress=compress)
            category = "cis" if key.split("-")[0] == key.split("-")[1] else "trans"
            data[category][key] = {"ixy": str(path.resolve())}
        total = sum(mat.shape[0] for mat in arrays.values())
        meta = {
            "Unique PETs": total if unique_override is None else unique_override,
            "Total PETs": total,
            "Total Cis PETs": 10,
            "Total Trans PETs": 10,
            "Retention": {
                "retain trans": True,
                "retained categories": ["cis", "trans"],
                "chromosome whitelist": [],
                "cut": 0,
                "mcut": -1,
                "mapq": 1,
            },
            "data": data,
        }
        (directory / "petMeta.json").write_text(json.dumps(meta))
        return arrays

    @staticmethod
    def _meta(directory):
        return json.loads((directory / "petMeta.json").read_text())

    @staticmethod
    def _arrays(directory):
        return {
            path.stem: joblib.load(str(path))
            for path in sorted(directory.glob("*.ixy"))
        }

    @staticmethod
    def _rows(directory):
        return sum(mat.shape[0]
                   for mat in SamplePETsTests._arrays(directory).values())

    @staticmethod
    def _flat_rows(directory):
        arrays = list(SamplePETsTests._arrays(directory).values())
        if not arrays:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(arrays, axis=0)

    @staticmethod
    def _actual_counts(directory):
        meta = SamplePETsTests._meta(directory)
        counts = {}
        for category in ("cis", "trans"):
            for key, entry in meta["data"][category].items():
                counts["%s:%s" % (category, key)] = int(
                    joblib.load(entry["ixy"], mmap_mode="r").shape[0])
        return counts

    def test_downsample_all_is_exact_and_without_replacement(self):
        output = self.root / "down-all"
        sampling = samplePETs(
            str(self.input_dir), str(output), 11,
            cpu=1, mode="all", seed=123,
        )
        meta = self._meta(output)
        rows = self._flat_rows(output)
        self.assertEqual(meta["Unique PETs"], 11)
        self.assertEqual(rows.shape[0], 11)
        self.assertEqual(np.unique(rows, axis=0).shape[0], 11)
        self.assertEqual(sampling["direction"], "downsample")
        self.assertFalse(sampling["replacement_this_step"])
        self.assertEqual(sum(sampling["selected_by_category"].values()), 11)
        self.assertEqual(sum(r["selected_rows"] for r in sampling["records"]),
                         11)
        self.assertEqual(meta["Retention"],
                         self._meta(self.input_dir)["Retention"])
        self.assertNotIn("Total Cis PETs", meta)

    def test_modes_are_projections_of_the_same_seeded_global_sample(self):
        outputs = {}
        samplings = {}
        for mode in ("all", "cis", "trans"):
            outputs[mode] = self.root / ("mode-" + mode)
            samplings[mode] = samplePETs(
                str(self.input_dir), str(outputs[mode]), 14,
                cpu=1, mode=mode, seed=2024,
            )

        all_arrays = self._arrays(outputs["all"])
        cis_arrays = self._arrays(outputs["cis"])
        trans_arrays = self._arrays(outputs["trans"])
        self.assertEqual(set(cis_arrays), {"chr1-chr1", "chr2-chr2"})
        self.assertEqual(set(trans_arrays), {"chr1-chr2", "chr1-chr3"})
        for key, mat in cis_arrays.items():
            np.testing.assert_array_equal(mat, all_arrays[key])
        for key, mat in trans_arrays.items():
            np.testing.assert_array_equal(mat, all_arrays[key])

        selected = samplings["all"]["selected_by_category"]
        self.assertEqual(samplings["cis"]["selected_by_category"], selected)
        self.assertEqual(samplings["trans"]["selected_by_category"], selected)
        self.assertEqual(self._meta(outputs["all"])["Unique PETs"], 14)
        self.assertEqual(self._meta(outputs["cis"])["Unique PETs"],
                         selected["cis"])
        self.assertEqual(self._meta(outputs["trans"])["Unique PETs"],
                         selected["trans"])
        self.assertEqual(samplings["trans"]["target_logical_total"], 14)
        self.assertEqual(samplings["trans"]["physical_output_total"],
                         selected["trans"])
        self.assertEqual(samplings["trans"]["source_kind"], "projection")
        for mode, output in outputs.items():
            context = build_library_context(
                self._meta(output), actual_counts=self._actual_counts(output))
            self.assertEqual(context.validity, "valid",
                             "%s: %s" % (mode, context.reasons))
            self.assertEqual(context.logical_total, 14)
            self.assertEqual(context.physical_total,
                             self._meta(output)["Unique PETs"])

    def test_same_seed_is_cpu_independent(self):
        one = self.root / "cpu-one"
        two = self.root / "cpu-two"
        samplePETs(str(self.input_dir), str(one), 13,
                   cpu=1, mode="all", seed=91)
        samplePETs(str(self.input_dir), str(two), 13,
                   cpu=2, mode="all", seed=91)
        one_arrays, two_arrays = self._arrays(one), self._arrays(two)
        self.assertEqual(set(one_arrays), set(two_arrays))
        for key in one_arrays:
            np.testing.assert_array_equal(one_arrays[key], two_arrays[key])

    def test_identity_preserves_every_source_array(self):
        output = self.root / "identity"
        sampling = samplePETs(
            str(self.input_dir), str(output), 20,
            cpu=-1, mode="all", seed=17,
        )
        self.assertEqual(sampling["direction"], "identity")
        self.assertFalse(sampling["replacement_this_step"])
        output_arrays = self._arrays(output)
        for key, source in self.arrays.items():
            np.testing.assert_array_equal(output_arrays[key], source)

    def test_root_upsampling_is_automatic_and_marked(self):
        outputs, samplings = {}, {}
        for mode in ("all", "cis", "trans"):
            outputs[mode] = self.root / ("upsample-" + mode)
            samplings[mode] = samplePETs(
                str(self.input_dir), str(outputs[mode]), 27,
                cpu=2, mode=mode, seed=8,
            )
        sampling = samplings["all"]
        rows = self._flat_rows(outputs["all"])
        self.assertEqual(rows.shape[0], 27)
        self.assertLessEqual(np.unique(rows, axis=0).shape[0], 20)
        self.assertEqual(sampling["direction"], "upsample")
        self.assertTrue(sampling["replacement_this_step"])
        self.assertTrue(sampling["replacement_ever"])
        self.assertEqual(sampling["target PETs"], 27)
        self.assertEqual(sampling["available PETs"], 20)
        all_arrays = self._arrays(outputs["all"])
        for mode, output in outputs.items():
            self.assertEqual(samplings[mode]["selected_by_category"],
                             sampling["selected_by_category"])
            context = build_library_context(
                self._meta(output), actual_counts=self._actual_counts(output))
            self.assertEqual(context.validity, "valid",
                             "%s: %s" % (mode, context.reasons))
            self.assertEqual(context.logical_total, 27)
            self.assertTrue(context.replacement_ever)
            if mode != "all":
                for key, mat in self._arrays(output).items():
                    np.testing.assert_array_equal(mat, all_arrays[key])

    def test_stale_unique_is_rejected_before_creating_output(self):
        stale = self.root / "stale"
        self._write_fixture(stale, unique_override=999)
        output = self.root / "stale-output"
        with self.assertRaisesRegex(ValueError, "structurally invalid"):
            samplePETs(str(stale), str(output), 10,
                       mode="all", seed=6)
        self.assertFalse(output.exists())

    def test_bare_transform_is_not_laundered_into_formal_provenance(self):
        transformed = self.root / "legacy-transform"
        transformed.mkdir()
        mat = np.array([[1, 11], [2, 12], [3, 13]], dtype=np.int64)
        path = transformed / "chr1-chr1.ixy"
        joblib.dump(mat, str(path))
        (transformed / "petMeta.json").write_text(json.dumps({
            "Unique PETs": 3,
            "data": {
                "cis": {"chr1-chr1": {"ixy": str(path.resolve())}},
                "trans": {},
            },
        }))

        output = self.root / "legacy-transform-sampled"
        sampling = samplePETs(
            str(transformed), str(output), 2, mode="cis", seed=17)
        self.assertIsNone(sampling["replacement_ever"])
        self.assertEqual(sampling["source_provenance_validity"], "unknown")
        context = build_library_context(
            self._meta(output), actual_counts=self._actual_counts(output))
        self.assertEqual(context.logical_total, 2)
        self.assertNotIn("formal_inference", context.capabilities)

    def test_projection_nested_downsample_uses_parent_logical_depth(self):
        projection = self.root / "projection"
        parent = samplePETs(
            str(self.input_dir), str(projection), 15,
            mode="trans", seed=33,
        )
        physical_parent = parent["physical_output_total"]
        self.assertLess(physical_parent, 12)

        nested = self.root / "nested"
        child = samplePETs(
            str(projection), str(nested), 12,
            mode="trans", seed=44,
        )
        self.assertEqual(child["source_logical_total"], 15)
        self.assertEqual(child["target_logical_total"], 12)
        self.assertEqual(child["direction"], "downsample")
        self.assertFalse(child["replacement_this_step"])
        self.assertLessEqual(child["physical_output_total"], physical_parent)
        self.assertEqual(sum(r["source_rows"] for r in child["records"]), 15)
        self.assertEqual(sum(r["selected_rows"] for r in child["records"]), 12)
        self.assertEqual(sum(r["written_rows"] for r in child["records"]),
                         child["physical_output_total"])
        omitted = [r for r in child["records"] if r["category"] == "cis"]
        self.assertTrue(omitted)
        self.assertTrue(all(not r["materialized"] for r in omitted))
        self.assertTrue(all(r["written_rows"] == 0 for r in omitted))

        meta = self._meta(nested)
        actual = {
            "trans:" + key: mat.shape[0]
            for key, mat in self._arrays(nested).items()
        }
        context = build_library_context(meta, actual_counts=actual)
        self.assertEqual(context.validity, "valid", context.reasons)
        self.assertEqual(context.logical_total, 12)
        self.assertEqual(context.physical_total,
                         child["physical_output_total"])

        with self.assertRaisesRegex(ValueError, "projection upsampling"):
            samplePETs(str(projection), str(self.root / "nested-up"), 16,
                       mode="trans", seed=44)
        with self.assertRaisesRegex(ValueError, "can only be sampled again"):
            samplePETs(str(projection), str(self.root / "nested-all"), 12,
                       mode="all", seed=44)

    def test_compressed_ixy_falls_back_from_mmap(self):
        compressed = self.root / "compressed"
        self._write_fixture(compressed, compress=3)
        output = self.root / "compressed-output"
        sampling = samplePETs(
            str(compressed), str(output), 9,
            mode="all", seed=123,
        )
        self.assertEqual(self._rows(output), 9)
        self.assertEqual(sampling["direction"], "downsample")

    def test_nonempty_output_and_invalid_inputs_fail_without_overwrite(self):
        output = self.root / "occupied"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "not empty"):
            samplePETs(str(self.input_dir), str(output), 10,
                       mode="all", seed=1)
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            samplePETs(str(self.input_dir), str(self.root / "bad-target"), -1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            samplePETs(str(self.input_dir), str(self.root / "bad-seed"), 1,
                       seed=-1)
        with self.assertRaisesRegex(ValueError, "mode must"):
            samplePETs(str(self.input_dir), str(self.root / "bad-mode"), 1,
                       mode="invalid")

    def test_zero_target_and_empty_retained_universe_are_explicit(self):
        zero = self.root / "zero"
        sampling = samplePETs(
            str(self.input_dir), str(zero), 0,
            mode="all", seed=7,
        )
        self.assertEqual(self._meta(zero)["Unique PETs"], 0)
        self.assertEqual(sampling["direction"], "downsample")
        self.assertEqual(sampling["selected_by_category"],
                         {"cis": 0, "trans": 0})
        self.assertEqual(list(zero.glob("*.ixy")), [])

        empty = self.root / "empty"
        empty.mkdir()
        (empty / "petMeta.json").write_text(json.dumps({
            "Unique PETs": 0,
            "data": {"cis": {}, "trans": {}},
        }))
        empty_zero = self.root / "empty-zero"
        empty_sampling = samplePETs(
            str(empty), str(empty_zero), 0,
            mode="all", seed=7,
        )
        self.assertEqual(empty_sampling["direction"], "identity")
        self.assertEqual(self._meta(empty_zero)["Unique PETs"], 0)
        with self.assertRaisesRegex(ValueError, "empty library"):
            samplePETs(str(empty), str(self.root / "empty-positive"), 1,
                       mode="all", seed=7)

    def test_missing_requested_trans_category_is_clear(self):
        cis_only = self.root / "cis-only"
        cis_only.mkdir()
        mat = np.array([[1, 11], [2, 12]], dtype=np.int64)
        path = cis_only / "chr1-chr1.ixy"
        joblib.dump(mat, str(path))
        (cis_only / "petMeta.json").write_text(json.dumps({
            "Unique PETs": 2,
            "data": {
                "cis": {"chr1-chr1": {"ixy": str(path.resolve())}},
                "trans": {},
            },
        }))
        with self.assertRaisesRegex(ValueError, "no retained trans"):
            samplePETs(str(cis_only), str(self.root / "missing-trans"), 1,
                       mode="trans", seed=3)


if __name__ == "__main__":
    unittest.main()
