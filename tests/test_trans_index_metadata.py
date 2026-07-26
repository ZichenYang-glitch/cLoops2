#!/usr/bin/env python
#--coding:utf-8--

import unittest

import numpy as np

from cLoops2.ds import TransContactIndex
from cLoops2.metadata import (CAP_FORMAL_INFERENCE,
                              CAP_GLOBAL_NORMALIZATION,
                              CAP_NESTED_DOWNSAMPLE,
                              CAP_PAIR_COUNTS, MetadataCapabilityError,
                              build_library_context)


class TestTransContactIndex(unittest.TestCase):

    def test_crossed_coordinates_do_not_mix_axes(self):
        index = TransContactIndex([100, 1000], [1000, 100])
        self.assertEqual(index.count_x(90, 110), 1)
        self.assertEqual(index.count_y(90, 110), 1)
        self.assertEqual(index.count_rect(90, 110, 90, 110), 0)
        np.testing.assert_array_equal(
            index.query_rect_ids(90, 110, 90, 110),
            np.array([], dtype=np.int64),
        )

    def test_closed_intervals_and_original_ids(self):
        mat = np.array([[10, 30], [10, 20], [20, 20], [30, 10]],
                       dtype=np.int32)
        index = TransContactIndex.from_matrix(mat)
        self.assertEqual(index.count_x(10, 20), 3)
        self.assertEqual(index.count_y(20, 20), 2)
        np.testing.assert_array_equal(
            index.query_rect_ids(10, 20, 20, 30), np.array([0, 1, 2]))
        # Equal coordinates retain deterministic original-row ids.
        np.testing.assert_array_equal(
            index.query_rect_ids(10, 10, 0, 100), np.array([0, 1]))

    def test_empty_index(self):
        index = TransContactIndex([], [])
        self.assertEqual(index.number, 0)
        self.assertEqual(index.count_rect(0, 10, 0, 10), 0)

    def test_input_and_interval_validation(self):
        with self.assertRaises(ValueError):
            TransContactIndex([1], [1, 2])
        with self.assertRaises(ValueError):
            TransContactIndex.from_matrix(np.zeros((2, 3), dtype=int))
        with self.assertRaises(TypeError):
            TransContactIndex([1.5], [2.5])
        with self.assertRaises(TypeError):
            TransContactIndex.from_matrix(np.zeros((2, 2), dtype=float))
        with self.assertRaises(OverflowError):
            TransContactIndex(np.array([2**63], dtype=np.uint64),
                              np.array([1], dtype=np.uint64))
        index = TransContactIndex([1], [2])
        with self.assertRaises(ValueError):
            index.count_x(2, 1)
        with self.assertRaises(TypeError):
            index.count_y(1.0, 2)


def _record(record_id, category, chrom_x, chrom_y, source, selected,
            written):
    return {
        "record_id": record_id,
        "category": category,
        "key": record_id.split(":", 1)[-1],
        "chromX": chrom_x,
        "chromY": chrom_y,
        "source_path": "/input/%s.ixy" % record_id,
        "output_path": "/output/%s.ixy" % record_id if written else None,
        "source_rows": source,
        "selected_rows": selected,
        "written_rows": written,
        "shape": [source, 2],
        "dtype": "int64",
    }


def _sample_meta(emit="all", replacement=False):
    records = [
        _record("cis:chr1-chr1", "cis", "chr1", "chr1", 4, 2,
                2 if emit in ("cis", "all") else 0),
        _record("trans:chr1-chr2", "trans", "chr1", "chr2", 6, 4,
                4 if emit in ("trans", "all") else 0),
    ]
    written = {
        "cis": 2 if emit in ("cis", "all") else 0,
        "trans": 4 if emit in ("trans", "all") else 0,
    }
    return {
        "Unique PETs": sum(written.values()),
        "data": {
            "cis": ({"chr1-chr1": {"ixy": "/output/chr1-chr1.ixy"}}
                    if written["cis"] else {}),
            "trans": ({"chr1-chr2": {"ixy": "/output/chr1-chr2.ixy"}}
                      if written["trans"] else {}),
        },
        "Sampling": {
            "schema_version": 2,
            "mode": emit,
            "universe": "retained_all",
            "emit": emit,
            "source_kind": "all_sample" if emit == "all" else "projection",
            "source_logical_total": 10,
            "target_logical_total": 6,
            "physical_output_total": sum(written.values()),
            "available_by_category": {"cis": 4, "trans": 6},
            "selected_by_category": {"cis": 2, "trans": 4},
            "written_by_category": written,
            "direction": "downsample",
            "replacement_this_step": replacement,
            "replacement_ever": replacement,
            "records": records,
        },
    }


class TestLibraryContext(unittest.TestCase):

    def test_valid_all_sample_closes_physical_and_logical_totals(self):
        meta = _sample_meta("all")
        actual = {
            "cis:chr1-chr1": 2,
            "trans:chr1-chr2": 4,
        }
        ctx = build_library_context(meta, actual)
        self.assertEqual(ctx.validity, "valid")
        self.assertEqual(ctx.physical_total, 6)
        self.assertEqual(ctx.logical_total, 6)
        self.assertEqual(ctx.emit, "all")
        self.assertEqual(ctx.source_kind, "all_sample")
        self.assertEqual(ctx.get_npair("trans:chr1-chr2"), 4)
        self.assertIn(CAP_GLOBAL_NORMALIZATION, ctx.capabilities)
        self.assertIn(CAP_FORMAL_INFERENCE, ctx.capabilities)

    def test_projection_keeps_logical_depth_and_zero_manifest_pair(self):
        meta = _sample_meta("trans")
        ctx = build_library_context(meta, {"trans:chr1-chr2": 4})
        self.assertEqual(ctx.validity, "valid")
        self.assertEqual(ctx.physical_total, 4)
        self.assertEqual(ctx.logical_total, 6)
        self.assertEqual(ctx.get_npair("cis:chr1-chr1"), 0)
        self.assertEqual(ctx.get_npair("trans:chr1-chr2"), 4)
        self.assertIn(CAP_NESTED_DOWNSAMPLE, ctx.capabilities)

    def test_replacement_removes_formal_inference_capability(self):
        meta = _sample_meta("all", replacement=True)
        # Replacement requires a logically larger target/direction.  Keep the
        # closure valid while testing the provenance capability itself.
        meta["Sampling"]["source_logical_total"] = 4
        meta["Sampling"]["direction"] = "upsample"
        meta["Sampling"]["records"][0]["source_rows"] = 2
        meta["Sampling"]["records"][0]["shape"][0] = 2
        meta["Sampling"]["records"][1]["source_rows"] = 2
        meta["Sampling"]["records"][1]["shape"][0] = 2
        meta["Sampling"]["available_by_category"] = {"cis": 2, "trans": 2}
        ctx = build_library_context(meta)
        self.assertEqual(ctx.validity, "valid")
        self.assertTrue(ctx.replacement_ever)
        self.assertNotIn(CAP_FORMAL_INFERENCE, ctx.capabilities)
        with self.assertRaises(MetadataCapabilityError):
            ctx.require(CAP_FORMAL_INFERENCE)

    def test_parent_replacement_remains_ineligible_after_downsample(self):
        meta = _sample_meta("all")
        meta["Sampling"]["replacement_ever"] = True
        ctx = build_library_context(meta)
        self.assertEqual(ctx.validity, "valid")
        self.assertFalse(ctx.replacement_this_step)
        self.assertTrue(ctx.replacement_ever)
        self.assertNotIn(CAP_FORMAL_INFERENCE, ctx.capabilities)

    def test_actual_rows_and_unique_mismatch_are_invalid(self):
        meta = _sample_meta("all")
        ctx = build_library_context(meta, {
            "cis:chr1-chr1": 1,
            "trans:chr1-chr2": 4,
        })
        self.assertEqual(ctx.validity, "invalid")
        self.assertEqual(ctx.physical_total, 5)
        self.assertIn(CAP_PAIR_COUNTS, ctx.capabilities)
        self.assertNotIn(CAP_GLOBAL_NORMALIZATION, ctx.capabilities)

    def test_legacy_pre_root_uses_only_retained_physical_rows(self):
        meta = {
            "Total PETs": 15,
            "Total Cis PETs": 4,
            "Total Trans PETs": 11,
            "Unique PETs": 4,
            "Unique Cis PETs": 4,
            "Unique Trans PETs": 0,
            "Retention": {
                "retain trans": False,
                "retained categories": ["cis"],
                "chromosome whitelist": [],
                "cut": 0,
                "mcut": -1,
            },
            "data": {
                "cis": {"chr1-chr1": {"ixy": "/root/chr1-chr1.ixy"}},
                "trans": {},
            },
        }
        ctx = build_library_context(meta, {"cis": {"chr1-chr1": 4},
                                                   "trans": {}})
        self.assertEqual(ctx.validity, "valid")
        self.assertEqual(ctx.physical_total, 4)
        self.assertEqual(ctx.logical_total, 4)
        self.assertEqual(ctx.normalization_scope, "retained_root")
        self.assertFalse(ctx.replacement_ever)

    def test_inconsistent_retention_is_invalid(self):
        meta = {
            "Total PETs": 4,
            "Total Cis PETs": 0,
            "Total Trans PETs": 4,
            "Unique PETs": 4,
            "Unique Cis PETs": 0,
            "Unique Trans PETs": 4,
            "Retention": {
                "retain trans": False,
                "retained categories": ["cis"],
            },
            "data": {
                "cis": {},
                "trans": {"chr1-chr2": {"ixy": "/root/chr1-chr2.ixy"}},
            },
        }
        ctx = build_library_context(meta, {"trans": {"chr1-chr2": 4}})
        self.assertEqual(ctx.validity, "invalid")
        self.assertNotIn(CAP_GLOBAL_NORMALIZATION, ctx.capabilities)
        self.assertTrue(any("not retained" in reason for reason in ctx.reasons))

    def test_historical_pre_root_is_cautiously_inferred(self):
        meta = {
            "Total PETs": 4,
            "Total Cis PETs": 4,
            "Total Trans PETs": 0,
            "Unique PETs": 4,
            "data": {
                "cis": {"chr1-chr1": {"ixy": "/old/chr1-chr1.ixy"}},
                "trans": {},
            },
        }
        ctx = build_library_context(meta, {"cis": {"chr1-chr1": 4},
                                                   "trans": {}})
        self.assertEqual(ctx.validity, "legacy_root_assumed")
        self.assertEqual(ctx.logical_total, 4)
        self.assertEqual(ctx.normalization_scope, "legacy_retained_root")

    def test_legacy_trans_only_without_provenance_has_unknown_lglobal(self):
        meta = {
            "Unique PETs": 6,
            "data": {
                "cis": {},
                "trans": {"chr1-chr2": {"ixy": "/old/chr1-chr2.ixy"}},
            },
        }
        ctx = build_library_context(meta, {"trans": {"chr1-chr2": 6}})
        self.assertEqual(ctx.validity, "unknown")
        self.assertEqual(ctx.physical_total, 6)
        self.assertIsNone(ctx.logical_total)
        self.assertIsNone(ctx.replacement_ever)
        self.assertNotIn(CAP_GLOBAL_NORMALIZATION, ctx.capabilities)
        with self.assertRaises(MetadataCapabilityError):
            ctx.require(CAP_GLOBAL_NORMALIZATION)

    def test_manifest_shape_and_category_are_validated(self):
        meta = _sample_meta("all")
        meta["Sampling"]["records"][1]["shape"] = [6, 3]
        meta["Sampling"]["records"][1]["chromY"] = "chr1"
        ctx = build_library_context(meta)
        self.assertEqual(ctx.validity, "invalid")
        self.assertNotIn(CAP_PAIR_COUNTS, ctx.capabilities)
        self.assertTrue(any("shape[1]" in reason for reason in ctx.reasons))
        self.assertTrue(any("category disagrees" in reason
                            for reason in ctx.reasons))


if __name__ == "__main__":
    unittest.main()
