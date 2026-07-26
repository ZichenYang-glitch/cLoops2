import json
import logging
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cLoops2.filter import samplePETs
from cLoops2.trans_agg import (
    AxisRegion, TransFeature, aggregate_trans_loops, plot_trans_aggregate,
    plot_trans_montage, plot_trans_viewpoint, read_axis_regions, trans_montage,
    trans_viewpoint, write_trans_aggregate, write_trans_montage,
    write_trans_viewpoint,
)


class TransAggregateViewpointMontageTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.predir = self.root / "input"
        self.predir.mkdir()
        self.xy = np.asarray([
            [100, 1000],
            [100, 1000],
            [110, 1020],
            [90, 980],
            [110, 980],
            # This PET is outside every local window but remains in Npair.
            [500, 5000],
        ], dtype=np.int64)
        cis = np.asarray([
            [10, 20], [30, 40], [50, 60], [70, 80],
        ], dtype=np.int64)
        trans_path = self.predir / "chr1-chr2.ixy"
        cis_path = self.predir / "chr1-chr1.ixy"
        joblib.dump(self.xy, str(trans_path))
        joblib.dump(cis, str(cis_path))
        meta = {
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
        self.feature = TransFeature(
            "center", "chr1", 100, 100, "chr2", 1000, 1000,
            "trans:chr1-chr2")
        self.x_regions = (
            AxisRegion("x-low", "chr1", 85, 104),
            AxisRegion("x-high", "chr1", 105, 114),
        )
        self.y_regions = (
            AxisRegion("y-low", "chr2", 970, 989),
            AxisRegion("y-high", "chr2", 990, 1029),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _aggregate(self, **kwargs):
        return aggregate_trans_loops(
            str(self.predir), [self.feature], x_bin_size=10,
            y_bin_size=20, x_flank_bins=1, y_flank_bins=1, **kwargs)

    def test_aggregate_is_rectangular_directional_and_never_mirrored(self):
        result = self._aggregate(method="obs")
        expected = np.asarray([
            [1, 0, 0],
            [0, 2, 0],
            [1, 0, 1],
        ], dtype=float)
        np.testing.assert_array_equal(result.matrix, expected)
        self.assertEqual(result.matrix.shape, (3, 3))
        self.assertEqual(result.matrix[2, 0], 1)
        self.assertEqual(result.matrix[0, 2], 0)
        self.assertEqual(int(result.raw_sum.sum()), 5)
        self.assertEqual(result.n_used, 1)
        self.assertEqual(result.used_record_ids, ("trans:chr1-chr2",))
        self.assertEqual(dict(result.npair_by_record),
                         {"trans:chr1-chr2": 6})
        np.testing.assert_array_equal(result.raw_x_profile, [1, 2, 2])
        np.testing.assert_array_equal(result.raw_y_profile, [2, 2, 1])

    def test_pair_and_window_oe_use_distinct_denominators(self):
        pair = self._aggregate(method="pair_oe")
        window = self._aggregate(method="window_oe")
        # Pair-global expected at the center is 2*2/6; the outside PET stays
        # in Npair. Window-conditional expected is 2*2/5.
        self.assertAlmostEqual(pair.expected_mean[1, 1], 4.0 / 6.0)
        self.assertAlmostEqual(pair.matrix[1, 1], 3.0)
        self.assertAlmostEqual(window.expected_mean[1, 1], 4.0 / 5.0)
        self.assertAlmostEqual(window.matrix[1, 1], 2.5)

    def test_undefined_oe_is_nan_with_an_explicit_valid_count(self):
        projection = self.root / "zero-oe-projection"
        samplePETs(str(self.predir), str(projection), 0,
                   mode="trans", seed=23)
        feature = TransFeature(
            "known-zero", "chr1", 100, 100, "chr2", 1000, 1000,
            "trans:chr1-chr2")
        aggregate = aggregate_trans_loops(
            str(projection), [feature], x_bin_size=10, y_bin_size=20,
            x_flank_bins=1, y_flank_bins=1, method="pair_oe")
        self.assertTrue(np.isnan(aggregate.matrix).all())
        self.assertFalse(aggregate.valid_mask.any())
        np.testing.assert_array_equal(aggregate.valid_counts,
                                      np.zeros((3, 3), dtype=np.int64))

        regions_x = (AxisRegion("x", "chr1", 0, 9999),)
        regions_y = (AxisRegion("y", "chr2", 0, 9999),)
        montage = trans_montage(
            str(projection), "chr1", "chr2", regions_x, regions_y,
            method="pair_oe")
        self.assertTrue(np.isnan(montage.values[0, 0]))
        self.assertFalse(montage.valid_mask[0, 0])

        viewpoint = trans_viewpoint(
            str(projection), "chr1", "chr2", "x", 95, 105,
            970, 1029, 20, method="pair_oe")
        self.assertTrue(np.isnan(viewpoint.values).all())
        self.assertFalse(viewpoint.valid_mask.any())

    def test_fixed_axis_resolutions_can_produce_a_non_square_aggregate(self):
        result = aggregate_trans_loops(
            str(self.predir), [self.feature], x_bin_size=10,
            y_bin_size=20, x_flank_bins=2, y_flank_bins=1)
        self.assertEqual(result.matrix.shape, (5, 3))

    def test_multi_pair_aggregate_is_equal_feature_not_npair_weighted(self):
        second_xy = np.asarray([
            [90, 980], [100, 1000], [110, 1020],
        ], dtype=np.int64)
        second_path = self.predir / "chr1-chr3.ixy"
        joblib.dump(second_xy, str(second_path))
        meta_path = self.predir / "petMeta.json"
        meta = json.loads(meta_path.read_text())
        meta["Unique PETs"] += 3
        meta["Unique Trans PETs"] += 3
        meta["Total PETs"] += 3
        meta["Total Trans PETs"] += 3
        meta["data"]["trans"]["chr1-chr3"] = {
            "ixy": str(second_path),
            "chromX": "chr1",
            "chromY": "chr3",
            "record_id": "trans:chr1-chr3",
        }
        meta_path.write_text(json.dumps(meta))
        second = TransFeature(
            "second", "chr1", 100, 100, "chr3", 1000, 1000,
            "trans:chr1-chr3")
        kwargs = dict(x_bin_size=10, y_bin_size=20, x_flank_bins=1,
                      y_flank_bins=1, method="pair_oe")
        first_only = aggregate_trans_loops(
            str(self.predir), [self.feature], **kwargs)
        second_only = aggregate_trans_loops(
            str(self.predir), [second], **kwargs)
        combined = aggregate_trans_loops(
            str(self.predir), [self.feature, second], **kwargs)
        np.testing.assert_allclose(
            combined.matrix,
            (first_only.matrix + second_only.matrix) / 2.0)
        np.testing.assert_array_equal(combined.valid_counts,
                                      np.full((3, 3), 2))
        self.assertEqual(dict(combined.npair_by_record), {
            "trans:chr1-chr2": 6,
            "trans:chr1-chr3": 3,
        })

    def test_edge_policy_records_or_rejects_unaligned_features(self):
        edge = TransFeature(
            "edge", "chr1", 1, 1, "chr2", 1, 1,
            "trans:chr1-chr2")
        kept = aggregate_trans_loops(
            str(self.predir), [self.feature, edge], x_bin_size=10,
            y_bin_size=20, x_flank_bins=1, y_flank_bins=1,
            edge_policy="skip")
        self.assertEqual(kept.used_feature_ids, ("center",))
        self.assertEqual(kept.skipped_feature_ids, ("edge",))
        with self.assertRaisesRegex(ValueError, "below coordinate zero"):
            aggregate_trans_loops(
                str(self.predir), [edge], x_bin_size=10,
                y_bin_size=20, x_flank_bins=1, y_flank_bins=1,
                edge_policy="error")

    def test_duplicate_feature_ids_are_rejected_before_weighting(self):
        duplicate = TransFeature(
            "center", "chr1", 110, 110, "chr2", 1020, 1020,
            "trans:chr1-chr2")
        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            aggregate_trans_loops(
                str(self.predir), [self.feature, duplicate], x_bin_size=10,
                y_bin_size=20, x_flank_bins=1, y_flank_bins=1)

    def test_viewpoint_pair_and_window_oe_are_directional(self):
        pair = trans_viewpoint(
            str(self.predir), "chr1", "chr2", "x", 95, 105,
            970, 1029, 20, method="pair_oe")
        np.testing.assert_array_equal(pair.observed, [0, 2, 0])
        np.testing.assert_allclose(pair.expected, [2.0 / 3.0,
                                                   2.0 / 3.0,
                                                   1.0 / 3.0])
        self.assertAlmostEqual(pair.values[1], 3.0)
        self.assertEqual(pair.anchor_chrom, "chr1")
        self.assertEqual(pair.target_chrom, "chr2")
        self.assertEqual(pair.n_pair, 6)

        window = trans_viewpoint(
            str(self.predir), "chr1", "chr2", "x", 95, 105,
            970, 1029, 20, method="window_oe",
            context_start=85, context_end=114)
        np.testing.assert_allclose(window.expected, [0.8, 0.8, 0.4])
        self.assertAlmostEqual(window.values[1], 2.5)

    def test_reverse_viewpoint_swaps_roles_not_coordinate_columns(self):
        result = trans_viewpoint(
            str(self.predir), "chr1", "chr2", "y", 975, 985,
            85, 114, 10, method="obs")
        np.testing.assert_array_equal(result.observed, [1, 0, 1])
        self.assertEqual(result.anchor_chrom, "chr2")
        self.assertEqual(result.target_chrom, "chr1")
        with self.assertRaisesRegex(ValueError, "reversed relative"):
            trans_viewpoint(
                str(self.predir), "chr2", "chr1", "x", 975, 985,
                85, 114, 10, method="obs")

    def test_window_viewpoint_requires_non_degenerate_context(self):
        with self.assertRaisesRegex(ValueError, "requires context"):
            trans_viewpoint(
                str(self.predir), "chr1", "chr2", "x", 95, 105,
                970, 1029, 20, method="window_oe")
        with self.assertRaisesRegex(ValueError, "contain the anchor"):
            trans_viewpoint(
                str(self.predir), "chr1", "chr2", "x", 95, 105,
                970, 1029, 20, method="window_oe",
                context_start=100, context_end=114)
        with self.assertRaisesRegex(ValueError, "strictly extend"):
            trans_viewpoint(
                str(self.predir), "chr1", "chr2", "x", 95, 105,
                970, 1029, 20, method="window_oe",
                context_start=95, context_end=105)
        with self.assertRaisesRegex(ValueError, "background is degenerate"):
            trans_viewpoint(
                str(self.predir), "chr1", "chr2", "x", 95, 105,
                990, 1009, 20, method="window_oe",
                context_start=85, context_end=105)

    def test_viewpoint_and_montage_dense_guards_precede_allocation(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_bins"):
            trans_viewpoint(
                str(self.predir), "chr1", "chr2", "x", 95, 105,
                0, 10000, 1, max_bins=100)
        with self.assertRaisesRegex(ValueError, "exceeds max_cells"):
            trans_montage(
                str(self.predir), "chr1", "chr2", self.x_regions,
                self.y_regions, max_cells=3)

    def test_montage_is_x_by_y_and_uses_pair_or_window_marginals(self):
        observed = trans_montage(
            str(self.predir), "chr1", "chr2", self.x_regions,
            self.y_regions, method="obs")
        np.testing.assert_array_equal(observed.observed, [[1, 2], [1, 1]])
        self.assertEqual(observed.x_region_ids, ("x-low", "x-high"))
        self.assertEqual(observed.y_region_ids, ("y-low", "y-high"))

        pair = trans_montage(
            str(self.predir), "chr1", "chr2", self.x_regions,
            self.y_regions, method="pair_oe")
        np.testing.assert_allclose(pair.expected,
                                   [[1.0, 1.5], [2.0 / 3.0, 1.0]])
        np.testing.assert_allclose(pair.values,
                                   [[1.0, 4.0 / 3.0], [1.5, 1.0]])

        window = trans_montage(
            str(self.predir), "chr1", "chr2", self.x_regions,
            self.y_regions, method="window_oe")
        np.testing.assert_allclose(window.expected,
                                   [[1.2, 1.8], [0.8, 1.2]])

    def test_montage_rejects_overlapping_or_wrong_axis_regions(self):
        overlapping = self.x_regions + (
            AxisRegion("x-overlap", "chr1", 100, 120),)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            trans_montage(
                str(self.predir), "chr1", "chr2", overlapping,
                self.y_regions)
        wrong = (AxisRegion("wrong", "chr3", 1, 2),)
        with self.assertRaisesRegex(ValueError, "expected chr1"):
            trans_montage(
                str(self.predir), "chr1", "chr2", wrong, self.y_regions)

    def test_bed_region_reader_converts_half_open_end_explicitly(self):
        bed = self.root / "x.bed"
        bed.write_text("chr1\t85\t105\tx-low\n")
        half_open = read_axis_regions(str(bed), expected_chrom="chr1")
        closed = read_axis_regions(
            str(bed), expected_chrom="chr1", bed_half_open=False)
        self.assertEqual((half_open[0].start, half_open[0].end), (85, 104))
        self.assertEqual((closed[0].start, closed[0].end), (85, 105))

    def test_projection_rpm_uses_lglobal_not_physical_trans_rows(self):
        projection = self.root / "trans-projection"
        samplePETs(str(self.predir), str(projection), 8,
                   mode="trans", seed=31)
        meta = json.loads((projection / "petMeta.json").read_text())
        physical_trans = int(meta["Unique PETs"])
        self.assertLess(physical_trans, 8)
        x_regions = (AxisRegion("all-x", "chr1", 0, 9999),)
        y_regions = (AxisRegion("all-y", "chr2", 0, 9999),)
        raw = trans_montage(
            str(projection), "chr1", "chr2", x_regions, y_regions)
        rpm = trans_montage(
            str(projection), "chr1", "chr2", x_regions, y_regions,
            normalization="library_rpm")
        self.assertEqual(raw.observed[0, 0], physical_trans)
        self.assertEqual(rpm.logical_total, 8)
        self.assertAlmostEqual(
            rpm.values[0, 0], physical_trans / 8.0 * 1000000.0)
        self.assertNotAlmostEqual(
            rpm.values[0, 0], physical_trans / float(physical_trans) * 1000000.0)

    def test_known_zero_sampled_pair_is_zero_not_missing(self):
        projection = self.root / "zero-trans-projection"
        samplePETs(str(self.predir), str(projection), 0,
                   mode="trans", seed=19)
        x_regions = (AxisRegion("all-x", "chr1", 0, 9999),)
        y_regions = (AxisRegion("all-y", "chr2", 0, 9999),)
        result = trans_montage(
            str(projection), "chr1", "chr2", x_regions, y_regions)
        self.assertEqual(result.n_pair, 0)
        self.assertEqual(result.observed[0, 0], 0)
        self.assertEqual(result.logical_total, 0)

    def test_legacy_trans_only_allows_pair_descriptives_but_not_rpm(self):
        legacy = self.root / "legacy-trans-only"
        legacy.mkdir()
        path = legacy / "chr1-chr2.ixy"
        joblib.dump(self.xy, str(path))
        (legacy / "petMeta.json").write_text(json.dumps({
            "Unique PETs": 6,
            "data": {
                "cis": {},
                "trans": {
                    "chr1-chr2": {
                        "ixy": str(path),
                        "chromX": "chr1",
                        "chromY": "chr2",
                        "record_id": "trans:chr1-chr2",
                    },
                },
            },
        }))
        raw = trans_montage(
            str(legacy), "chr1", "chr2", self.x_regions, self.y_regions,
            method="pair_oe")
        self.assertEqual(raw.n_pair, 6)
        self.assertIsNone(raw.logical_total)
        with self.assertRaisesRegex(ValueError, "global_normalization"):
            trans_montage(
                str(legacy), "chr1", "chr2", self.x_regions,
                self.y_regions, normalization="library_rpm")

    def test_oe_cannot_be_double_normalized_by_library_depth(self):
        with self.assertRaisesRegex(ValueError, "dimensionless"):
            self._aggregate(method="pair_oe", normalization="library_rpm")
        with self.assertRaisesRegex(ValueError, "dimensionless"):
            trans_montage(
                str(self.predir), "chr1", "chr2", self.x_regions,
                self.y_regions, method="window_oe",
                normalization="library_rpm")

    def test_writers_and_minimal_plots_preserve_provenance(self):
        aggregate = self._aggregate(method="pair_oe")
        viewpoint = trans_viewpoint(
            str(self.predir), "chr1", "chr2", "x", 95, 105,
            970, 1029, 20, method="pair_oe")
        montage = trans_montage(
            str(self.predir), "chr1", "chr2", self.x_regions,
            self.y_regions, method="pair_oe")

        aggregate_paths = write_trans_aggregate(
            aggregate, self.root / "aggregate")
        viewpoint_paths = write_trans_viewpoint(
            viewpoint, self.root / "viewpoint")
        montage_paths = write_trans_montage(
            montage, self.root / "montage")
        for path in aggregate_paths + viewpoint_paths + montage_paths:
            self.assertTrue(Path(path).is_file(), path)
        aggregate_meta = json.loads(Path(aggregate_paths[1]).read_text())
        viewpoint_meta = json.loads(Path(viewpoint_paths[1]).read_text())
        montage_meta = json.loads(Path(montage_paths[2]).read_text())
        self.assertEqual(
            aggregate_meta["axis_semantics"],
            "rows=chromX,columns=chromY,no_mirroring")
        self.assertEqual(aggregate_meta["Npair_by_record"],
                         {"trans:chr1-chr2": 6})
        self.assertEqual(aggregate_meta["oe_aggregation"],
                         "mean_of_feature_level_ratios_over_defined_cells")
        self.assertEqual(aggregate_meta["used_features"][0]["chromX"],
                         "chr1")
        self.assertEqual(viewpoint_meta["Npair"], 6)
        self.assertEqual(montage_meta["record_id"], "trans:chr1-chr2")
        self.assertEqual(montage_meta["x_regions"][0],
                         {"id": "x-low", "start": 85, "end": 104})

        old_mpl_config = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = str(self.root / "mpl")
        try:
            plot_trans_aggregate(aggregate, self.root / "aggregate.pdf")
            plot_trans_viewpoint(viewpoint, self.root / "viewpoint.pdf")
            plot_trans_montage(montage, self.root / "montage.pdf")
        finally:
            if old_mpl_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_mpl_config
        for name in ("aggregate.pdf", "viewpoint.pdf", "montage.pdf"):
            self.assertTrue((self.root / name).is_file())


if __name__ == "__main__":
    unittest.main()
