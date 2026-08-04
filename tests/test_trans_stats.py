#!/usr/bin/env python
# --coding:utf-8--

import os
import tempfile
import unittest
from typing import Optional

import numpy as np

from cLoops2.ds import TransContactIndex
from cLoops2.metadata import (CAP_FORMAL_INFERENCE,
                              CAP_GLOBAL_NORMALIZATION, CAP_PAIR_COUNTS,
                              CAP_PHYSICAL_ROWS, LibraryContext)
from cLoops2.trans_stats import (CONTINUITY_CORRECTION,
                                 FormalInferenceError, TSV_COLUMNS,
                                 TransCandidate, adjust_pvalues,
                                 describe_de_novo_candidates,
                                 evaluate_fixed_candidates,
                                 read_trans_loop_results,
                                 test_split_validation_candidates,
                                 write_trans_loop_results)


RECORD = "trans:chr1-chr2"


def _context(pair_pets,
             logical_total: Optional[int] = 100,
             replacement_ever=False,
             formal=True):
    capabilities = {CAP_PAIR_COUNTS, CAP_PHYSICAL_ROWS}
    if logical_total is not None:
        capabilities.add(CAP_GLOBAL_NORMALIZATION)
    if formal:
        capabilities.add(CAP_FORMAL_INFERENCE)
    return LibraryContext(
        physical_total=pair_pets,
        logical_total=logical_total,
        npair_by_record={RECORD: pair_pets},
        emit="trans",
        source_kind="projection",
        replacement_this_step=replacement_ever,
        replacement_ever=replacement_ever,
        validity="valid",
        capabilities=frozenset(capabilities),
        reasons=(),
        normalization_scope="retained_all",
    )


def _candidate(loop_id="fixed-1",
               x_start=10,
               x_end=20,
               y_start=100,
               y_end=110,
               source="independent:hand-table"):
    return TransCandidate(
        loop_id=loop_id,
        chrom_x="chr1",
        x_start=x_start,
        x_end=x_end,
        chrom_y="chr2",
        y_start=y_start,
        y_end=y_end,
        record_id=RECORD,
        candidate_source=source,
    )


def _hand_table_index():
    # Candidate A=10..20, B=100..110 gives the exact table:
    #             Y in B  Y not B
    # X in A          3        1
    # X not A         2        4
    rows = ([[15, 105]] * 3 + [[15, 200]] + [[50, 105]] * 2 +
            [[50, 200]] * 4)
    return TransContactIndex.from_matrix(np.asarray(rows, dtype=np.int64))


class TestExactDirectionalStatistics(unittest.TestCase):

    def test_global_hand_calculated_2x2_table(self):
        index = _hand_table_index()
        result = evaluate_fixed_candidates(
            [_candidate()], {RECORD: index}, _context(10), local_pad=50,
            test_scope="global")[0]

        self.assertEqual(result.pair_pets, 10)
        self.assertEqual(result.reads_a, 4)
        self.assertEqual(result.reads_b, 5)
        self.assertEqual(result.pets, 3)
        expected_global = result.expected_global
        global_enrichment = result.global_enrichment
        library_rpm = result.library_rpm
        assert expected_global is not None
        assert global_enrichment is not None
        assert library_rpm is not None
        self.assertAlmostEqual(expected_global, 2.0)
        self.assertAlmostEqual(
            global_enrichment,
            (3.0 + CONTINUITY_CORRECTION) /
            (2.0 + CONTINUITY_CORRECTION),
        )
        # Corrected OR=(3.5*4.5)/(1.5*2.5)=4.2.
        self.assertAlmostEqual(result.odds_ratio_global, 4.2)
        # P[X>=3] = (C(4,3)C(6,2)+C(4,4)C(6,1))/C(10,5)
        #         = 66/252 = 11/42.
        self.assertAlmostEqual(result.global_conditional_pvalue, 11.0 / 42.0)
        self.assertEqual(result.primary_pvalue,
                         result.global_conditional_pvalue)
        self.assertEqual(result.local_status, "not_requested_global_scope")
        self.assertIsNone(result.local_conditional_pvalue)
        self.assertIsNone(result.local_test_available)
        self.assertAlmostEqual(library_rpm, 30000.0)

    def test_local_hand_calculated_table_and_iut(self):
        # A=100..110, B=200..210, pad=20.  In WX x WY the table is:
        #             Y in B  Y not B
        # X in A          2        1
        # X not A         2        3
        # Two extra PETs are outside the local window but remain in Npair.
        rows = ([[105, 205]] * 2 + [[105, 220]] + [[120, 205]] * 2 +
                [[120, 220]] * 3 + [[1000, 1000]] * 2)
        index = TransContactIndex.from_matrix(np.asarray(rows, dtype=np.int64))
        candidate = _candidate(x_start=100,
                               x_end=110,
                               y_start=200,
                               y_end=210)
        result = evaluate_fixed_candidates(
            [candidate], {RECORD: index}, _context(10), local_pad=20,
            test_scope="both")[0]

        self.assertEqual(result.local_x_start, 80)
        self.assertEqual(result.local_x_end, 130)
        self.assertEqual(result.local_y_start, 180)
        self.assertEqual(result.local_y_end, 230)
        self.assertEqual(result.local_pair_pets, 8)
        self.assertEqual(result.local_reads_a, 3)
        self.assertEqual(result.local_reads_b, 4)
        self.assertEqual(result.pets, 2)
        expected_local = result.expected_local
        local_enrichment = result.local_enrichment
        odds_ratio_local = result.odds_ratio_local
        local_pvalue = result.local_conditional_pvalue
        assert expected_local is not None
        assert local_enrichment is not None
        assert odds_ratio_local is not None
        assert local_pvalue is not None
        self.assertAlmostEqual(expected_local, 1.5)
        self.assertAlmostEqual(local_enrichment, 2.5 / 2.0)
        self.assertAlmostEqual(odds_ratio_local,
                               (2.5 * 3.5) / (1.5 * 2.5))
        self.assertTrue(result.local_test_available)
        self.assertEqual(result.local_status, "available")
        self.assertEqual(
            result.primary_pvalue,
            max(result.global_conditional_pvalue, local_pvalue),
        )

    def test_crossed_coordinates_do_not_mix_axes(self):
        index = TransContactIndex([100, 1000], [1000, 100])
        candidate = _candidate(x_start=90,
                               x_end=110,
                               y_start=90,
                               y_end=110)
        result = evaluate_fixed_candidates(
            [candidate], {RECORD: index}, _context(2), local_pad=0,
            test_scope="global")[0]
        self.assertEqual(result.reads_a, 1)
        self.assertEqual(result.reads_b, 1)
        self.assertEqual(result.pets, 0)
        self.assertEqual(result.global_conditional_pvalue, 1.0)

    def test_both_scope_does_not_fallback_when_local_table_degenerates(self):
        index = _hand_table_index()
        result = evaluate_fixed_candidates(
            [_candidate()], {RECORD: index}, _context(10), local_pad=0,
            test_scope="both")[0]
        self.assertFalse(result.local_test_available)
        self.assertEqual(result.local_status, "unavailable_no_background")
        self.assertEqual(result.local_conditional_pvalue, 1.0)
        self.assertEqual(result.primary_pvalue, 1.0)
        self.assertLess(result.global_conditional_pvalue, 1.0)

    def test_local_lower_bound_is_clipped_and_recorded(self):
        index = TransContactIndex([0, 3, 20], [0, 3, 20])
        candidate = _candidate(x_start=0,
                               x_end=0,
                               y_start=0,
                               y_end=0)
        result = evaluate_fixed_candidates(
            [candidate], {RECORD: index}, _context(3), local_pad=5,
            test_scope="both")[0]
        self.assertEqual(result.local_x_start, 0)
        self.assertEqual(result.local_y_start, 0)
        self.assertTrue(result.local_lower_clipped)


class TestFamilyCorrectionAndGuards(unittest.TestCase):

    def test_bh_and_by_known_values(self):
        pvalues = [0.01, 0.04, 0.03]
        bh = adjust_pvalues(pvalues, "BH")
        by = adjust_pvalues(pvalues, "BY")
        expected_bh = [0.03, 0.04, 0.04]
        harmonic = 1.0 + 1.0 / 2.0 + 1.0 / 3.0
        expected_by = [value * harmonic for value in expected_bh]
        np.testing.assert_allclose(bh, expected_bh)
        np.testing.assert_allclose(by, expected_by)

    def test_full_family_keeps_zero_count_candidates(self):
        index = _hand_table_index()
        candidates = [
            _candidate("hit"),
            _candidate("zero", x_start=500, x_end=510,
                       y_start=500, y_end=510),
        ]
        results = evaluate_fixed_candidates(
            candidates, {RECORD: index}, _context(10), local_pad=20,
            test_scope="global", adjustment="BY")
        self.assertEqual([result.loop_id for result in results], ["hit", "zero"])
        self.assertEqual(results[1].pets, 0)
        self.assertEqual(results[1].primary_pvalue, 1.0)
        for result in results:
            self.assertIsNotNone(result.bh_adjusted_p)
            self.assertIsNotNone(result.by_adjusted_p)
            self.assertEqual(result.adjustment_method, "BY")
            by_adjusted_p = result.by_adjusted_p
            alpha = result.alpha
            assert by_adjusted_p is not None
            assert alpha is not None
            self.assertEqual(result.significant, by_adjusted_p <= alpha)

    def test_replacement_and_missing_formal_capability_are_rejected(self):
        index = _hand_table_index()
        replacement_context = _context(10,
                                       replacement_ever=True,
                                       formal=False)
        with self.assertRaisesRegex(FormalInferenceError,
                                    "replacement_ever=false"):
            evaluate_fixed_candidates([_candidate()], {RECORD: index},
                                      replacement_context, local_pad=20)

        no_formal_context = _context(10,
                                     replacement_ever=False,
                                     formal=False)
        with self.assertRaises(FormalInferenceError):
            evaluate_fixed_candidates([_candidate()], {RECORD: index},
                                      no_formal_context, local_pad=20)

    def test_pair_manifest_must_match_concrete_index(self):
        with self.assertRaisesRegex(ValueError, "pair count mismatch"):
            evaluate_fixed_candidates([_candidate()],
                                      {RECORD: _hand_table_index()},
                                      _context(9),
                                      local_pad=20)

    def test_candidate_source_is_required(self):
        with self.assertRaisesRegex(ValueError, "candidate_source"):
            _candidate(source="  ")

    def test_de_novo_track_never_emits_adjusted_or_significant(self):
        index = _hand_table_index()
        context = _context(10, replacement_ever=True, formal=False)
        result = describe_de_novo_candidates(
            [_candidate(source="de_novo:blockDBSCAN")],
            {RECORD: index},
            context,
            local_pad=20,
            test_scope="global",
        )[0]
        self.assertEqual(result.inference_mode, "exploratory_de_novo")
        self.assertFalse(result.inferential_validity)
        self.assertFalse(result.selection_adjusted)
        self.assertIsNone(result.bh_adjusted_p)
        self.assertIsNone(result.by_adjusted_p)
        self.assertIsNone(result.adjustment_method)
        self.assertIsNone(result.alpha)
        self.assertIsNone(result.significant)

    def test_exploratory_pair_stats_allow_unavailable_lglobal_without_guessing(self):
        index = _hand_table_index()
        context = _context(10, logical_total=None, formal=False)
        result = describe_de_novo_candidates(
            [_candidate(source="de_novo:external")], {RECORD: index}, context,
            local_pad=20, test_scope="global")[0]
        self.assertIsNone(result.library_depth)
        self.assertIsNone(result.library_rpm)
        self.assertEqual(result.pair_pets, 10)

    def test_split_validation_uses_only_held_out_counts_and_one_family(self):
        # Source has ten rows, but the validation index deliberately contains
        # only the first four.  Discovery geometry is represented solely by the
        # fixed candidate and its six-row provenance count.
        validation_rows = np.asarray(
            [[15, 105], [15, 105], [15, 105], [15, 200]],
            dtype=np.int64)
        validation = TransContactIndex.from_matrix(validation_rows)
        candidates = [
            _candidate("held-out-hit"),
            _candidate("held-out-zero", x_start=500, x_end=510,
                       y_start=500, y_end=510),
        ]
        results = test_split_validation_candidates(
            candidates,
            {RECORD: validation},
            {RECORD: 6},
            _context(10, logical_total=100),
            validation_fraction=0.4,
            local_pad=20,
            test_scope="global",
            adjustment="BY",
            alpha=0.05,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual([result.pair_pets for result in results], [4, 4])
        self.assertEqual([result.pets for result in results], [3, 0])
        self.assertEqual([result.library_depth for result in results], [40, 40])
        for result in results:
            self.assertEqual(result.inference_mode,
                             "formal_split_validation")
            self.assertTrue(result.selection_adjusted)
            self.assertTrue(result.inferential_validity)
            self.assertIsNotNone(result.bh_adjusted_p)
            self.assertIsNotNone(result.by_adjusted_p)
            self.assertEqual(result.adjustment_method, "BY")

    def test_split_validation_checks_partition_closure_and_replacement(self):
        validation = _hand_table_index()
        with self.assertRaisesRegex(ValueError, "split count mismatch"):
            test_split_validation_candidates(
                [_candidate()], {RECORD: validation}, {RECORD: 1},
                _context(10), validation_fraction=0.5, local_pad=20,
                test_scope="global")

        with self.assertRaisesRegex(FormalInferenceError, "replacement"):
            test_split_validation_candidates(
                [_candidate()], {RECORD: validation}, {RECORD: 0},
                _context(10, replacement_ever=True, formal=False),
                validation_fraction=0.5, local_pad=20,
                test_scope="global")


class TestVersionedSchema(unittest.TestCase):

    def test_formal_and_exploratory_round_trip_with_common_coordinate_prefix(self):
        index = _hand_table_index()
        formal = evaluate_fixed_candidates(
            [_candidate("formal")], {RECORD: index}, _context(10),
            local_pad=50, test_scope="global")[0]
        exploratory = describe_de_novo_candidates(
            [_candidate("exploratory", source="de_novo:external")],
            {RECORD: index}, _context(10), local_pad=0,
            test_scope="both")[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trans_results.tsv")
            write_trans_loop_results(path, [formal, exploratory])
            with open(path) as handle:
                lines = handle.read().splitlines()
            header = lines[1].split("\t")
            self.assertEqual(tuple(header), TSV_COLUMNS)
            self.assertEqual(
                header[:8],
                ["loopId", "chrA", "startA", "endA", "chrB", "startB",
                 "endB", "distance(bp)"],
            )
            self.assertIn("\tNA\tNA\t", lines[3])
            parsed = read_trans_loop_results(path)

        self.assertEqual(parsed, [formal, exploratory])
        self.assertIsNone(parsed[1].bh_adjusted_p)
        self.assertIsNone(parsed[1].by_adjusted_p)
        self.assertIsNone(parsed[1].significant)


if __name__ == "__main__":
    unittest.main()
