#!/usr/bin/env python
# --coding:utf-8--
"""Strict-axis statistics for fixed inter-chromosomal candidates.

This module deliberately does not discover candidates.  Formal inference is
only exposed for rectangles declared independently of the X--Y pairings being
tested.  Data-driven (for example, de novo DBSCAN) candidates can use the
descriptive API, but their adjusted p-values and significance calls remain
unavailable by construction.

Two different library sizes are kept explicit:

``pair_pets``
    Physical rows in the concrete chromosome-pair index.  This is the
    population size for the directional conditional hypergeometric tests.
``library_depth``
    Validated logical whole-library depth from :class:`LibraryContext`.  It is
    used for library RPM only, never as the pair-level hypergeometric
    population size.
"""

import csv
import math
from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scipy.stats import hypergeom

from cLoops2.ds import TransContactIndex
from cLoops2.metadata import (CAP_FORMAL_INFERENCE,
                              CAP_GLOBAL_NORMALIZATION, CAP_PAIR_COUNTS,
                              LibraryContext, MetadataCapabilityError)


TRANS_STATS_SCHEMA_VERSION = 2
TRANS_STATS_MAGIC = "#cLoops2-trans-loop-result\t%s" % TRANS_STATS_SCHEMA_VERSION
CONTINUITY_CORRECTION = 0.5


class TransStatsError(ValueError):
    """Base error for invalid trans-statistics inputs or state."""


class FormalInferenceError(TransStatsError):
    """Raised when validated provenance cannot support formal inference."""


def _required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % label)
    if "\t" in value or "\n" in value or "\r" in value:
        raise ValueError("%s must not contain tab or newline characters" % label)
    return value.strip()


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("%s must be a non-negative integer" % label)
    value = int(value)
    if value < 0:
        raise ValueError("%s must be a non-negative integer" % label)
    return value


@dataclass(frozen=True)
class TransCandidate(object):
    """One pre-declared directional ``chromX x chromY`` rectangle.

    Calling :func:`test_fixed_candidates` asserts that ``candidate_source`` is
    independent of the current X--Y pairings.  The implementation can enforce
    a non-empty provenance declaration, but independence remains a scientific
    property for the caller to establish.
    """

    loop_id: str
    chrom_x: str
    x_start: int
    x_end: int
    chrom_y: str
    y_start: int
    y_end: int
    record_id: str
    candidate_source: str

    def __post_init__(self):
        object.__setattr__(self, "loop_id", _required_text(self.loop_id,
                                                           "loop_id"))
        object.__setattr__(self, "chrom_x", _required_text(self.chrom_x,
                                                            "chrom_x"))
        object.__setattr__(self, "chrom_y", _required_text(self.chrom_y,
                                                            "chrom_y"))
        object.__setattr__(self, "record_id", _required_text(
            self.record_id, "record_id"))
        object.__setattr__(self, "candidate_source", _required_text(
            self.candidate_source, "candidate_source"))
        for name in ("x_start", "x_end", "y_start", "y_end"):
            object.__setattr__(self, name,
                               _nonnegative_int(getattr(self, name), name))
        if self.x_start > self.x_end:
            raise ValueError("x_start must not exceed x_end")
        if self.y_start > self.y_end:
            raise ValueError("y_start must not exceed y_end")
        if self.chrom_x == self.chrom_y:
            raise ValueError("TransCandidate requires different chromosomes")

    @property
    def distance(self):
        """Trans candidates have no same-chromosome genomic distance."""
        return -1


@dataclass(frozen=True)
class TransLoopResult(object):
    """Version-2 result for one directional trans candidate."""

    loop_id: str
    chrom_x: str
    x_start: int
    x_end: int
    chrom_y: str
    y_start: int
    y_end: int
    record_id: str
    candidate_source: str
    inference_mode: str
    test_scope: str
    selection_adjusted: bool
    inferential_validity: bool
    pair_pets: int
    library_depth: Optional[int]
    reads_a: int
    reads_b: int
    pets: int
    expected_global: Optional[float]
    expected_local: Optional[float]
    global_enrichment: Optional[float]
    local_enrichment: Optional[float]
    odds_ratio_global: float
    odds_ratio_local: Optional[float]
    global_conditional_pvalue: float
    local_conditional_pvalue: Optional[float]
    primary_pvalue: float
    bh_adjusted_p: Optional[float]
    by_adjusted_p: Optional[float]
    adjustment_method: Optional[str]
    local_pad: int
    local_x_start: Optional[int]
    local_x_end: Optional[int]
    local_y_start: Optional[int]
    local_y_end: Optional[int]
    local_pair_pets: Optional[int]
    local_reads_a: Optional[int]
    local_reads_b: Optional[int]
    local_test_available: Optional[bool]
    local_status: str
    local_lower_clipped: bool
    library_rpm: Optional[float]
    alpha: Optional[float]
    significant: Optional[bool]

    @property
    def distance(self):
        return -1


@dataclass(frozen=True)
class _TableStats(object):
    population: int
    row_margin: int
    column_margin: int
    observed: int
    expected: Optional[float]
    enrichment: Optional[float]
    odds_ratio: float
    pvalue: float


@dataclass(frozen=True)
class _LocalStats(object):
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    lower_clipped: bool
    available: bool
    status: str
    table: _TableStats


def _table_stats(population, row_margin, column_margin, observed):
    """Return exact one-sided conditional statistics for a valid 2x2 table."""
    values = (population, row_margin, column_margin, observed)
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           for value in values):
        raise TypeError("2x2 table counts must be integers")
    population, row_margin, column_margin, observed = map(int, values)
    if min(values) < 0:
        raise ValueError("2x2 table counts must be non-negative")
    if row_margin > population or column_margin > population:
        raise ValueError("2x2 margins must not exceed the population")
    if observed > row_margin or observed > column_margin:
        raise ValueError("2x2 overlap must not exceed either margin")

    n10 = row_margin - observed
    n01 = column_margin - observed
    n00 = population - row_margin - column_margin + observed
    if n00 < 0:
        raise ValueError("2x2 table has a negative neither-cell count")

    if population == 0:
        expected = None
        enrichment = None
        pvalue = 1.0
    else:
        expected = float(row_margin) * float(column_margin) / population
        enrichment = ((observed + CONTINUITY_CORRECTION) /
                      (expected + CONTINUITY_CORRECTION))
        # sf(rab - 1) is the exact enrichment tail P[X >= rab].
        pvalue = float(
            hypergeom.sf(observed - 1, population, row_margin, column_margin))
        if not math.isfinite(pvalue):
            raise ArithmeticError("hypergeometric survival function is not finite")
        pvalue = min(1.0, max(0.0, pvalue))

    odds_ratio = ((observed + CONTINUITY_CORRECTION) *
                  (n00 + CONTINUITY_CORRECTION) /
                  ((n10 + CONTINUITY_CORRECTION) *
                   (n01 + CONTINUITY_CORRECTION)))
    return _TableStats(
        population=population,
        row_margin=row_margin,
        column_margin=column_margin,
        observed=observed,
        expected=expected,
        enrichment=enrichment,
        odds_ratio=float(odds_ratio),
        pvalue=pvalue,
    )


def _local_stats(index, candidate, local_pad, observed):
    raw_x_start = candidate.x_start - local_pad
    raw_y_start = candidate.y_start - local_pad
    x_start = max(0, raw_x_start)
    y_start = max(0, raw_y_start)
    x_end = candidate.x_end + local_pad
    y_end = candidate.y_end + local_pad
    clipped = raw_x_start < 0 or raw_y_start < 0

    n_local = index.count_rect(x_start, x_end, y_start, y_end)
    reads_a = index.count_rect(candidate.x_start, candidate.x_end,
                               y_start, y_end)
    reads_b = index.count_rect(x_start, x_end, candidate.y_start,
                               candidate.y_end)
    table = _table_stats(n_local, reads_a, reads_b, observed)

    if n_local == 0:
        available, status = False, "unavailable_empty_window"
    elif n_local - observed <= 0:
        available, status = False, "unavailable_no_background"
    elif reads_a in (0, n_local):
        available, status = False, "unavailable_degenerate_x_margin"
    elif reads_b in (0, n_local):
        available, status = False, "unavailable_degenerate_y_margin"
    else:
        available, status = True, "available"

    return _LocalStats(
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        lower_clipped=clipped,
        available=available,
        status=status,
        table=table,
    )


def _validate_common(candidates, indexes, context, local_pad, test_scope):
    if not isinstance(context, LibraryContext):
        raise TypeError("context must be a LibraryContext")
    if not isinstance(indexes, Mapping):
        raise TypeError("indexes must map record_id to TransContactIndex")
    local_pad = _nonnegative_int(local_pad, "local_pad")
    if test_scope not in ("both", "global"):
        raise ValueError("test_scope must be 'both' or 'global'")

    try:
        context.require(CAP_PAIR_COUNTS)
    except MetadataCapabilityError as exc:
        raise TransStatsError(str(exc))

    candidates = list(candidates)
    loop_ids = set()
    for candidate in candidates:
        if not isinstance(candidate, TransCandidate):
            raise TypeError("all candidates must be TransCandidate instances")
        if candidate.loop_id in loop_ids:
            raise ValueError("duplicate loop_id %r" % candidate.loop_id)
        loop_ids.add(candidate.loop_id)
        if candidate.record_id not in indexes:
            raise KeyError("missing TransContactIndex for record %r" %
                           candidate.record_id)
        index = indexes[candidate.record_id]
        if not isinstance(index, TransContactIndex):
            raise TypeError("index for %r must be TransContactIndex" %
                            candidate.record_id)
        manifest_n = context.get_npair(candidate.record_id)
        if manifest_n != index.number:
            raise TransStatsError(
                "pair count mismatch for %r: manifest=%s, index=%s" %
                (candidate.record_id, manifest_n, index.number))
    return candidates, local_pad


def _library_depth(context, formal):
    if CAP_GLOBAL_NORMALIZATION not in context.capabilities:
        if formal:
            try:
                context.require(CAP_GLOBAL_NORMALIZATION)
            except MetadataCapabilityError as exc:
                raise FormalInferenceError(str(exc))
        return None
    value = context.logical_total
    if value is None:
        if formal:
            raise FormalInferenceError(
                "formal inference requires validated logical library depth")
        return None
    return _nonnegative_int(value, "context.logical_total")


def _evaluate_one(candidate, index, library_depth, local_pad, test_scope,
                  inference_mode, inferential_validity):
    pair_pets = int(index.number)
    reads_a = index.count_x(candidate.x_start, candidate.x_end)
    reads_b = index.count_y(candidate.y_start, candidate.y_end)
    pets = index.count_rect(candidate.x_start, candidate.x_end,
                            candidate.y_start, candidate.y_end)
    global_table = _table_stats(pair_pets, reads_a, reads_b, pets)

    if test_scope == "global":
        local = None
        p_local = None
        p_primary = global_table.pvalue
        local_status = "not_requested_global_scope"
    else:
        local = _local_stats(index, candidate, local_pad, pets)
        if local.available:
            p_local = local.table.pvalue
            p_primary = max(global_table.pvalue, p_local)
        else:
            # The IUT null cannot silently change after observing an
            # unavailable local table.  A pre-selected global-only analysis is
            # represented by test_scope='global' instead.
            p_local = 1.0
            p_primary = 1.0
        local_status = local.status

    if library_depth in (None, 0):
        library_rpm = None
    else:
        library_rpm = float(pets) / library_depth * 10.0**6

    return TransLoopResult(
        loop_id=candidate.loop_id,
        chrom_x=candidate.chrom_x,
        x_start=candidate.x_start,
        x_end=candidate.x_end,
        chrom_y=candidate.chrom_y,
        y_start=candidate.y_start,
        y_end=candidate.y_end,
        record_id=candidate.record_id,
        candidate_source=candidate.candidate_source,
        inference_mode=inference_mode,
        test_scope=test_scope,
        selection_adjusted=False,
        inferential_validity=inferential_validity,
        pair_pets=pair_pets,
        library_depth=library_depth,
        reads_a=reads_a,
        reads_b=reads_b,
        pets=pets,
        expected_global=global_table.expected,
        expected_local=(None if local is None else local.table.expected),
        global_enrichment=global_table.enrichment,
        local_enrichment=(None if local is None else local.table.enrichment),
        odds_ratio_global=global_table.odds_ratio,
        odds_ratio_local=(None if local is None else local.table.odds_ratio),
        global_conditional_pvalue=global_table.pvalue,
        local_conditional_pvalue=p_local,
        primary_pvalue=p_primary,
        bh_adjusted_p=None,
        by_adjusted_p=None,
        adjustment_method=None,
        local_pad=local_pad,
        local_x_start=(None if local is None else local.x_start),
        local_x_end=(None if local is None else local.x_end),
        local_y_start=(None if local is None else local.y_start),
        local_y_end=(None if local is None else local.y_end),
        local_pair_pets=(None if local is None else local.table.population),
        local_reads_a=(None if local is None else local.table.row_margin),
        local_reads_b=(None if local is None else local.table.column_margin),
        local_test_available=(None if local is None else local.available),
        local_status=local_status,
        local_lower_clipped=(False if local is None else local.lower_clipped),
        library_rpm=library_rpm,
        alpha=None,
        significant=None,
    )


def adjust_pvalues(pvalues, method="BH"):
    """Adjust one complete p-value family by BH or BY.

    No candidate filtering occurs here.  Callers must pass the full,
    pre-declared analysis family, including candidates with zero observed PETs.
    """
    method = str(method).upper()
    if method not in ("BH", "BY"):
        raise ValueError("adjustment method must be BH or BY")
    pvalues = list(pvalues)
    for value in pvalues:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("p-values must be numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("p-values must be finite values in [0, 1]")
    size = len(pvalues)
    if size == 0:
        return []

    order = sorted(range(size), key=lambda i: (float(pvalues[i]), i))
    scale = (sum(1.0 / rank for rank in range(1, size + 1))
             if method == "BY" else 1.0)
    adjusted = [1.0] * size
    running = 1.0
    for position in range(size - 1, -1, -1):
        index = order[position]
        rank = position + 1
        value = float(pvalues[index]) * size * scale / rank
        running = min(running, value)
        adjusted[index] = min(1.0, max(0.0, running))
    return adjusted


def test_fixed_candidates(candidates, indexes, context, local_pad,
                          test_scope="both", adjustment="BH", alpha=0.05):
    """Formally test one complete family of independent fixed candidates.

    Replacement provenance is an unconditional error.  There is deliberately
    no override that emits formal p/q/significance values for replacement data.
    """
    candidates, local_pad = _validate_common(
        candidates, indexes, context, local_pad, test_scope)
    if context.replacement_ever is not False:
        raise FormalInferenceError(
            "formal fixed-candidate inference requires replacement_ever=false")
    try:
        context.require(CAP_FORMAL_INFERENCE)
    except MetadataCapabilityError as exc:
        raise FormalInferenceError(str(exc))
    library_depth = _library_depth(context, formal=True)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric")
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    adjustment = str(adjustment).upper()
    if adjustment not in ("BH", "BY"):
        raise ValueError("adjustment must be BH or BY")

    results = [
        _evaluate_one(candidate, indexes[candidate.record_id], library_depth,
                      local_pad, test_scope, "formal_fixed", True)
        for candidate in candidates
    ]
    # Both corrections are reported under unambiguous column names.  The
    # pre-selected adjustment controls the final significant flag.
    bh = adjust_pvalues([result.primary_pvalue for result in results], "BH")
    by = adjust_pvalues([result.primary_pvalue for result in results], "BY")
    selected = bh if adjustment == "BH" else by
    return [
        replace(result,
                bh_adjusted_p=bh[i],
                by_adjusted_p=by[i],
                adjustment_method=adjustment,
                alpha=alpha,
                significant=bool(selected[i] <= alpha))
        for i, result in enumerate(results)
    ]


def test_split_validation_candidates(candidates,
                                     validation_indexes,
                                     discovery_counts,
                                     context,
                                     validation_fraction,
                                     local_pad,
                                     test_scope="both",
                                     adjustment="BH",
                                     alpha=0.05):
    """Formally test discovery-selected rectangles on held-out PETs.

    Candidate geometry must have been determined exclusively from each
    record's discovery partition.  ``validation_indexes`` contains the
    disjoint held-out rows used for every count and conditional test, while
    ``discovery_counts`` records how many rows were available to candidate
    discovery.  For every represented record the two counts must close to the
    validated source manifest exactly::

        discovery rows + validation rows == source pair rows

    Conditional on the discovery data, the candidate family is fixed with
    respect to the validation pairings.  BH/BY is therefore applied once to
    the complete family of discovered rectangles.  A no-candidate family is a
    valid result and returns an empty list.

    ``libraryDepth`` in returned results is the deterministic effective
    validation depth, ``round(source logical depth * validation_fraction)``.
    It is used only for descriptive library RPM; pair-level tests always use
    concrete held-out row counts.
    """
    if not isinstance(context, LibraryContext):
        raise TypeError("context must be a LibraryContext")
    if (isinstance(validation_fraction, bool) or
            not isinstance(validation_fraction, (int, float))):
        raise TypeError("validation_fraction must be numeric")
    validation_fraction = float(validation_fraction)
    if (not math.isfinite(validation_fraction) or
            not 0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0, 1)")
    if not isinstance(validation_indexes, Mapping):
        raise TypeError(
            "validation_indexes must map record_id to TransContactIndex")
    if not isinstance(discovery_counts, Mapping):
        raise TypeError("discovery_counts must map record_id to row count")
    if set(validation_indexes) != set(discovery_counts):
        raise ValueError(
            "validation_indexes and discovery_counts must cover the same records")

    # The source provenance, rather than the derived validation context below,
    # decides whether formal inference is permitted.  Parent replacement can
    # never be repaired by subsequently splitting or downsampling rows.
    if context.replacement_ever is not False:
        raise FormalInferenceError(
            "formal split-validation inference requires replacement_ever=false")
    try:
        context.require(CAP_FORMAL_INFERENCE)
    except MetadataCapabilityError as exc:
        raise FormalInferenceError(str(exc))
    source_library_depth = _library_depth(context, formal=True)
    if source_library_depth is None:
        raise FormalInferenceError(
            "formal split-validation requires a logical library depth")

    validation_counts = {}
    for record_id in sorted(validation_indexes):
        index = validation_indexes[record_id]
        if not isinstance(index, TransContactIndex):
            raise TypeError("validation index for %r must be TransContactIndex" %
                            record_id)
        discovery_n = _nonnegative_int(discovery_counts[record_id],
                                       "discovery_counts[%s]" % record_id)
        source_n = context.get_npair(record_id)
        validation_n = int(index.number)
        if discovery_n + validation_n != source_n:
            raise TransStatsError(
                "split count mismatch for %r: discovery=%s, validation=%s, "
                "source=%s" %
                (record_id, discovery_n, validation_n, source_n))
        validation_counts[record_id] = validation_n

    # Use half-up rounding instead of Python's bankers rounding so the contract
    # is stable and easy to reproduce outside Python.  A non-empty logical
    # source retains at least one effective validation PET.
    validation_depth = int(
        math.floor(source_library_depth * validation_fraction + 0.5))
    if source_library_depth > 0:
        validation_depth = max(1, validation_depth)

    validation_context = replace(
        context,
        physical_total=int(sum(validation_counts.values())),
        logical_total=validation_depth,
        npair_by_record=MappingProxyType(dict(validation_counts)),
        emit="trans",
        source_kind="validation_split",
        replacement_this_step=False,
        normalization_scope="discovery_validation_split",
        reasons=tuple(context.reasons) + (
            "formal candidates discovered on a disjoint PET partition", ),
    )
    results = test_fixed_candidates(
        candidates,
        validation_indexes,
        validation_context,
        local_pad=local_pad,
        test_scope=test_scope,
        adjustment=adjustment,
        alpha=alpha,
    )
    return [
        replace(result,
                inference_mode="formal_split_validation",
                selection_adjusted=True,
                inferential_validity=True)
        for result in results
    ]


def describe_de_novo_candidates(candidates, indexes, context, local_pad,
                                test_scope="both"):
    """Describe data-driven candidates without making formal discoveries.

    Raw conditional tail values are retained as exploratory diagnostics.
    Adjusted p-values and ``significant`` are always ``None``/``NA``.
    Replacement data are allowed here, but never gain inferential validity.
    """
    candidates, local_pad = _validate_common(
        candidates, indexes, context, local_pad, test_scope)
    library_depth = _library_depth(context, formal=False)
    results = [
        _evaluate_one(candidate, indexes[candidate.record_id], library_depth,
                      local_pad, test_scope, "exploratory_de_novo", False)
        for candidate in candidates
    ]
    return mark_de_novo_exploratory(results)


def mark_de_novo_exploratory(results):
    """Clear inferential fields on externally produced de novo results."""
    marked = []
    for result in results:
        if not isinstance(result, TransLoopResult):
            raise TypeError("results must contain TransLoopResult instances")
        marked.append(
            replace(result,
                    inference_mode="exploratory_de_novo",
                    selection_adjusted=False,
                    inferential_validity=False,
                    bh_adjusted_p=None,
                    by_adjusted_p=None,
                    adjustment_method=None,
                    alpha=None,
                    significant=None))
    return marked


TSV_COLUMNS = (
    "loopId", "chrA", "startA", "endA", "chrB", "startB", "endB",
    "distance(bp)", "schemaVersion", "recordId", "candidateSource",
    "inferenceMode", "testScope", "selectionAdjusted",
    "inferentialValidity", "pairPETs", "libraryDepth", "readsA", "readsB",
    "PETs", "expectedGlobal", "expectedLocal", "globalEnrichment",
    "localEnrichment", "oddsRatioGlobal", "oddsRatioLocal",
    "globalConditionalPvalue", "localConditionalPvalue", "primaryPvalue",
    "bhAdjustedP", "byAdjustedP", "adjustmentMethod", "localPad",
    "localXStart", "localXEnd", "localYStart", "localYEnd", "localPairPETs",
    "localReadsA", "localReadsB", "localTestAvailable", "localStatus",
    "localLowerClipped", "libraryRPM", "alpha", "significant",
)


def _format_optional(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _result_row(result):
    return (
        result.loop_id, result.chrom_x, result.x_start, result.x_end,
        result.chrom_y, result.y_start, result.y_end, result.distance,
        TRANS_STATS_SCHEMA_VERSION, result.record_id, result.candidate_source,
        result.inference_mode, result.test_scope, result.selection_adjusted,
        result.inferential_validity, result.pair_pets, result.library_depth,
        result.reads_a, result.reads_b, result.pets, result.expected_global,
        result.expected_local, result.global_enrichment,
        result.local_enrichment, result.odds_ratio_global,
        result.odds_ratio_local, result.global_conditional_pvalue,
        result.local_conditional_pvalue, result.primary_pvalue,
        result.bh_adjusted_p, result.by_adjusted_p, result.adjustment_method,
        result.local_pad, result.local_x_start, result.local_x_end,
        result.local_y_start, result.local_y_end, result.local_pair_pets,
        result.local_reads_a, result.local_reads_b,
        result.local_test_available, result.local_status,
        result.local_lower_clipped, result.library_rpm, result.alpha,
        result.significant,
    )


def write_trans_loop_results(path, results):
    """Write versioned TSV while preserving the common first eight columns."""
    results = list(results)
    for result in results:
        if not isinstance(result, TransLoopResult):
            raise TypeError("results must contain TransLoopResult instances")
    with open(path, "w", newline="") as handle:
        handle.write(TRANS_STATS_MAGIC + "\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for result in results:
            writer.writerow([_format_optional(value)
                             for value in _result_row(result)])


def _parse_optional(value, parser):
    if value == "NA":
        return None
    return parser(value)


def _parse_bool(value):
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("invalid boolean value %r" % value)


def _result_from_row(row):
    data = dict(zip(TSV_COLUMNS, row))
    version = int(data["schemaVersion"])
    if version != TRANS_STATS_SCHEMA_VERSION:
        raise ValueError("unsupported trans result schema version %s" % version)
    distance = int(data["distance(bp)"])
    if distance != -1:
        raise ValueError("trans result distance(bp) must be -1")
    return TransLoopResult(
        loop_id=data["loopId"],
        chrom_x=data["chrA"],
        x_start=int(data["startA"]),
        x_end=int(data["endA"]),
        chrom_y=data["chrB"],
        y_start=int(data["startB"]),
        y_end=int(data["endB"]),
        record_id=data["recordId"],
        candidate_source=data["candidateSource"],
        inference_mode=data["inferenceMode"],
        test_scope=data["testScope"],
        selection_adjusted=_parse_bool(data["selectionAdjusted"]),
        inferential_validity=_parse_bool(data["inferentialValidity"]),
        pair_pets=int(data["pairPETs"]),
        library_depth=_parse_optional(data["libraryDepth"], int),
        reads_a=int(data["readsA"]),
        reads_b=int(data["readsB"]),
        pets=int(data["PETs"]),
        expected_global=_parse_optional(data["expectedGlobal"], float),
        expected_local=_parse_optional(data["expectedLocal"], float),
        global_enrichment=_parse_optional(data["globalEnrichment"], float),
        local_enrichment=_parse_optional(data["localEnrichment"], float),
        odds_ratio_global=float(data["oddsRatioGlobal"]),
        odds_ratio_local=_parse_optional(data["oddsRatioLocal"], float),
        global_conditional_pvalue=float(data["globalConditionalPvalue"]),
        local_conditional_pvalue=_parse_optional(
            data["localConditionalPvalue"], float),
        primary_pvalue=float(data["primaryPvalue"]),
        bh_adjusted_p=_parse_optional(data["bhAdjustedP"], float),
        by_adjusted_p=_parse_optional(data["byAdjustedP"], float),
        adjustment_method=_parse_optional(data["adjustmentMethod"], str),
        local_pad=int(data["localPad"]),
        local_x_start=_parse_optional(data["localXStart"], int),
        local_x_end=_parse_optional(data["localXEnd"], int),
        local_y_start=_parse_optional(data["localYStart"], int),
        local_y_end=_parse_optional(data["localYEnd"], int),
        local_pair_pets=_parse_optional(data["localPairPETs"], int),
        local_reads_a=_parse_optional(data["localReadsA"], int),
        local_reads_b=_parse_optional(data["localReadsB"], int),
        local_test_available=_parse_optional(data["localTestAvailable"],
                                             _parse_bool),
        local_status=data["localStatus"],
        local_lower_clipped=_parse_bool(data["localLowerClipped"]),
        library_rpm=_parse_optional(data["libraryRPM"], float),
        alpha=_parse_optional(data["alpha"], float),
        significant=_parse_optional(data["significant"], _parse_bool),
    )


def read_trans_loop_results(path):
    """Read and validate a version-2 trans-loop-result TSV."""
    with open(path, "r", newline="") as handle:
        magic = handle.readline().rstrip("\n\r")
        if magic != TRANS_STATS_MAGIC:
            raise ValueError("missing or unsupported trans result schema marker")
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise ValueError("trans result TSV is missing its header")
        if header != TSV_COLUMNS:
            raise ValueError("trans result TSV header does not match schema v%s" %
                             TRANS_STATS_SCHEMA_VERSION)
        results = []
        for line_number, row in enumerate(reader, start=3):
            if not row:
                continue
            if len(row) != len(TSV_COLUMNS):
                raise ValueError("line %s has %s columns, expected %s" %
                                 (line_number, len(row), len(TSV_COLUMNS)))
            results.append(_result_from_row(row))
    return results


# Descriptive aliases make the two public tracks explicit without coupling the
# module to the existing CLI during Phase 1.
evaluate_fixed_candidates = test_fixed_candidates
evaluate_de_novo_candidates = describe_de_novo_candidates
evaluate_split_validation_candidates = test_split_validation_candidates


__all__ = [
    "CONTINUITY_CORRECTION", "FormalInferenceError", "TransCandidate",
    "TransLoopResult", "TransStatsError", "TRANS_STATS_SCHEMA_VERSION",
    "TSV_COLUMNS", "adjust_pvalues", "describe_de_novo_candidates",
    "evaluate_de_novo_candidates", "evaluate_fixed_candidates",
    "evaluate_split_validation_candidates",
    "mark_de_novo_exploratory", "read_trans_loop_results",
    "test_fixed_candidates", "test_split_validation_candidates",
    "write_trans_loop_results",
]
