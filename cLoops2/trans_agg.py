#!/usr/bin/env python
# --coding:utf-8--
"""Directional aggregate, viewpoint, and montage analysis for trans PETs.

The historical functions in :mod:`cLoops2.agg` and :mod:`cLoops2.montage`
assume that both PET ends lie on one interchangeable genomic axis.  This
module deliberately provides separate trans APIs instead of adding trans
branches to those symmetric implementations.

Three denominators remain distinct throughout:

``Npair``
    Physical PET rows in one directional ``chromX x chromY`` record.  It is
    the denominator for pair-global expected matrices/profiles.
``Lglobal``
    Validated logical whole-library depth from ``petMeta.json``.  It is used
    only for library-RPM scaling of observed counts.
``Unique PETs``
    Physical rows materialized in the current directory.  It is validated by
    :func:`cLoops2.metadata.build_library_context` but is never substituted
    for either of the quantities above.

All coordinate intervals are closed, matching cLoops2 loop/query semantics.
No function mirrors or transposes observations to fill a symmetric matrix.
"""

import csv
import json
import os
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional, Tuple, cast

import joblib
import numpy as np

from cLoops2.cmat import getTransObsMat
from cLoops2.metadata import (CAP_GLOBAL_NORMALIZATION, CAP_PAIR_COUNTS,
                              LibraryContext, build_library_context,
                              inspect_actual_counts)


TRANS_AGG_SCHEMA_VERSION = 1
_METHODS = ("obs", "pair_oe", "window_oe")
_NORMALIZATIONS = ("raw", "library_rpm")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % label)
    value = value.strip()
    if any(character in value for character in ("\t", "\n", "\r")):
        raise ValueError("%s must not contain tab or newline characters" %
                         label)
    return value


def _coordinate(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("%s must be a non-negative integer" % label)
    value = int(value)
    if value < 0:
        raise ValueError("%s must be a non-negative integer" % label)
    return value


def _positive_int(value, label):
    value = _coordinate(value, label)
    if value == 0:
        raise ValueError("%s must be a positive integer" % label)
    return value


def _validate_method_and_normalization(method, normalization):
    if method not in _METHODS:
        raise ValueError("method must be one of: %s" % ", ".join(_METHODS))
    if normalization not in _NORMALIZATIONS:
        raise ValueError("normalization must be raw or library_rpm")
    if method != "obs" and normalization != "raw":
        raise ValueError(
            "pair/window O/E is dimensionless and cannot be library-RPM "
            "normalized")


@dataclass(frozen=True)
class TransFeature(object):
    """One directional trans rectangle used for aggregate analysis."""

    feature_id: str
    chrom_x: str
    x_start: int
    x_end: int
    chrom_y: str
    y_start: int
    y_end: int
    record_id: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "feature_id",
                           _text(self.feature_id, "feature_id"))
        object.__setattr__(self, "chrom_x", _text(self.chrom_x, "chrom_x"))
        object.__setattr__(self, "chrom_y", _text(self.chrom_y, "chrom_y"))
        for name in ("x_start", "x_end", "y_start", "y_end"):
            object.__setattr__(self, name,
                               _coordinate(getattr(self, name), name))
        if self.x_start > self.x_end:
            raise ValueError("x_start must not exceed x_end")
        if self.y_start > self.y_end:
            raise ValueError("y_start must not exceed y_end")
        if self.chrom_x == self.chrom_y:
            raise ValueError("TransFeature requires different chromosomes")
        if self.record_id is not None:
            object.__setattr__(self, "record_id",
                               _text(self.record_id, "record_id"))


@dataclass(frozen=True)
class AxisRegion(object):
    """One named interval on exactly one montage axis."""

    region_id: str
    chrom: str
    start: int
    end: int

    def __post_init__(self):
        object.__setattr__(self, "region_id",
                           _text(self.region_id, "region_id"))
        object.__setattr__(self, "chrom", _text(self.chrom, "chrom"))
        object.__setattr__(self, "start", _coordinate(self.start, "start"))
        object.__setattr__(self, "end", _coordinate(self.end, "end"))
        if self.start > self.end:
            raise ValueError("region start must not exceed region end")


@dataclass(frozen=True)
class TransAggregateResult(object):
    """Equal-feature aggregate of directional rectangular matrices."""

    matrix: np.ndarray
    valid_counts: np.ndarray
    raw_sum: np.ndarray
    raw_mean: np.ndarray
    expected_mean: Optional[np.ndarray]
    method: str
    normalization: str
    logical_total: Optional[int]
    n_input: int
    n_used: int
    used_feature_ids: Tuple[str, ...]
    used_record_ids: Tuple[str, ...]
    used_features: Tuple[TransFeature, ...]
    skipped_feature_ids: Tuple[str, ...]
    npair_by_record: Tuple[Tuple[str, int], ...]
    normalization_scope: str
    x_bin_size: int
    y_bin_size: int
    x_flank_bins: int
    y_flank_bins: int

    @property
    def x_profile(self):
        """Selected-scale profile along chromX (sum over chromY bins)."""
        profile = np.nansum(self.matrix, axis=1)
        profile[np.sum(self.valid_mask, axis=1) == 0] = np.nan
        return profile

    @property
    def y_profile(self):
        """Selected-scale profile along chromY (sum over chromX bins)."""
        profile = np.nansum(self.matrix, axis=0)
        profile[np.sum(self.valid_mask, axis=0) == 0] = np.nan
        return profile

    @property
    def raw_x_profile(self):
        """Mean raw chromX endpoint profile for the retained features."""
        return self.raw_mean.sum(axis=1)

    @property
    def raw_y_profile(self):
        """Mean raw chromY endpoint profile for the retained features."""
        return self.raw_mean.sum(axis=0)

    @property
    def valid_mask(self):
        """Cells supported by at least one defined feature-level value."""
        return self.valid_counts > 0


@dataclass(frozen=True)
class TransViewpointResult(object):
    """Directional 1D profile from one anchor axis to the opposite axis."""

    values: np.ndarray
    observed: np.ndarray
    expected: Optional[np.ndarray]
    target_bin_starts: np.ndarray
    target_bin_ends: np.ndarray
    chrom_x: str
    chrom_y: str
    record_id: str
    anchor_axis: str
    anchor_start: int
    anchor_end: int
    target_start: int
    target_end: int
    target_bin_size: int
    context_start: Optional[int]
    context_end: Optional[int]
    method: str
    normalization: str
    n_pair: int
    logical_total: Optional[int]
    normalization_scope: str

    @property
    def anchor_chrom(self):
        return self.chrom_x if self.anchor_axis == "x" else self.chrom_y

    @property
    def target_chrom(self):
        return self.chrom_y if self.anchor_axis == "x" else self.chrom_x

    @property
    def valid_mask(self):
        """Bins with a defined observed/RPM or O/E value."""
        return np.isfinite(self.values)


@dataclass(frozen=True)
class TransMontageResult(object):
    """Directional region-by-region matrix for one chromosome pair."""

    values: np.ndarray
    observed: np.ndarray
    expected: Optional[np.ndarray]
    x_regions: Tuple[AxisRegion, ...]
    y_regions: Tuple[AxisRegion, ...]
    x_region_ids: Tuple[str, ...]
    y_region_ids: Tuple[str, ...]
    chrom_x: str
    chrom_y: str
    record_id: str
    method: str
    normalization: str
    n_pair: int
    logical_total: Optional[int]
    normalization_scope: str

    @property
    def valid_mask(self):
        """Cells with a defined observed/RPM or O/E value."""
        return np.isfinite(self.values)


@dataclass(frozen=True)
class _TransRecord(object):
    key: str
    record_id: str
    chrom_x: str
    chrom_y: str
    path: Optional[str]
    n_pair: int


@dataclass(frozen=True)
class _TransDataset(object):
    meta: dict
    context: LibraryContext
    by_pair: dict
    by_record_id: dict


def _load_ixy(path):
    if path is None:
        return np.empty((0, 2), dtype=np.int64)
    try:
        matrix = joblib.load(path, mmap_mode="r")
    except Exception:
        matrix = joblib.load(path)
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or
            matrix.shape[1] != 2):
        raise ValueError("trans ixy %s must have shape (n, 2)" % path)
    if matrix.size and not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError("trans ixy %s must contain integer coordinates" % path)
    return matrix


def _metadata_axes(key, entry):
    chrom_x, chrom_y = entry.get("chromX"), entry.get("chromY")
    if chrom_x is None or chrom_y is None:
        axes = key.split("-")
        if len(axes) != 2 or not axes[0] or not axes[1]:
            raise ValueError(
                "ambiguous trans key %r; chromX/chromY metadata is required" %
                key)
        chrom_x, chrom_y = axes
    chrom_x, chrom_y = str(chrom_x), str(chrom_y)
    if chrom_x == chrom_y:
        raise ValueError("trans metadata record %r has identical axes" % key)
    return chrom_x, chrom_y


def _load_dataset(predir):
    metaf = os.path.join(str(predir), "petMeta.json")
    with open(metaf) as handle:
        meta = json.load(handle)
    actual_counts = inspect_actual_counts(meta)
    context = build_library_context(meta, actual_counts=actual_counts)
    if context.validity == "invalid":
        raise ValueError("invalid PET metadata: %s" %
                         ("; ".join(context.reasons) or "unknown reason"))
    context.require(CAP_PAIR_COUNTS)

    by_pair = {}
    by_record_id = {}
    entries = meta.get("data", {}).get("trans", {})
    if not isinstance(entries, dict):
        raise ValueError("petMeta data.trans must be an object")
    for key in sorted(entries):
        entry = entries[key]
        if not isinstance(entry, dict) or not entry.get("ixy"):
            raise ValueError("metadata trans record %r has no ixy path" % key)
        chrom_x, chrom_y = _metadata_axes(key, entry)
        record_id = str(entry.get("record_id", "trans:%s" % key))
        if record_id not in actual_counts:
            raise ValueError("metadata trans record %r was not counted" % key)
        record = _TransRecord(
            key=str(key), record_id=record_id, chrom_x=chrom_x,
            chrom_y=chrom_y, path=str(entry["ixy"]),
            n_pair=int(actual_counts[record_id]))
        pair = (chrom_x, chrom_y)
        if pair in by_pair:
            raise ValueError("duplicate trans chromosome pair %s,%s" % pair)
        if record_id in by_record_id:
            raise ValueError("duplicate trans record_id %r" % record_id)
        by_pair[pair] = record
        by_record_id[record_id] = record

    # A globally sampled trans/all projection may legitimately allocate zero
    # rows to a retained pair and omit its physical .ixy file.  Preserve that
    # pair as an explicit zero record rather than confusing it with an
    # upstream run that never retained trans PETs.
    sampling = meta.get("Sampling")
    if (isinstance(sampling, dict) and
            sampling.get("emit", sampling.get("mode")) in ("trans", "all")):
        for item in sampling.get("records", []):
            if (not isinstance(item, dict) or
                    item.get("category") != "trans" or
                    int(item.get("selected_rows", -1)) != 0 or
                    int(item.get("written_rows", -1)) != 0):
                continue
            record_id = str(item.get("record_id", ""))
            chrom_x, chrom_y = item.get("chromX"), item.get("chromY")
            if not record_id or chrom_x is None or chrom_y is None:
                raise ValueError("invalid zero-row trans Sampling record")
            pair = (str(chrom_x), str(chrom_y))
            if pair in by_pair:
                continue
            if record_id in by_record_id:
                raise ValueError("duplicate trans record_id %r" % record_id)
            record = _TransRecord(
                key=str(item.get("key", record_id.split(":", 1)[-1])),
                record_id=record_id, chrom_x=pair[0], chrom_y=pair[1],
                path=None, n_pair=0)
            by_pair[pair] = record
            by_record_id[record_id] = record

    if not by_pair:
        raise ValueError(
            "no retained trans PET records; rerun cLoops2 pre -trans")
    return _TransDataset(meta, context, by_pair, by_record_id)


def _coerce_feature(value):
    if isinstance(value, TransFeature):
        return value
    names = {
        "feature_id": ("feature_id", "loop_id", "id"),
        "chrom_x": ("chrom_x", "chromX"),
        "x_start": ("x_start",),
        "x_end": ("x_end",),
        "chrom_y": ("chrom_y", "chromY"),
        "y_start": ("y_start",),
        "y_end": ("y_end",),
        "record_id": ("record_id",),
    }
    values = {}
    for output_name, alternatives in names.items():
        found = None
        for name in alternatives:
            if hasattr(value, name):
                found = getattr(value, name)
                break
            if isinstance(value, dict) and name in value:
                found = value[name]
                break
        if output_name != "record_id" and found is None:
            raise TypeError("feature is missing %s" % output_name)
        values[output_name] = found
    return TransFeature(**values)


def read_trans_features(path):
    """Read cLoops2 trans result/candidate TSV as :class:`TransFeature`.

    Version-2 result files retain their structured ``recordId``.  Generic
    eight-column loop files are parsed with the public axis-aware parser.
    """
    with open(path) as handle:
        first = handle.readline().rstrip("\n\r")
    if first.startswith("#cLoops2-trans-loop-result"):
        from cLoops2.trans_stats import read_trans_loop_results
        return tuple(_coerce_feature(value)
                     for value in read_trans_loop_results(path))

    from cLoops2.io import parseTxt2Loops
    loops = parseTxt2Loops(path, mode="trans")
    return tuple(_coerce_feature(loop)
                 for key in sorted(loops) for loop in loops[key])


def read_axis_regions(path, expected_chrom=None, bed_half_open=True):
    """Read BED-like intervals for one montage axis.

    Canonical BED input is half-open and is converted to cLoops2's closed
    interval convention when ``bed_half_open=True`` (the default).  Set it to
    false only for an input table whose end coordinate is already inclusive.
    """
    regions = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n\r").split("\t")
            if len(fields) < 3:
                raise ValueError("line %s has fewer than three columns" %
                                 line_number)
            region_id = (fields[3] if len(fields) > 3 and fields[3]
                         else "region_%s" % line_number)
            start, end = int(fields[1]), int(fields[2])
            if bed_half_open:
                if end <= start:
                    raise ValueError(
                        "line %s is not a positive-width BED interval" %
                        line_number)
                end -= 1
            region = AxisRegion(region_id, fields[0], start, end)
            if expected_chrom is not None and region.chrom != expected_chrom:
                raise ValueError("region %s is on %s, expected %s" %
                                 (region.region_id, region.chrom,
                                  expected_chrom))
            regions.append(region)
    if not regions:
        raise ValueError("region file contains no intervals")
    return tuple(regions)


def _resolve_record(dataset, chrom_x, chrom_y, record_id=None):
    pair = (_text(chrom_x, "chrom_x"), _text(chrom_y, "chrom_y"))
    if pair[0] == pair[1]:
        raise ValueError("trans analysis requires different chromosomes")
    if record_id is not None:
        record_id = _text(record_id, "record_id")
        try:
            record = dataset.by_record_id[record_id]
        except KeyError:
            raise KeyError("unknown trans record_id %r" % record_id)
        if (record.chrom_x, record.chrom_y) != pair:
            raise ValueError(
                "record %r has axes %s x %s, not %s x %s" %
                (record_id, record.chrom_x, record.chrom_y, pair[0], pair[1]))
        return record
    try:
        return dataset.by_pair[pair]
    except KeyError:
        reverse = (pair[1], pair[0])
        if reverse in dataset.by_pair:
            raise ValueError(
                "requested axes are reversed relative to metadata; swap both "
                "chromosomes and their coordinates explicitly")
        raise KeyError("no retained trans record for %s x %s" % pair)


def _centered_bounds(start, end, bin_size, flank_bins):
    center = (int(start) + int(end)) // 2
    n_bins = 2 * int(flank_bins) + 1
    width = n_bins * int(bin_size)
    window_start = center - width // 2
    return window_start, window_start + width - 1


def _axis_histogram(values, start, end, bin_size):
    n_bins = (int(end) - int(start)) // int(bin_size) + 1
    values = np.asarray(values)
    keep = (values >= start) & (values <= end)
    if not np.any(keep):
        return np.zeros(n_bins, dtype=np.int64)
    indexes = ((values[keep] - start) // bin_size).astype(np.int64)
    return np.bincount(indexes, minlength=n_bins).astype(np.int64,
                                                          copy=False)


def _expected_and_values(xy, observed, x_start, x_end, y_start, y_end,
                         x_bin_size, y_bin_size, method):
    if method == "obs":
        return None, observed.astype(float, copy=False)
    if method == "pair_oe":
        n_pair = int(xy.shape[0])
        x_margin = _axis_histogram(xy[:, 0], x_start, x_end, x_bin_size)
        y_margin = _axis_histogram(xy[:, 1], y_start, y_end, y_bin_size)
        if n_pair == 0:
            expected = np.zeros(observed.shape, dtype=float)
        else:
            expected = (np.outer(x_margin, y_margin).astype(float) /
                        float(n_pair))
    else:
        n_rect = int(observed.sum())
        if n_rect == 0:
            expected = np.zeros(observed.shape, dtype=float)
        else:
            expected = (np.outer(observed.sum(axis=1),
                                 observed.sum(axis=0)).astype(float) /
                        float(n_rect))
    # A zero expected count makes O/E undefined (0/0), not zero enrichment.
    # Preserve that distinction for downstream plotting/aggregation.
    values = np.full(observed.shape, np.nan, dtype=float)
    np.divide(observed, expected, out=values, where=expected > 0)
    return expected, values


def _logical_total(dataset, normalization):
    if normalization != "library_rpm":
        if CAP_GLOBAL_NORMALIZATION in dataset.context.capabilities:
            return dataset.context.logical_total
        return None
    dataset.context.require(CAP_GLOBAL_NORMALIZATION)
    if dataset.context.logical_total is None:
        raise RuntimeError(
            "global-normalization capability has no logical total")
    logical_total = int(dataset.context.logical_total)
    if logical_total <= 0:
        raise ValueError("library_rpm requires a positive Lglobal")
    return logical_total


def aggregate_trans_loops(predir,
                          features,
                          x_bin_size,
                          y_bin_size=None,
                          x_flank_bins=10,
                          y_flank_bins=10,
                          method="obs",
                          normalization="raw",
                          skip_zeros=False,
                          edge_policy="skip",
                          max_dense_cells=10000000):
    """Aggregate fixed-size directional windows around trans features.

    Every feature contributes one matrix with
    ``(2*x_flank_bins+1, 2*y_flank_bins+1)`` cells and therefore receives
    equal weight.  The X and Y resolutions/flanks are independent.  Features
    whose centered window would start below coordinate zero are either
    reported in ``skipped_feature_ids`` or rejected according to
    ``edge_policy``.
    """
    _validate_method_and_normalization(method, normalization)
    x_bin_size = _positive_int(x_bin_size, "x_bin_size")
    if y_bin_size is None:
        y_bin_size = x_bin_size
    y_bin_size = _positive_int(y_bin_size, "y_bin_size")
    x_flank_bins = _coordinate(x_flank_bins, "x_flank_bins")
    y_flank_bins = _coordinate(y_flank_bins, "y_flank_bins")
    max_dense_cells = _positive_int(max_dense_cells, "max_dense_cells")
    if edge_policy not in ("skip", "error"):
        raise ValueError("edge_policy must be skip or error")
    features = tuple(_coerce_feature(feature) for feature in features)
    if not features:
        raise ValueError("no trans features were supplied")
    feature_ids = [feature.feature_id for feature in features]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("trans feature ids must be unique")

    dataset = _load_dataset(predir)
    logical_total = _logical_total(dataset, normalization)
    grouped = {}
    for index, feature in enumerate(features):
        record = _resolve_record(dataset, feature.chrom_x, feature.chrom_y,
                                 feature.record_id)
        grouped.setdefault(record.record_id, (record, []))[1].append(
            (index, feature))

    shape = (2 * x_flank_bins + 1, 2 * y_flank_bins + 1)
    if shape[0] * shape[1] > max_dense_cells:
        raise ValueError("aggregate matrix exceeds max_dense_cells")
    raw_sum = np.zeros(shape, dtype=np.int64)
    value_sum = np.zeros(shape, dtype=float)
    valid_counts = np.zeros(shape, dtype=np.int64)
    expected_sum = np.zeros(shape, dtype=float)
    used_indexes = set()
    record_id_by_index = {}
    for record_id in sorted(grouped):
        record, record_features = grouped[record_id]
        xy = _load_ixy(record.path)
        if int(xy.shape[0]) != record.n_pair:
            raise ValueError("trans record %r changed after metadata validation" %
                             record_id)
        for index, feature in record_features:
            x_start, x_end = _centered_bounds(
                feature.x_start, feature.x_end, x_bin_size, x_flank_bins)
            y_start, y_end = _centered_bounds(
                feature.y_start, feature.y_end, y_bin_size, y_flank_bins)
            if x_start < 0 or y_start < 0:
                if edge_policy == "error":
                    raise ValueError(
                        "feature %r has a centered window below coordinate zero" %
                        feature.feature_id)
                continue
            observed = getTransObsMat(
                xy, x_start, x_end, y_start, y_end, x_bin_size, y_bin_size,
                max_dense_cells=max_dense_cells)
            if skip_zeros and not np.any(observed):
                continue
            expected, values = _expected_and_values(
                xy, observed, x_start, x_end, y_start, y_end, x_bin_size,
                y_bin_size, method)
            if normalization == "library_rpm":
                if logical_total is None:
                    raise RuntimeError("library_rpm has no logical total")
                values = values / float(logical_total) * 1000000.0
            raw_sum += observed
            valid = np.isfinite(values)
            value_sum[valid] += values[valid]
            valid_counts[valid] += 1
            if expected is not None:
                expected_sum += expected
            used_indexes.add(index)
            record_id_by_index[index] = record.record_id
        del xy

    used_indexes = [index for index in range(len(features))
                    if index in used_indexes]
    if not used_indexes:
        raise ValueError("no trans features remained after window filtering")
    n_used = len(used_indexes)
    expected_mean = (expected_sum / float(n_used)
                     if method != "obs" else None)
    aggregate = np.full(shape, np.nan, dtype=float)
    np.divide(value_sum, valid_counts, out=aggregate,
              where=valid_counts > 0)
    return TransAggregateResult(
        matrix=aggregate,
        valid_counts=valid_counts,
        raw_sum=raw_sum,
        raw_mean=raw_sum.astype(float) / float(n_used),
        expected_mean=expected_mean,
        method=method,
        normalization=normalization,
        logical_total=logical_total,
        n_input=len(features),
        n_used=n_used,
        used_feature_ids=tuple(features[index].feature_id
                               for index in used_indexes),
        used_record_ids=tuple(record_id_by_index[index]
                              for index in used_indexes),
        used_features=tuple(features[index] for index in used_indexes),
        skipped_feature_ids=tuple(features[index].feature_id
                                  for index in range(len(features))
                                  if index not in used_indexes),
        npair_by_record=tuple(
            (record_id, int(grouped[record_id][0].n_pair))
            for record_id in sorted(grouped)),
        normalization_scope=dataset.context.normalization_scope,
        x_bin_size=x_bin_size,
        y_bin_size=y_bin_size,
        x_flank_bins=x_flank_bins,
        y_flank_bins=y_flank_bins,
    )


def trans_viewpoint(predir,
                    chrom_x,
                    chrom_y,
                    anchor_axis,
                    anchor_start,
                    anchor_end,
                    target_start,
                    target_end,
                    target_bin_size,
                    method="obs",
                    normalization="raw",
                    context_start=None,
                    context_end=None,
                    record_id=None,
                    max_bins=1000000):
    """Profile opposite-axis endpoints contacted by one trans viewpoint.

    ``pair_oe`` compares the anchor-to-target-bin count with endpoint
    marginals from the complete chromosome pair.  ``window_oe`` conditions on
    ``anchor-axis context x target window`` and therefore requires an explicit
    context interval containing the anchor.  The latter avoids the degenerate
    definition obtained by treating the anchor itself as the entire window.
    """
    _validate_method_and_normalization(method, normalization)
    if anchor_axis not in ("x", "y"):
        raise ValueError("anchor_axis must be x or y")
    anchor_start = _coordinate(anchor_start, "anchor_start")
    anchor_end = _coordinate(anchor_end, "anchor_end")
    target_start = _coordinate(target_start, "target_start")
    target_end = _coordinate(target_end, "target_end")
    target_bin_size = _positive_int(target_bin_size, "target_bin_size")
    max_bins = _positive_int(max_bins, "max_bins")
    if anchor_start > anchor_end:
        raise ValueError("anchor_start must not exceed anchor_end")
    if target_start > target_end:
        raise ValueError("target_start must not exceed target_end")
    if method == "window_oe":
        if context_start is None or context_end is None:
            raise ValueError(
                "window_oe viewpoint requires context_start and context_end")
        context_start = _coordinate(context_start, "context_start")
        context_end = _coordinate(context_end, "context_end")
        if context_start > anchor_start or context_end < anchor_end:
            raise ValueError("viewpoint context must contain the anchor")
        if context_start == anchor_start and context_end == anchor_end:
            raise ValueError(
                "window_oe viewpoint context must strictly extend the anchor")
    elif context_start is not None or context_end is not None:
        raise ValueError("context bounds are only defined for window_oe")
    n_bins = (target_end - target_start) // target_bin_size + 1
    if n_bins > max_bins:
        raise ValueError("viewpoint profile exceeds max_bins")

    dataset = _load_dataset(predir)
    record = _resolve_record(dataset, chrom_x, chrom_y, record_id)
    logical_total = _logical_total(dataset, normalization)
    xy = _load_ixy(record.path)
    if int(xy.shape[0]) != record.n_pair:
        raise ValueError("trans record changed after metadata validation")
    anchor_values = xy[:, 0] if anchor_axis == "x" else xy[:, 1]
    target_values = xy[:, 1] if anchor_axis == "x" else xy[:, 0]
    anchor_mask = ((anchor_values >= anchor_start) &
                   (anchor_values <= anchor_end))
    observed = _axis_histogram(target_values[anchor_mask], target_start,
                               target_end, target_bin_size)

    expected = None
    if method == "pair_oe":
        target_margin = _axis_histogram(target_values, target_start,
                                        target_end, target_bin_size)
        if record.n_pair == 0:
            expected = np.zeros(observed.shape, dtype=float)
        else:
            expected = (float(np.count_nonzero(anchor_mask)) * target_margin /
                        float(record.n_pair))
    elif method == "window_oe":
        if context_start is None or context_end is None:
            raise RuntimeError("validated window_oe context is unavailable")
        context_mask = ((anchor_values >= context_start) &
                        (anchor_values <= context_end))
        target_window = ((target_values >= target_start) &
                         (target_values <= target_end))
        rectangle_mask = context_mask & target_window
        n_rectangle = int(np.count_nonzero(rectangle_mask))
        target_margin = _axis_histogram(target_values[context_mask],
                                        target_start, target_end,
                                        target_bin_size)
        anchor_margin = int(np.count_nonzero(anchor_mask & target_window))
        if n_rectangle <= anchor_margin:
            raise ValueError(
                "window_oe viewpoint background is degenerate: the context "
                "contains no target-window PET outside the anchor")
        expected = (float(anchor_margin) * target_margin /
                    float(n_rectangle))

    if method == "obs":
        values = observed.astype(float)
        if normalization == "library_rpm":
            if logical_total is None:
                raise RuntimeError("library_rpm has no logical total")
            values = values / float(logical_total) * 1000000.0
    else:
        if expected is None:
            raise RuntimeError("O/E method did not produce an expectation")
        values = np.full(observed.shape, np.nan, dtype=float)
        np.divide(observed, expected, out=values, where=expected > 0)
    n_bins = observed.shape[0]
    bin_starts = target_start + np.arange(n_bins) * target_bin_size
    bin_ends = np.minimum(bin_starts + target_bin_size - 1, target_end)
    return TransViewpointResult(
        values=values,
        observed=observed,
        expected=expected,
        target_bin_starts=bin_starts.astype(np.int64),
        target_bin_ends=bin_ends.astype(np.int64),
        chrom_x=record.chrom_x,
        chrom_y=record.chrom_y,
        record_id=record.record_id,
        anchor_axis=anchor_axis,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        target_start=target_start,
        target_end=target_end,
        target_bin_size=target_bin_size,
        context_start=context_start,
        context_end=context_end,
        method=method,
        normalization=normalization,
        n_pair=record.n_pair,
        logical_total=logical_total,
        normalization_scope=dataset.context.normalization_scope,
    )


def _validate_regions(regions, chrom, label):
    regions = tuple(region if isinstance(region, AxisRegion)
                    else AxisRegion(**region) for region in regions)
    if not regions:
        raise ValueError("%s regions must not be empty" % label)
    ids = [region.region_id for region in regions]
    if len(ids) != len(set(ids)):
        raise ValueError("%s region ids must be unique" % label)
    for region in regions:
        if region.chrom != chrom:
            raise ValueError("%s region %r is on %s, expected %s" %
                             (label, region.region_id, region.chrom, chrom))
    ordered = sorted(regions, key=lambda region: (region.start, region.end,
                                                   region.region_id))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start <= previous.end:
            raise ValueError(
                "%s regions must not overlap; %r overlaps %r" %
                (label, previous.region_id, current.region_id))
    return regions


def _region_labels(values, regions):
    """Assign each coordinate to one non-overlapping region in O(N log R)."""
    values = np.asarray(values)
    order = np.asarray(sorted(range(len(regions)),
                              key=lambda index: regions[index].start),
                       dtype=np.int64)
    starts = np.asarray([regions[index].start for index in order],
                        dtype=np.int64)
    ends = np.asarray([regions[index].end for index in order],
                      dtype=np.int64)
    candidates = np.searchsorted(starts, values, side="right") - 1
    labels = np.full(values.shape, -1, dtype=np.int64)
    has_candidate = candidates >= 0
    positions = np.nonzero(has_candidate)[0]
    if positions.size:
        candidate_indexes = candidates[positions]
        inside = values[positions] <= ends[candidate_indexes]
        positions = positions[inside]
        candidate_indexes = candidate_indexes[inside]
        labels[positions] = order[candidate_indexes]
    return labels


def trans_montage(predir,
                  chrom_x,
                  chrom_y,
                  x_regions,
                  y_regions,
                  method="obs",
                  normalization="raw",
                  record_id=None,
                  max_cells=1000000):
    """Build a non-mirrored X-region by Y-region trans montage matrix.

    Regions must be non-overlapping within each axis, ensuring that one PET
    contributes to at most one montage cell.  Pair O/E uses full-pair endpoint
    marginals and ``Npair``; window O/E uses only PETs assigned to the selected
    X/Y region unions.
    """
    _validate_method_and_normalization(method, normalization)
    dataset = _load_dataset(predir)
    record = _resolve_record(dataset, chrom_x, chrom_y, record_id)
    x_regions = _validate_regions(x_regions, record.chrom_x, "X")
    y_regions = _validate_regions(y_regions, record.chrom_y, "Y")
    max_cells = _positive_int(max_cells, "max_cells")
    cells = len(x_regions) * len(y_regions)
    if cells > max_cells:
        raise ValueError("trans montage exceeds max_cells")
    logical_total = _logical_total(dataset, normalization)
    xy = _load_ixy(record.path)
    if int(xy.shape[0]) != record.n_pair:
        raise ValueError("trans record changed after metadata validation")

    x_labels = _region_labels(xy[:, 0], x_regions)
    y_labels = _region_labels(xy[:, 1], y_regions)
    assigned = (x_labels >= 0) & (y_labels >= 0)
    flat = x_labels[assigned] * len(y_regions) + y_labels[assigned]
    observed = np.bincount(
        flat, minlength=cells).reshape(len(x_regions), len(y_regions))
    observed = observed.astype(np.int64, copy=False)

    expected = None
    if method == "pair_oe":
        x_margin = np.bincount(
            x_labels[x_labels >= 0], minlength=len(x_regions)).astype(
                np.int64, copy=False)
        y_margin = np.bincount(
            y_labels[y_labels >= 0], minlength=len(y_regions)).astype(
                np.int64, copy=False)
        if record.n_pair == 0:
            expected = np.zeros(observed.shape, dtype=float)
        else:
            expected = (np.outer(x_margin, y_margin).astype(float) /
                        float(record.n_pair))
    elif method == "window_oe":
        n_rectangle = int(observed.sum())
        if n_rectangle == 0:
            expected = np.zeros(observed.shape, dtype=float)
        else:
            expected = (np.outer(observed.sum(axis=1),
                                 observed.sum(axis=0)).astype(float) /
                        float(n_rectangle))

    if method == "obs":
        values = observed.astype(float)
        if normalization == "library_rpm":
            if logical_total is None:
                raise RuntimeError("library_rpm has no logical total")
            values = values / float(logical_total) * 1000000.0
    else:
        if expected is None:
            raise RuntimeError("O/E method did not produce an expectation")
        values = np.full(observed.shape, np.nan, dtype=float)
        np.divide(observed, expected, out=values, where=expected > 0)
    return TransMontageResult(
        values=values,
        observed=observed,
        expected=expected,
        x_regions=x_regions,
        y_regions=y_regions,
        x_region_ids=tuple(region.region_id for region in x_regions),
        y_region_ids=tuple(region.region_id for region in y_regions),
        chrom_x=record.chrom_x,
        chrom_y=record.chrom_y,
        record_id=record.record_id,
        method=method,
        normalization=normalization,
        n_pair=record.n_pair,
        logical_total=logical_total,
        normalization_scope=dataset.context.normalization_scope,
    )


def _write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_trans_aggregate(result, prefix):
    """Write aggregate arrays plus a provenance sidecar; return both paths."""
    if not isinstance(result, TransAggregateResult):
        raise TypeError("result must be a TransAggregateResult")
    npz_path = str(prefix) + "_trans_agg.npz"
    json_path = str(prefix) + "_trans_agg.json"
    expected = (result.expected_mean if result.expected_mean is not None
                else np.empty((0, 0), dtype=float))
    np.savez_compressed(npz_path, matrix=result.matrix,
                        valid_counts=result.valid_counts,
                        raw_sum=result.raw_sum, raw_mean=result.raw_mean,
                        expected_mean=expected,
                        x_profile=result.x_profile,
                        y_profile=result.y_profile,
                        raw_x_profile=result.raw_x_profile,
                        raw_y_profile=result.raw_y_profile)
    _write_json(json_path, {
        "schema_version": TRANS_AGG_SCHEMA_VERSION,
        "analysis": "trans_aggregate",
        "axis_semantics": "rows=chromX,columns=chromY,no_mirroring",
        "interval_semantics": "closed",
        "method": result.method,
        "normalization": result.normalization,
        "aggregation": "equal_feature_mean",
        "oe_aggregation": "mean_of_feature_level_ratios_over_defined_cells",
        "logical_total": result.logical_total,
        "n_input": result.n_input,
        "n_used": result.n_used,
        "used_feature_ids": list(result.used_feature_ids),
        "used_record_ids": list(result.used_record_ids),
        "used_features": [{
            "feature_id": feature.feature_id,
            "chromX": feature.chrom_x,
            "x_start": feature.x_start,
            "x_end": feature.x_end,
            "chromY": feature.chrom_y,
            "y_start": feature.y_start,
            "y_end": feature.y_end,
            "record_id": feature.record_id,
        } for feature in result.used_features],
        "skipped_feature_ids": list(result.skipped_feature_ids),
        "Npair_by_record": dict(result.npair_by_record),
        "normalization_scope": result.normalization_scope,
        "x_bin_size": result.x_bin_size,
        "y_bin_size": result.y_bin_size,
        "x_flank_bins": result.x_flank_bins,
        "y_flank_bins": result.y_flank_bins,
    })
    return npz_path, json_path


def write_trans_viewpoint(result, prefix):
    """Write one directional viewpoint profile and provenance sidecar."""
    if not isinstance(result, TransViewpointResult):
        raise TypeError("result must be a TransViewpointResult")
    table_path = str(prefix) + "_trans_viewpoint.tsv"
    json_path = str(prefix) + "_trans_viewpoint.json"
    with open(table_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("targetChrom", "start", "end", "observed",
                         "expected", "value", "valid"))
        for index in range(result.values.shape[0]):
            expected = ("NA" if result.expected is None else
                        repr(float(result.expected[index])))
            valid = bool(result.valid_mask[index])
            value = repr(float(result.values[index])) if valid else "NA"
            writer.writerow((result.target_chrom,
                             int(result.target_bin_starts[index]),
                             int(result.target_bin_ends[index]),
                             int(result.observed[index]), expected, value,
                             "true" if valid else "false"))
    _write_json(json_path, {
        "schema_version": TRANS_AGG_SCHEMA_VERSION,
        "analysis": "trans_viewpoint",
        "axis_semantics": "directional,no_mirroring",
        "interval_semantics": "closed",
        "chromX": result.chrom_x,
        "chromY": result.chrom_y,
        "record_id": result.record_id,
        "anchor_axis": result.anchor_axis,
        "anchor_chrom": result.anchor_chrom,
        "target_chrom": result.target_chrom,
        "anchor_start": result.anchor_start,
        "anchor_end": result.anchor_end,
        "target_start": result.target_start,
        "target_end": result.target_end,
        "target_bin_size": result.target_bin_size,
        "context_start": result.context_start,
        "context_end": result.context_end,
        "method": result.method,
        "normalization": result.normalization,
        "Npair": result.n_pair,
        "Lglobal": result.logical_total,
        "normalization_scope": result.normalization_scope,
    })
    return table_path, json_path


def write_trans_montage(result, prefix):
    """Write selected montage matrix, raw/expected arrays, and provenance."""
    if not isinstance(result, TransMontageResult):
        raise TypeError("result must be a TransMontageResult")
    table_path = str(prefix) + "_trans_montage.tsv"
    npz_path = str(prefix) + "_trans_montage.npz"
    json_path = str(prefix) + "_trans_montage.json"
    with open(table_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("X\\Y",) + result.y_region_ids)
        for region_id, row in zip(result.x_region_ids, result.values):
            values = tuple("NA" if not np.isfinite(value)
                           else repr(float(value)) for value in row)
            writer.writerow((region_id,) + values)
    expected = (result.expected if result.expected is not None
                else np.empty((0, 0), dtype=float))
    np.savez_compressed(npz_path, matrix=result.values,
                        valid_mask=result.valid_mask,
                        observed=result.observed, expected=expected)
    _write_json(json_path, {
        "schema_version": TRANS_AGG_SCHEMA_VERSION,
        "analysis": "trans_montage",
        "axis_semantics": "rows=chromX,columns=chromY,no_mirroring",
        "interval_semantics": "closed",
        "chromX": result.chrom_x,
        "chromY": result.chrom_y,
        "record_id": result.record_id,
        "x_regions": [{"id": region.region_id, "start": region.start,
                       "end": region.end}
                      for region in result.x_regions],
        "y_regions": [{"id": region.region_id, "start": region.start,
                       "end": region.end}
                      for region in result.y_regions],
        "method": result.method,
        "normalization": result.normalization,
        "Npair": result.n_pair,
        "Lglobal": result.logical_total,
        "normalization_scope": result.normalization_scope,
    })
    return table_path, npz_path, json_path


def plot_trans_aggregate(result, path):
    """Plot a bounded aggregate rectangle without enforcing square axes."""
    if not isinstance(result, TransAggregateResult):
        raise TypeError("result must be a TransAggregateResult")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(4, 3))
    figure, axis = cast(Any, figure), cast(Any, axis)
    image = axis.imshow(result.matrix, origin="lower", aspect="auto")
    axis.set_xlabel("chromY offset bins")
    axis.set_ylabel("chromX offset bins")
    axis.set_title("Trans aggregate (%s, n=%s)" %
                   (result.method, result.n_used))
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def plot_trans_viewpoint(result, path):
    """Plot a directional viewpoint profile on its true target chromosome."""
    if not isinstance(result, TransViewpointResult):
        raise TypeError("result must be a TransViewpointResult")
    import matplotlib.pyplot as plt
    centers = ((result.target_bin_starts + result.target_bin_ends) / 2.0)
    figure, axis = plt.subplots(figsize=(5, 2.5))
    figure, axis = cast(Any, figure), cast(Any, axis)
    axis.plot(centers, result.values)
    axis.set_xlabel("%s coordinate" % result.target_chrom)
    axis.set_ylabel("%s%s" %
                    (result.method,
                     " (RPM)" if result.normalization == "library_rpm" else ""))
    axis.set_title("%s viewpoint on %s" %
                   (result.anchor_chrom, result.target_chrom))
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def plot_trans_montage(result, path):
    """Plot a directional region-by-region montage heatmap."""
    if not isinstance(result, TransMontageResult):
        raise TypeError("result must be a TransMontageResult")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(4, 3))
    figure, axis = cast(Any, figure), cast(Any, axis)
    image = axis.imshow(result.values, origin="upper", aspect="auto")
    axis.set_xticks(np.arange(len(result.y_region_ids)))
    axis.set_xticklabels(result.y_region_ids, rotation=90)
    axis.set_yticks(np.arange(len(result.x_region_ids)))
    axis.set_yticklabels(result.x_region_ids)
    axis.set_xlabel(result.chrom_y)
    axis.set_ylabel(result.chrom_x)
    axis.set_title("Trans montage (%s)" % result.method)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return str(path)


# Camel-case aliases fit the existing public Python API while keeping the new
# implementation itself PEP-8 readable.
aggTransLoops = aggregate_trans_loops
transViewPoint = trans_viewpoint
montageTrans = trans_montage


__all__ = [
    "AxisRegion", "TRANS_AGG_SCHEMA_VERSION", "TransAggregateResult",
    "TransFeature", "TransMontageResult", "TransViewpointResult",
    "aggTransLoops", "aggregate_trans_loops", "montageTrans",
    "plot_trans_aggregate", "plot_trans_montage", "plot_trans_viewpoint",
    "read_axis_regions", "read_trans_features", "transViewPoint",
    "trans_montage", "trans_viewpoint", "write_trans_aggregate",
    "write_trans_montage", "write_trans_viewpoint",
]
