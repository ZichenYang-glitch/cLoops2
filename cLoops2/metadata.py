#!/usr/bin/env python
#--coding:utf-8--
"""Metadata validation and library-depth semantics for cLoops2 datasets.

The historical ``petMeta.json`` format has one physical count (``Unique
PETs``).  That is sufficient for a freshly preprocessed, fully materialized
directory, but not for a cis/trans projection of an overall sample.  A
projection may physically contain ``Ktrans`` rows while representing a
logical overall depth ``T``.  This module keeps those concepts separate:

``physical_total``
    Rows currently materialized in the output ``.ixy`` files.
``logical_total``
    Overall retained-library depth represented by those rows.  It is unknown
    when provenance is insufficient; it is never guessed from a legacy
    trans-only directory.
``npair_by_record``
    Physical row count for each individual chromosome-pair record.

Validation is deliberately read-only.  Callers that have inspected the files
can pass ``actual_counts`` (record id -> row count) to verify the structural
manifest.  Content hashes are optional and outside this minimal structural
contract.
"""

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Sequence, Tuple


CAP_PHYSICAL_ROWS = "physical_rows"
CAP_PAIR_COUNTS = "pair_counts"
CAP_GLOBAL_NORMALIZATION = "global_normalization"
CAP_FORMAL_INFERENCE = "formal_inference"
CAP_NESTED_DOWNSAMPLE = "nested_downsample"


class MetadataCapabilityError(ValueError):
    """Raised when metadata cannot support a requested operation."""


@dataclass(frozen=True)
class LibraryContext(object):
    """Validated physical and logical library context.

    ``formal_inference`` only means that the sampling provenance is known and
    has never used replacement.  It does not, by itself, correct candidate
    selection or validate a downstream statistical model.
    """

    physical_total: Optional[int]
    logical_total: Optional[int]
    npair_by_record: Mapping[str, int]
    emit: Optional[str]
    source_kind: str
    replacement_this_step: Optional[bool]
    replacement_ever: Optional[bool]
    validity: str
    capabilities: FrozenSet[str]
    reasons: Tuple[str, ...]
    normalization_scope: str = "unknown"

    @property
    def pair_totals(self):
        """Backward-friendly alias for ``npair_by_record``."""
        return self.npair_by_record

    def require(self, capability):
        """Require one validated capability and return this context.

        This makes call sites explicit, for example::

            ctx.require(CAP_GLOBAL_NORMALIZATION).logical_total
        """
        if capability not in self.capabilities:
            if capability == CAP_FORMAL_INFERENCE:
                if self.replacement_ever is True:
                    detail = ("replacement_ever=true; resampled rows are not "
                              "independent evidence")
                elif self.replacement_ever is None:
                    detail = ("replacement history is unknown; formal "
                              "inference requires replacement_ever=false")
                else:
                    detail = ("the source transform is not eligible for "
                              "formal inference")
            else:
                detail = "; ".join(self.reasons) or "capability is unavailable"
            raise MetadataCapabilityError(
                "metadata does not support %r: %s" % (capability, detail))
        return self

    def get_npair(self, record_id):
        """Return the physical PET count for one chromosome-pair record."""
        self.require(CAP_PAIR_COUNTS)
        try:
            return int(self.npair_by_record[record_id])
        except KeyError:
            raise KeyError("unknown chromosome-pair record %r" % record_id)


def inspect_actual_counts(meta):
    """Read authoritative per-record row counts from metadata ``.ixy`` paths.

    Files are memory-mapped when possible and loaded one at a time otherwise.
    Formal consumers use this helper so an edited file cannot remain trusted
    merely because aggregate ``Unique PETs`` or an old manifest was unchanged.
    """
    import joblib
    import numpy as np

    if not isinstance(meta, Mapping):
        raise TypeError("meta must be a mapping")
    data = meta.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("petMeta.json is missing data")
    counts = {}
    for category in ("cis", "trans"):
        entries = data.get(category, {})
        if not isinstance(entries, Mapping):
            raise ValueError("petMeta data.%s must be an object" % category)
        for key in sorted(entries):
            entry = entries[key]
            if not isinstance(entry, Mapping) or not entry.get("ixy"):
                raise ValueError("metadata record %s:%s has no ixy path" %
                                 (category, key))
            try:
                matrix = joblib.load(entry["ixy"], mmap_mode="r")
            except Exception:
                matrix = joblib.load(entry["ixy"])
            if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or
                    matrix.shape[1] != 2):
                raise ValueError("ixy file %s must have shape (n, 2)" %
                                 entry["ixy"])
            record_id = str(entry.get(
                "record_id", "%s:%s" % (category, key)))
            if record_id in counts:
                raise ValueError("duplicate metadata record_id %r" % record_id)
            counts[record_id] = int(matrix.shape[0])
            del matrix
    return counts


def _first(mapping, *keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _nonnegative_int(value, label, errors, required=False):
    if value is None:
        if required:
            errors.append("missing %s" % label)
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        errors.append("%s must be a non-negative integer" % label)
        return None
    value = int(value)
    if value < 0:
        errors.append("%s must be a non-negative integer" % label)
        return None
    return value


def _optional_bool(value, label, errors, required=False):
    if value is None:
        if required:
            errors.append("missing %s" % label)
        return None
    if not isinstance(value, bool):
        errors.append("%s must be boolean" % label)
        return None
    return value


def _normalise_counts(counts, label, errors):
    """Normalize flat or ``{cis: {}, trans: {}}`` counts by record id."""
    if counts is None:
        return None
    if not isinstance(counts, Mapping):
        errors.append("%s must be a mapping" % label)
        return None
    out = {}
    is_nested = any(key in counts for key in ("cis", "trans"))
    if is_nested:
        unknown = set(counts).difference(("cis", "trans"))
        if unknown:
            errors.append("%s mixes category and record keys" % label)
            return None
        for category in ("cis", "trans"):
            category_counts = counts.get(category, {})
            if not isinstance(category_counts, Mapping):
                errors.append("%s.%s must be a mapping" % (label, category))
                continue
            for key, value in category_counts.items():
                record_id = "%s:%s" % (category, key)
                n = _nonnegative_int(value, "%s[%s]" % (label, record_id),
                                     errors, required=True)
                if n is not None:
                    out[record_id] = n
    else:
        for key, value in counts.items():
            record_id = str(key)
            n = _nonnegative_int(value, "%s[%s]" % (label, record_id),
                                 errors, required=True)
            if n is not None:
                out[record_id] = n
    return out


def _category_totals(value, label, errors, required=False):
    if value is None:
        if required:
            errors.append("missing %s" % label)
        return None
    if not isinstance(value, Mapping):
        errors.append("%s must be a mapping" % label)
        return None
    totals = {}
    for category in ("cis", "trans"):
        totals[category] = _nonnegative_int(value.get(category, 0),
                                            "%s.%s" % (label, category),
                                            errors,
                                            required=True)
    unknown = set(value).difference(("cis", "trans"))
    if unknown:
        errors.append("%s has unknown categories: %s" %
                      (label, ", ".join(sorted(map(str, unknown)))))
    return totals


def _emit_from_data(meta):
    data = meta.get("data", {})
    if not isinstance(data, Mapping):
        return None
    cis = data.get("cis", {})
    trans = data.get("trans", {})
    has_cis = isinstance(cis, Mapping) and bool(cis)
    has_trans = isinstance(trans, Mapping) and bool(trans)
    if has_cis and has_trans:
        return "all"
    if has_cis:
        return "cis"
    if has_trans:
        return "trans"
    return "all"


def _looks_like_legacy_root(meta):
    """Recognize metadata written directly by the historical ``pre`` path."""
    required = ("Total PETs", "Total Cis PETs", "Total Trans PETs")
    return all(key in meta for key in required)


def _validate_physical_categories(meta, actual, physical_total, errors):
    """Validate optional v0.2 per-category and Retention root metadata."""
    has_cis = "Unique Cis PETs" in meta
    has_trans = "Unique Trans PETs" in meta
    if has_cis != has_trans:
        errors.append(
            "Unique Cis PETs and Unique Trans PETs must be recorded together")
        return

    unique_by_category = None
    if has_cis:
        unique_by_category = {
            "cis": _nonnegative_int(meta.get("Unique Cis PETs"),
                                    "Unique Cis PETs", errors, required=True),
            "trans": _nonnegative_int(meta.get("Unique Trans PETs"),
                                      "Unique Trans PETs", errors,
                                      required=True),
        }
        if (None not in unique_by_category.values() and
                physical_total is not None and
                sum(unique_by_category.values()) != physical_total):
            errors.append(
                "Unique Cis PETs + Unique Trans PETs does not match physical "
                "total")

    if actual is not None and unique_by_category is not None:
        observed = {"cis": 0, "trans": 0}
        categorized = True
        for record_id, rows in actual.items():
            if record_id.startswith("cis:"):
                observed["cis"] += rows
            elif record_id.startswith("trans:"):
                observed["trans"] += rows
            else:
                categorized = False
                break
        if categorized and observed != unique_by_category:
            errors.append(
                "Unique Cis/Trans PETs do not match actual category counts")

    retention = meta.get("Retention")
    if retention is None:
        return
    if not isinstance(retention, Mapping):
        errors.append("Retention must be a mapping")
        return
    retain_trans = retention.get("retain trans")
    if not isinstance(retain_trans, bool):
        errors.append("Retention.retain trans must be boolean")
    categories = retention.get("retained categories")
    if (not isinstance(categories, Sequence) or
            isinstance(categories, (str, bytes))):
        errors.append("Retention.retained categories must be a list")
        return
    if any(category not in ("cis", "trans") for category in categories):
        errors.append(
            "Retention.retained categories must contain unique cis/trans values")
        return
    if len(set(categories)) != len(categories):
        errors.append(
            "Retention.retained categories must contain unique cis/trans values")
        return
    if isinstance(retain_trans, bool) and retain_trans != ("trans" in categories):
        errors.append(
            "Retention.retain trans disagrees with retained categories")
    data = meta.get("data", {})
    if isinstance(data, Mapping):
        for category in ("cis", "trans"):
            entries = data.get(category, {})
            if category not in categories and isinstance(entries, Mapping) and entries:
                errors.append(
                    "data.%s is non-empty but the category was not retained" %
                    category)
    if (unique_by_category is not None and "trans" not in categories and
            unique_by_category["trans"] not in (None, 0)):
        errors.append(
            "Unique Trans PETs is non-zero although trans was not retained")


def _manifest_counts(sampling, errors):
    records = sampling.get("records")
    if records is None:
        errors.append("missing Sampling.records")
        return None, None, None
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        errors.append("Sampling.records must be a list")
        return None, None, None

    source, selected, written = {}, {}, {}
    for index, record in enumerate(records):
        prefix = "Sampling.records[%s]" % index
        if not isinstance(record, Mapping):
            errors.append("%s must be a mapping" % prefix)
            continue
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            errors.append("%s.record_id must be a non-empty string" % prefix)
            continue
        if record_id in written:
            errors.append("duplicate Sampling record_id %r" % record_id)
            continue
        category = record.get("category")
        if category not in ("cis", "trans"):
            errors.append("%s.category must be cis or trans" % prefix)
        chrom_x = record.get("chromX")
        chrom_y = record.get("chromY")
        if not isinstance(chrom_x, str) or not chrom_x:
            errors.append("%s.chromX must be a non-empty string" % prefix)
        if not isinstance(chrom_y, str) or not chrom_y:
            errors.append("%s.chromY must be a non-empty string" % prefix)
        if (category == "cis" and chrom_x != chrom_y) or (
                category == "trans" and chrom_x == chrom_y):
            errors.append("%s category disagrees with chromX/chromY" % prefix)
        shape = record.get("shape")
        if (not isinstance(shape, Sequence) or isinstance(shape, (str, bytes))
                or len(shape) != 2):
            errors.append("%s.shape must be [rows, 2]" % prefix)
            shape_rows = None
        else:
            shape_rows = _nonnegative_int(shape[0], "%s.shape[0]" % prefix,
                                          errors, required=True)
            shape_columns = _nonnegative_int(shape[1],
                                             "%s.shape[1]" % prefix,
                                             errors,
                                             required=True)
            if shape_columns is not None and shape_columns != 2:
                errors.append("%s.shape[1] must equal 2" % prefix)
        dtype = record.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            errors.append("%s.dtype must be a non-empty string" % prefix)
        materialized = record.get("materialized")
        if materialized is not None and not isinstance(materialized, bool):
            errors.append("%s.materialized must be boolean" % prefix)
        source_n = _nonnegative_int(record.get("source_rows"),
                                    "%s.source_rows" % prefix,
                                    errors,
                                    required=True)
        selected_n = _nonnegative_int(record.get("selected_rows"),
                                      "%s.selected_rows" % prefix,
                                      errors,
                                      required=True)
        written_n = _nonnegative_int(record.get("written_rows"),
                                     "%s.written_rows" % prefix,
                                     errors,
                                     required=True)
        if None in (source_n, selected_n, written_n):
            continue
        if shape_rows is not None and shape_rows != source_n:
            errors.append("%s.shape[0] does not match source_rows" % prefix)
        if written_n > selected_n:
            errors.append("%s.written_rows exceeds selected_rows" % prefix)
        if materialized is False and written_n > 0:
            errors.append(
                "%s cannot write rows from a non-materialized source" %
                prefix)
        source[record_id] = source_n
        selected[record_id] = selected_n
        written[record_id] = written_n
    return source, selected, written


def _compare_actual_to_manifest(actual, written, errors):
    if actual is None or written is None:
        return
    for record_id, expected in written.items():
        observed = actual.get(record_id)
        if observed is None:
            if expected != 0:
                errors.append("actual_counts is missing written record %r" %
                              record_id)
        elif observed != expected:
            errors.append(
                "actual row count for %r is %s, expected %s" %
                (record_id, observed, expected))
    extras = set(actual).difference(written)
    if extras:
        errors.append("actual_counts has records absent from manifest: %s" %
                      ", ".join(sorted(extras)))


def _sum_or_none(counts):
    if counts is None:
        return None
    return int(sum(counts.values()))


def _check_category_closure(records, sampling, source, selected, written,
                            errors):
    source_by_category = _category_totals(
        sampling.get("available_by_category"),
        "Sampling.available_by_category", errors, required=True)
    selected_by_category = _category_totals(
        sampling.get("selected_by_category"),
        "Sampling.selected_by_category", errors, required=True)
    written_by_category = _category_totals(
        sampling.get("written_by_category"),
        "Sampling.written_by_category", errors, required=True)
    if (source is None or selected is None or written is None or
            not isinstance(records, Sequence)):
        return source_by_category, selected_by_category, written_by_category

    calculated_source = {"cis": 0, "trans": 0}
    calculated_selected = {"cis": 0, "trans": 0}
    calculated_written = {"cis": 0, "trans": 0}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        record_id = record.get("record_id")
        category = record.get("category")
        if record_id not in selected or category not in calculated_selected:
            continue
        calculated_source[category] += source[record_id]
        calculated_selected[category] += selected[record_id]
        calculated_written[category] += written[record_id]
    if (source_by_category is not None and
            source_by_category != calculated_source):
        errors.append("Sampling.available_by_category does not match records")
    if (selected_by_category is not None and
            selected_by_category != calculated_selected):
        errors.append("Sampling.selected_by_category does not match records")
    if (written_by_category is not None and
            written_by_category != calculated_written):
        errors.append("Sampling.written_by_category does not match records")
    return source_by_category, selected_by_category, written_by_category


def build_library_context(meta, actual_counts=None):
    """Validate ``petMeta`` and return its library semantics.

    Parameters
    ----------
    meta : mapping
        Parsed ``petMeta.json`` object.
    actual_counts : mapping, optional
        Complete physical counts obtained by inspecting output files.  Use
        either ``{record_id: rows}`` or legacy nested
        ``{cis: {key: rows}, trans: {key: rows}}`` form.  Record ids must match
        ``Sampling.records[].record_id`` for schema-v2 metadata.

    Semantic inconsistencies produce a context with ``validity='invalid'``;
    callers can still use independently observed physical/pair counts, but no
    global normalization or formal-inference capability is granted.
    """
    if not isinstance(meta, Mapping):
        raise TypeError("meta must be a mapping")

    errors = []
    notes = []
    actual_error_count = len(errors)
    actual = _normalise_counts(actual_counts, "actual_counts", errors)
    actual_valid = (actual_counts is not None and actual is not None and
                    len(errors) == actual_error_count)
    unique = _nonnegative_int(meta.get("Unique PETs"), "Unique PETs", errors,
                              required=True)
    sampling = meta.get("Sampling")
    transformation = meta.get("Transformation")

    physical_total = _sum_or_none(actual)
    logical_total = None
    pair_totals = actual
    emit = _emit_from_data(meta)
    source_kind = "unknown"
    replacement_this_step = None
    replacement_ever = None
    normalization_scope = "unknown"
    legacy_root = False
    explicit_root = False
    recognised_transform = False
    transform_declared_valid = False
    formal_inference_eligible = True

    if sampling is not None:
        if not isinstance(sampling, Mapping):
            errors.append("Sampling must be a mapping")
            sampling = {}
        schema_version = _nonnegative_int(sampling.get("schema_version"),
                                          "Sampling.schema_version", errors,
                                          required=True)
        if schema_version is not None and schema_version < 2:
            errors.append("Sampling.schema_version must be at least 2")
        sampling_validity = sampling.get("validity", "valid")
        if sampling_validity not in ("valid", "invalid"):
            errors.append("Sampling.validity must be valid or invalid")
        elif sampling_validity == "invalid":
            invalidation = sampling.get("invalidation_reasons", [])
            if isinstance(invalidation, Sequence) and not isinstance(
                    invalidation, (str, bytes)):
                errors.extend("Sampling invalidated: %s" % reason
                              for reason in invalidation)
            else:
                errors.append("Sampling metadata has been invalidated")

        emit = _first(sampling, "emit", "mode")
        if emit not in ("cis", "trans", "all"):
            errors.append("Sampling.emit must be cis, trans, or all")

        source_kind = sampling.get("source_kind")
        if source_kind not in ("root", "all_sample", "projection"):
            errors.append(
                "Sampling.source_kind must be root, all_sample, or projection")
            source_kind = "unknown"

        universe = sampling.get("universe")
        if universe != "retained_all":
            errors.append("Sampling.universe must be retained_all")
        else:
            normalization_scope = "retained_all"

        logical_total = _nonnegative_int(
            _first(sampling, "target_logical_total", "target PETs",
                   "target total PETs"),
            "Sampling.target_logical_total",
            errors,
            required=True)
        source_logical_total = _nonnegative_int(
            _first(sampling, "source_logical_total", "available PETs",
                   "available total PETs"),
            "Sampling.source_logical_total",
            errors,
            required=True)
        declared_physical = _nonnegative_int(
            sampling.get("physical_output_total"),
            "Sampling.physical_output_total",
            errors,
            required=True)
        replacement_this_step = _optional_bool(
            _first(sampling, "replacement_this_step", "replacement"),
            "Sampling.replacement_this_step",
            errors,
            required=True)
        if "replacement_ever" not in sampling:
            errors.append("missing Sampling.replacement_ever")
        else:
            replacement_ever = _optional_bool(
                sampling.get("replacement_ever"),
                "Sampling.replacement_ever",
                errors,
                required=False)
            if replacement_ever is None:
                notes.append(
                    "Sampling ancestry does not prove whether replacement "
                    "occurred")
        formal_value = sampling.get(
            "formal_inference_eligible", replacement_ever is False)
        if not isinstance(formal_value, bool):
            errors.append(
                "Sampling.formal_inference_eligible must be boolean")
            formal_inference_eligible = False
        else:
            formal_inference_eligible = formal_value
        if formal_inference_eligible and replacement_ever is not False:
            errors.append(
                "formal Sampling provenance requires "
                "replacement_ever=false")
        if replacement_this_step is True and replacement_ever is not True:
            errors.append(
                "Sampling.replacement_ever must be true after replacement")

        direction = sampling.get("direction")
        if (source_logical_total is not None and logical_total is not None and
                direction in ("downsample", "identity", "upsample")):
            expected_direction = ("downsample" if logical_total <
                                  source_logical_total else "upsample" if
                                  logical_total > source_logical_total else
                                  "identity")
            if direction != expected_direction:
                errors.append(
                    "Sampling.direction is %s, expected %s" %
                    (direction, expected_direction))
        elif direction not in ("downsample", "identity", "upsample"):
            errors.append(
                "Sampling.direction must be downsample, identity, or upsample")
        if (direction in ("downsample", "identity") and
                replacement_this_step is True):
            errors.append(
                "downsample/identity operations must not use replacement")
        if direction == "upsample" and replacement_this_step is False:
            errors.append("upsample operations must use replacement")

        records = sampling.get("records")
        source_rows, selected_rows, written_rows = _manifest_counts(
            sampling, errors)
        _compare_actual_to_manifest(actual, written_rows, errors)
        _check_category_closure(records, sampling, source_rows, selected_rows,
                                written_rows, errors)
        manifest_physical = _sum_or_none(written_rows)
        manifest_selected = _sum_or_none(selected_rows)
        manifest_source = _sum_or_none(source_rows)
        if pair_totals is None:
            pair_totals = written_rows
        elif written_rows is not None:
            # Keep explicit zero-row manifest entries so callers can
            # distinguish a known, unallocated pair from an unknown/missing
            # record, while actual file counts remain authoritative.
            pair_totals = dict(written_rows, **pair_totals)
        if physical_total is None:
            physical_total = manifest_physical
        for label, value in (("Sampling.physical_output_total",
                              declared_physical),
                             ("manifest written total", manifest_physical),
                             ("Unique PETs", unique)):
            if (physical_total is not None and value is not None and
                    value != physical_total):
                errors.append("%s is %s, expected physical total %s" %
                              (label, value, physical_total))
        if (logical_total is not None and manifest_selected is not None and
                manifest_selected != logical_total):
            errors.append(
                "manifest selected total is %s, expected logical total %s" %
                (manifest_selected, logical_total))
        if (source_logical_total is not None and manifest_source is not None and
                manifest_source != source_logical_total):
            errors.append(
                "manifest source total is %s, expected source logical total %s"
                % (manifest_source, source_logical_total))
        if (emit == "all" and selected_rows is not None and
                written_rows is not None and selected_rows != written_rows):
            errors.append(
                "emit=all requires every selected record to be written")
        if emit in ("cis", "trans") and isinstance(records, Sequence):
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                record_id = record.get("record_id")
                category = record.get("category")
                if (category != emit and written_rows is not None and
                        written_rows.get(record_id, 0) != 0):
                    errors.append(
                        "projection writes record %r outside emit=%s" %
                        (record_id, emit))
        if replacement_this_step is False and source_rows is not None and \
                selected_rows is not None:
            for record_id, selected_n in selected_rows.items():
                if selected_n > source_rows.get(record_id, 0):
                    errors.append(
                        "record %r selects more rows than available without "
                        "replacement" % record_id)
    elif transformation is not None:
        recognised_transform = True
        if not isinstance(transformation, Mapping):
            errors.append("Transformation must be a mapping")
            transformation = {}
        schema_version = _nonnegative_int(
            transformation.get("schema_version"),
            "Transformation.schema_version", errors, required=True)
        if schema_version is not None and schema_version < 1:
            errors.append("Transformation.schema_version must be at least 1")
        operation = transformation.get("operation")
        if operation not in ("combine", "external_update", "filter"):
            errors.append("unsupported Transformation.operation %r" %
                          operation)
        declared_physical = _nonnegative_int(
            transformation.get("physical_output_total"),
            "Transformation.physical_output_total", errors, required=True)
        logical_total = _nonnegative_int(
            transformation.get("logical_total"),
            "Transformation.logical_total", errors, required=True)
        if "replacement_ever" not in transformation:
            errors.append("missing Transformation.replacement_ever")
        else:
            replacement_ever = _optional_bool(
                transformation.get("replacement_ever"),
                "Transformation.replacement_ever", errors, required=False)
            if replacement_ever is None:
                notes.append(
                    "Transformation ancestry does not prove whether "
                    "replacement occurred")
        replacement_this_step = False
        source_kind = "all_sample"
        emit = _emit_from_data(meta)
        normalization_scope = "transformed_retained"
        formal_value = transformation.get("formal_inference_eligible", False)
        if not isinstance(formal_value, bool):
            errors.append(
                "Transformation.formal_inference_eligible must be boolean")
            formal_inference_eligible = False
        else:
            formal_inference_eligible = formal_value
        if (operation in ("combine", "external_update", "filter") and
                formal_inference_eligible):
            errors.append(
                "%s Transformation is not eligible for formal inference" %
                operation)
        transform_declared_valid = (
            transformation.get("validity") == "valid")
        if formal_inference_eligible and replacement_ever is not False:
            errors.append(
                "formal Transformation provenance requires "
                "replacement_ever=false")
        if (formal_inference_eligible and
                transformation.get("parents_fully_materialized") is not True):
            errors.append(
                "formal Transformation provenance requires fully "
                "materialized parents")
        if not transform_declared_valid:
            notes.append("Transformation provenance is not fully validated")
        if physical_total is None:
            physical_total = unique
        for label, value in (
                ("Transformation.physical_output_total", declared_physical),
                ("Transformation.logical_total", logical_total),
                ("Unique PETs", unique)):
            if (physical_total is not None and value is not None and
                    value != physical_total):
                errors.append("%s is %s, expected physical total %s" %
                              (label, value, physical_total))
    else:
        legacy_root = _looks_like_legacy_root(meta)
        explicit_root = (
            legacy_root and isinstance(meta.get("Retention"), Mapping) and
            "Unique Cis PETs" in meta and "Unique Trans PETs" in meta)
        if legacy_root:
            for key in ("Total PETs", "Total Cis PETs", "Total Trans PETs"):
                _nonnegative_int(meta.get(key), key, errors, required=True)
        if physical_total is None:
            physical_total = unique
        if unique is not None and physical_total is not None and unique != physical_total:
            errors.append("Unique PETs is %s, expected physical total %s" %
                          (unique, physical_total))
        if legacy_root and not errors:
            # Historical ``pre`` writes actual retained rows to Unique PETs.
            # Even when Total Trans PETs is non-zero, default pre may have
            # intentionally retained only cis; the logical scope is therefore
            # the retained output, not all raw input PETs.
            logical_total = physical_total
            source_kind = "root"
            replacement_this_step = False
            replacement_ever = False
            if explicit_root:
                normalization_scope = "retained_root"
                notes.append(
                    "pre root validated from Retention and category counts")
            else:
                normalization_scope = "legacy_retained_root"
                notes.append(
                    "legacy pre root inferred from preprocessing QC fields")
        else:
            notes.append(
                "legacy metadata lacks Sampling provenance; logical library "
                "depth and replacement history are unknown")

    if physical_total is None:
        physical_total = unique
    if unique is not None and physical_total is not None and unique != physical_total:
        message = "Unique PETs is %s, expected physical total %s" % (
            unique, physical_total)
        if message not in errors:
            errors.append(message)
    _validate_physical_categories(meta, actual, physical_total, errors)

    capabilities = set()
    if physical_total is not None and (actual_valid or not errors):
        capabilities.add(CAP_PHYSICAL_ROWS)
    if pair_totals is not None and (actual_valid or not errors):
        capabilities.add(CAP_PAIR_COUNTS)

    if errors:
        validity = "invalid"
    elif sampling is not None:
        validity = "valid"
    elif recognised_transform and transform_declared_valid:
        validity = "valid"
    elif recognised_transform:
        validity = "unknown"
        logical_total = None
    elif explicit_root:
        validity = "valid"
    elif legacy_root:
        validity = "legacy_root_assumed"
    else:
        validity = "unknown"

    if validity in ("valid", "legacy_root_assumed") and logical_total is not None:
        capabilities.add(CAP_GLOBAL_NORMALIZATION)
        if replacement_ever is False and formal_inference_eligible:
            capabilities.add(CAP_FORMAL_INFERENCE)
        if source_kind == "projection" and replacement_ever is False:
            capabilities.add(CAP_NESTED_DOWNSAMPLE)

    reasons = tuple(errors + notes)
    return LibraryContext(
        physical_total=physical_total,
        logical_total=logical_total,
        npair_by_record=MappingProxyType(dict(pair_totals or {})),
        emit=emit,
        source_kind=source_kind,
        replacement_this_step=replacement_this_step,
        replacement_ever=replacement_ever,
        validity=validity,
        capabilities=frozenset(capabilities),
        reasons=reasons,
        normalization_scope=normalization_scope,
    )


def validate_library_metadata(meta, actual_counts=None):
    """Alias with a validation-oriented name for public call sites."""
    return build_library_context(meta, actual_counts=actual_counts)
