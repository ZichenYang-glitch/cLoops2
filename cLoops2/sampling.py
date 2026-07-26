#!/usr/bin/env python
# --coding:utf-8--
"""Deterministic whole-library PET sampling.

The public ``samplePETs`` command exposes an output mode (cis, trans, or all),
but the sampling universe is the complete set of PET categories retained by
the input directory.  Allocation is therefore independent of the output mode:
running the same input and seed in ``all`` and ``trans`` modes produces the
same trans allocation and row selections.

For a fully materialized input, a target smaller than the logical library is
allocated with sequential hypergeometric draws and sampled without
replacement.  A larger target is allocated with a multinomial draw and sampled
with replacement.  Files are loaded one at a time per worker; the complete
library is never concatenated in memory.

Sampling metadata distinguishes the represented (logical) library depth from
the number of rows physically written.  This matters for cis/trans projection
outputs, whose physical row count is generally smaller than the whole-library
target.
"""

from __future__ import absolute_import

import copy
import hashlib
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed

from cLoops2.io import writeNewJson
from cLoops2.metadata import (CAP_FORMAL_INFERENCE,
                              build_library_context)


_CATEGORIES = ("cis", "trans")
_CATEGORY_ORDER = {"cis": 0, "trans": 1}
_ALGORITHM_VERSION = "whole-library-hypergeom-multinomial-v0.2"


def _as_nonnegative_int(value, name):
    """Return *value* as an int, rejecting booleans and negative values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError("%s must be a non-negative integer" % name)
    value = int(value)
    if value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


def _load_ixy(path, mmap=True):
    """Load an ixy array, falling back for compressed joblib payloads.

    Normal cLoops2 ``.ixy`` files are uncompressed and can be memory mapped.
    joblib versions differ in how they react when ``mmap_mode`` is requested
    for a compressed payload, so a normal load is used as a compatibility
    fallback.
    """
    path = str(path)
    if mmap:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return joblib.load(path, mmap_mode="r")
        except (OSError, TypeError, ValueError) as mmap_error:
            try:
                return joblib.load(path)
            except Exception:
                raise ValueError("unable to load ixy file %s: %s" %
                                 (path, mmap_error))
    return joblib.load(path)


def _parse_record_chromosomes(category, key, entry):
    """Get explicit axes, with a fail-fast fallback for legacy metadata."""
    chrom_x = entry.get("chromX") or entry.get("chrom_x")
    chrom_y = entry.get("chromY") or entry.get("chrom_y")
    if chrom_x is None or chrom_y is None:
        fields = key.split("-")
        if len(fields) != 2:
            raise ValueError(
                "metadata key %r is not an unambiguous chromosome pair; "
                "v0.2 metadata must provide chromX and chromY" % key)
        chrom_x, chrom_y = fields
    chrom_x, chrom_y = str(chrom_x), str(chrom_y)
    if category == "cis" and chrom_x != chrom_y:
        raise ValueError("cis record %r has different chromosome axes" % key)
    if category == "trans" and chrom_x == chrom_y:
        raise ValueError("trans record %r has identical chromosome axes" % key)
    return chrom_x, chrom_y


def _collect_records(predir, meta):
    """Collect stable records and count their actual rows."""
    data = meta.get("data")
    if not isinstance(data, dict):
        raise ValueError("petMeta.json is missing a data object")

    records = []
    seen_paths = set()
    seen_names = set()
    for category in _CATEGORIES:
        entries = data.get(category, {})
        if entries is None:
            entries = {}
        if not isinstance(entries, dict):
            raise ValueError("meta data.%s must be an object" % category)
        for key in sorted(entries):
            entry = entries[key]
            if not isinstance(entry, dict) or not entry.get("ixy"):
                raise ValueError("record %s.%s has no ixy path" %
                                 (category, key))
            source = Path(entry["ixy"]).expanduser()
            if not source.is_absolute():
                source = Path(predir) / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError("ixy file does not exist: %s" % source)
            if source.suffix != ".ixy":
                raise ValueError("ixy path has an unexpected suffix: %s" %
                                 source)
            if source.stem != key:
                raise ValueError(
                    "metadata key %r does not match source basename %r" %
                    (key, source.stem))
            if str(source) in seen_paths:
                raise ValueError("the same ixy path is referenced more than once: %s" %
                                 source)
            if source.name in seen_names:
                raise ValueError("multiple records would overwrite %s" % source.name)
            seen_paths.add(str(source))
            seen_names.add(source.name)

            chrom_x, chrom_y = _parse_record_chromosomes(category, key, entry)
            mat = _load_ixy(source, mmap=True)
            if (not isinstance(mat, np.ndarray) or mat.ndim != 2 or
                    mat.shape[1] != 2):
                raise ValueError("ixy file %s must contain a two-dimensional array "
                                 "with exactly two columns" % source)
            rows = int(mat.shape[0])
            shape = [int(v) for v in mat.shape]
            dtype = str(mat.dtype)
            del mat
            records.append({
                "record_id": "%s:%s" % (category, key),
                "category": category,
                "key": key,
                "chromX": chrom_x,
                "chromY": chrom_y,
                "source_path": str(source),
                "filename": source.name,
                "source_rows": rows,
                "shape": shape,
                "dtype": dtype,
                "materialized": True,
            })
    records.sort(key=lambda r: (_CATEGORY_ORDER[r["category"]], r["key"]))
    return records


def _rng_for(effective_seed, label):
    """Derive a stable PCG64 stream without Python's randomized ``hash``."""
    payload = ("cLoops2-samplePETs-v0.2\0%s\0%s" %
               (effective_seed, label)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    entropy = np.frombuffer(digest[:16], dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _allocate_without_replacement(counts, target, rng):
    """Allocate an exact global subset with sequential hypergeometrics."""
    counts = [int(v) for v in counts]
    total = int(sum(counts))
    target = int(target)
    if target < 0 or target > total:
        raise ValueError("without-replacement target %s exceeds total %s" %
                         (target, total))
    if target == total:
        return list(counts)

    remaining_total = total
    remaining_target = target
    allocations = []
    for index, count in enumerate(counts):
        if remaining_target == 0 or count == 0:
            take = 0
        elif remaining_target == remaining_total:
            take = count
        elif index == len(counts) - 1:
            take = remaining_target
        else:
            take = int(rng.hypergeometric(
                ngood=count,
                nbad=remaining_total - count,
                nsample=remaining_target,
            ))
        if take < 0 or take > count:
            raise RuntimeError("invalid hypergeometric allocation")
        allocations.append(take)
        remaining_total -= count
        remaining_target -= take
    if sum(allocations) != target or remaining_target != 0:
        raise RuntimeError("sampling allocation did not reach the exact target")
    return allocations


def _allocate_with_replacement(counts, target, rng):
    """Allocate global iid draws to files with a multinomial."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("cannot upsample an empty PET library")
    probabilities = counts.astype(float) / float(total)
    allocations = rng.multinomial(int(target), probabilities)
    return [int(v) for v in allocations]


def _sampling_parent(meta):
    sampling = meta.get("Sampling")
    if not isinstance(sampling, dict):
        return None
    try:
        schema_version = int(sampling.get("schema_version", 0))
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version < 2:
        raise ValueError(
            "input contains Sampling metadata without a supported v0.2 schema; "
            "its logical library depth cannot be trusted")
    return sampling


def _category_totals(records, field="source_rows"):
    totals = {"cis": 0, "trans": 0}
    for record in records:
        totals[record["category"]] += int(record[field])
    return totals


def _plan_full_source(records, target, mode, allocation_rng):
    counts = [record["source_rows"] for record in records]
    source_total = int(sum(counts))
    if source_total == 0 and target > 0:
        raise ValueError("cannot sample a positive target from an empty library")

    if target < source_total:
        direction = "downsample"
        allocations = _allocate_without_replacement(counts, target,
                                                     allocation_rng)
        replacement = False
    elif target == source_total:
        direction = "identity"
        allocations = list(counts)
        replacement = False
    else:
        direction = "upsample"
        allocations = _allocate_with_replacement(counts, target,
                                                  allocation_rng)
        replacement = True

    source_by_category = _category_totals(records)
    if target > 0 and mode in _CATEGORIES and source_by_category[mode] == 0:
        raise ValueError(
            "mode=%s requested, but the input has no retained %s PET records" %
            (mode, mode))
    selected_by_category = {"cis": 0, "trans": 0}
    for record, take in zip(records, allocations):
        selected_by_category[record["category"]] += int(take)
    return {
        "source_kind": "root",
        "source_logical_total": source_total,
        "source_by_category": source_by_category,
        "allocations": allocations,
        "selected_by_category": selected_by_category,
        "direction": direction,
        "replacement_this_step": replacement,
        "replacement_ever": replacement,
        "records": records,
        "physical_source_total": source_total,
    }


def _plan_materialized_sample(records, target, mode, parent, allocation_rng):
    """Plan sampling from a previous all-sample output."""
    plan = _plan_full_source(records, target, mode, allocation_rng)
    actual_total = sum(record["source_rows"] for record in records)
    parent_logical = _as_nonnegative_int(
        parent.get("target_logical_total"),
        "Sampling.target_logical_total",
    )
    if actual_total != parent_logical:
        raise ValueError(
            "all-sample input has %s physical rows but represents logical depth %s"
            % (actual_total, parent_logical))
    plan["source_kind"] = "all_sample"
    plan["source_logical_total"] = parent_logical
    parent_replacement = parent.get("replacement_ever")
    if plan["replacement_this_step"]:
        plan["replacement_ever"] = True
    elif parent_replacement is False:
        plan["replacement_ever"] = False
    else:
        # ``None`` means that an older/foreign ancestor did not record enough
        # provenance to prove that replacement never happened.  A later
        # without-replacement sample must not launder that uncertainty.
        plan["replacement_ever"] = None
    return plan


def _plan_projection(records, target, mode, parent, allocation_rng):
    """Plan a strict nested downsample from a materialized projection."""
    parent_emit = parent.get("emit", parent.get("mode"))
    if parent_emit not in _CATEGORIES:
        raise ValueError("projection metadata has an invalid emit category")
    if mode != parent_emit:
        raise ValueError(
            "a %s projection can only be sampled again with mode=%s; omitted "
            "categories cannot be recovered" % (parent_emit, parent_emit))

    source_logical = _as_nonnegative_int(
        parent.get("target_logical_total"),
        "Sampling.target_logical_total",
    )
    physical_rows = int(sum(record["source_rows"] for record in records))
    parent_physical = _as_nonnegative_int(
        parent.get("physical_output_total", physical_rows),
        "Sampling.physical_output_total",
    )
    if physical_rows != parent_physical:
        raise ValueError(
            "projection content no longer matches its Sampling metadata: "
            "%s rows found, %s expected" % (physical_rows, parent_physical))
    if physical_rows > source_logical:
        raise ValueError("projection physical rows exceed its logical depth")
    if target > source_logical:
        raise ValueError(
            "projection upsampling is not supported in v0.2: target logical "
            "depth %s exceeds parent logical depth %s" %
            (target, source_logical))

    if target == source_logical:
        selected_projection = physical_rows
        direction = "identity"
    elif target == 0 or physical_rows == 0:
        selected_projection = 0
        direction = "downsample"
    else:
        selected_projection = int(allocation_rng.hypergeometric(
            ngood=physical_rows,
            nbad=source_logical - physical_rows,
            nsample=target,
        ))
        direction = "downsample"

    # The parent projection manifest still describes the selected but omitted
    # category.  Preserve those virtual records so source/selected category
    # totals remain closed over the whole logical library, while only the emit
    # category needs physical source files for this nested downsample.
    parent_records = parent.get("records")
    if not isinstance(parent_records, list):
        raise ValueError(
            "projection metadata is missing the record manifest required for "
            "nested downsampling")
    physical_by_id = {record["record_id"]: record for record in records}
    manifest_records = []
    for parent_record in parent_records:
        if not isinstance(parent_record, dict):
            raise ValueError("projection record manifest is invalid")
        record_id = parent_record.get("record_id")
        category = parent_record.get("category")
        if not isinstance(record_id, str) or category not in _CATEGORIES:
            raise ValueError("projection record manifest is invalid")
        parent_selected_rows = _as_nonnegative_int(
            parent_record.get("selected_rows"),
            "Sampling.records[%s].selected_rows" % record_id,
        )
        if category == parent_emit:
            physical = physical_by_id.pop(record_id, None)
            if physical is None:
                if parent_selected_rows != 0:
                    raise ValueError(
                        "projection is missing materialized record %s" %
                        record_id)
                # A zero-row record has no physical file, but remains a known
                # member of the logical manifest.
                physical = {
                    "record_id": record_id,
                    "category": category,
                    "key": parent_record.get("key", record_id.split(":", 1)[-1]),
                    "chromX": parent_record.get("chromX"),
                    "chromY": parent_record.get("chromY"),
                    "source_path": None,
                    "filename": None,
                    "source_rows": 0,
                    "shape": [0, int(parent_record.get("shape", [0, 2])[1])],
                    "dtype": parent_record.get("dtype", "int64"),
                    "materialized": False,
                }
            elif int(physical["source_rows"]) != parent_selected_rows:
                raise ValueError(
                    "projection record %s has %s physical rows, expected %s" %
                    (record_id, physical["source_rows"], parent_selected_rows))
            manifest_records.append(physical)
        else:
            parent_shape = parent_record.get("shape", [parent_selected_rows, 2])
            columns = (int(parent_shape[1]) if isinstance(parent_shape, list)
                       and len(parent_shape) == 2 else 2)
            manifest_records.append({
                "record_id": record_id,
                "category": category,
                "key": parent_record.get("key", record_id.split(":", 1)[-1]),
                "chromX": parent_record.get("chromX"),
                "chromY": parent_record.get("chromY"),
                "source_path": None,
                "filename": None,
                "source_rows": parent_selected_rows,
                "shape": [parent_selected_rows, columns],
                "dtype": parent_record.get("dtype", "int64"),
                "materialized": False,
                "origin_source_rows": parent_record.get("source_rows"),
            })
    if physical_by_id:
        raise ValueError(
            "projection contains physical records absent from its Sampling "
            "manifest: %s" % ", ".join(sorted(physical_by_id)))
    manifest_records.sort(
        key=lambda r: (_CATEGORY_ORDER[r["category"]], r["key"]))

    emit_records = [r for r in manifest_records
                    if r["category"] == parent_emit]
    other_records = [r for r in manifest_records
                     if r["category"] != parent_emit]
    emit_allocations = _allocate_without_replacement(
        [record["source_rows"] for record in emit_records],
        selected_projection,
        allocation_rng,
    )
    other_allocations = _allocate_without_replacement(
        [record["source_rows"] for record in other_records],
        target - selected_projection,
        allocation_rng,
    )
    allocation_by_id = {
        record["record_id"]: take
        for record, take in zip(emit_records, emit_allocations)
    }
    allocation_by_id.update({
        record["record_id"]: take
        for record, take in zip(other_records, other_allocations)
    })
    allocations = [allocation_by_id[record["record_id"]]
                   for record in manifest_records]
    selected_by_category = {
        parent_emit: selected_projection,
        ("trans" if parent_emit == "cis" else "cis"):
        target - selected_projection,
    }
    parent_selected = parent.get("selected_by_category")
    if isinstance(parent_selected, dict):
        source_by_category = {
            category: _as_nonnegative_int(parent_selected.get(category, 0),
                                          "Sampling.selected_by_category.%s" %
                                          category)
            for category in _CATEGORIES
        }
        if (sum(source_by_category.values()) != source_logical or
                source_by_category[parent_emit] != physical_rows):
            raise ValueError(
                "projection category totals do not match its physical/logical "
                "Sampling metadata")
    else:
        source_by_category = {
            parent_emit: physical_rows,
            ("trans" if parent_emit == "cis" else "cis"):
            source_logical - physical_rows,
        }
    return {
        "source_kind": "projection",
        "source_logical_total": source_logical,
        "source_by_category": source_by_category,
        "allocations": allocations,
        "selected_by_category": selected_by_category,
        "direction": direction,
        "replacement_this_step": False,
        "replacement_ever": parent.get("replacement_ever"),
        "records": manifest_records,
        "physical_source_total": physical_rows,
    }


def _sample_one_record(record, outdir, selected_rows, write_record,
                       replacement, effective_seed):
    """Sample one file with a record-specific deterministic RNG."""
    selected_rows = int(selected_rows)
    source_rows = int(record["source_rows"])
    if not write_record or selected_rows == 0:
        return 0
    if source_rows == 0:
        raise ValueError("cannot sample from empty record %s" %
                         record["record_id"])
    if not replacement and selected_rows > source_rows:
        raise ValueError("without-replacement allocation exceeds source rows")

    if not record.get("materialized", True) or not record.get("source_path"):
        raise ValueError("selected record is not physically materialized: %s" %
                         record["record_id"])
    source = Path(record["source_path"])
    destination = Path(outdir) / record["filename"]
    mat = _load_ixy(source, mmap=True)
    if int(mat.shape[0]) != source_rows:
        raise ValueError("source record changed after planning: %s" % source)
    if not replacement and selected_rows == source_rows:
        # Re-dump rather than byte-copy so compressed legacy payloads become a
        # normal uncompressed cLoops2 ixy that downstream mmap readers accept.
        # This still preserves source row order and processes only one file per
        # worker.
        joblib.dump(np.asarray(mat), str(destination))
        return selected_rows
    rng = _rng_for(effective_seed, "record:%s" % record["record_id"])
    indices = rng.choice(source_rows, size=selected_rows,
                         replace=bool(replacement))
    indices.sort()
    sampled = np.asarray(mat[indices, :])
    joblib.dump(sampled, str(destination))
    return int(sampled.shape[0])


def _write_sampling_metadata(outdir, source_meta, records, plan, mode,
                             target, requested_seed, effective_seed):
    """Rebuild petMeta.json and append the v0.2 Sampling contract."""
    writeNewJson(str(outdir))
    meta_path = Path(outdir) / "petMeta.json"
    with meta_path.open() as handle:
        output_meta = json.load(handle)

    physical_total = int(output_meta.get("Unique PETs", -1))
    written_by_category = {"cis": 0, "trans": 0}
    manifest = []
    emit_categories = set(_CATEGORIES if mode == "all" else (mode, ))
    for record, selected in zip(records, plan["allocations"]):
        written = int(selected) if record["category"] in emit_categories else 0
        written_by_category[record["category"]] += written
        manifest.append({
            "record_id": record["record_id"],
            "category": record["category"],
            "key": record["key"],
            "chromX": record["chromX"],
            "chromY": record["chromY"],
            "source_path": record.get("source_path"),
            "output_path": (str((Path(outdir) / record["filename"]).resolve())
                            if written > 0 and record.get("filename") else None),
            "source_rows": int(record["source_rows"]),
            "selected_rows": int(selected),
            "written_rows": written,
            "shape": list(record["shape"]),
            "dtype": record["dtype"],
            "materialized": bool(record.get("materialized", True)),
        })
        if "origin_source_rows" in record:
            manifest[-1]["origin_source_rows"] = record["origin_source_rows"]
    expected_physical = int(sum(written_by_category.values()))
    if physical_total != expected_physical:
        raise RuntimeError(
            "rebuilt petMeta.json reports %s rows; sampling wrote %s" %
            (physical_total, expected_physical))

    available_categories = {
        category for category, count in plan["source_by_category"].items()
        if int(count) > 0
    }
    emit_categories = set(_CATEGORIES if mode == "all" else (mode, ))
    output_source_kind = (
        "projection" if plan["source_kind"] == "projection" or
        bool(available_categories - emit_categories) else "all_sample")

    input_unique = source_meta.get("Unique PETs")
    actual_source_physical = int(plan["physical_source_total"])
    sampling = {
        "schema_version": 2,
        "validity": "valid",
        "mode": mode,
        "universe": "retained_all",
        "emit": mode,
        "source_kind": output_source_kind,
        "parent_source_kind": plan["source_kind"],
        "source_logical_total": int(plan["source_logical_total"]),
        "target_logical_total": int(target),
        "physical_source_total": actual_source_physical,
        "physical_output_total": physical_total,
        "available_by_category": {
            category: int(plan["source_by_category"].get(category, 0))
            for category in _CATEGORIES
        },
        "selected_by_category": {
            category: int(plan["selected_by_category"].get(category, 0))
            for category in _CATEGORIES
        },
        "written_by_category": written_by_category,
        "direction": plan["direction"],
        "seed": int(effective_seed),
        "requested_seed": (None if requested_seed is None else
                           int(requested_seed)),
        "replacement_this_step": bool(plan["replacement_this_step"]),
        "replacement_ever": plan["replacement_ever"],
        "replacement_history_known": plan["replacement_ever"] is not None,
        "formal_inference_eligible": bool(
            plan.get("formal_inference_eligible", False) and
            plan["replacement_ever"] is False),
        "source_provenance_validity": plan.get(
            "source_provenance_validity", "unknown"),
        "integrity_level": "structural",
        "bit_generator": "PCG64",
        "algorithm_version": _ALGORITHM_VERSION,
        "source_directory": str(Path(plan["predir"]).resolve()),
        "source_metadata_unique_pets": input_unique,
        "source_count_mismatch": (
            input_unique is not None and
            int(input_unique) != actual_source_physical
        ),
        "records": manifest,
        # Compatibility aliases requested in the original command contract.
        "target PETs": int(target),
        "available PETs": int(plan["source_logical_total"]),
        "source directory": str(Path(plan["predir"]).resolve()),
    }
    output_meta["Sampling"] = sampling
    # Retention describes the upstream preprocessing scope (for example,
    # whether ``pre -trans`` was used and which chromosome whitelist applied),
    # so it remains true after sampling.  Raw/redundancy totals are purposely
    # not copied because they describe a different physical directory.
    if "Retention" in source_meta:
        output_meta["Retention"] = copy.deepcopy(source_meta["Retention"])
    temporary = meta_path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(output_meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(meta_path))
    return sampling


def sample_pets(predir, outdir, target_total, cpu=1, mode="cis", seed=None):
    """Sample a complete retained PET library and emit a category projection.

    Parameters
    ----------
    predir : str or path-like
        cLoops2 directory containing ``petMeta.json`` and source ``.ixy`` files.
    outdir : str or path-like
        Non-existent or empty destination directory.
    target_total : int
        Logical whole-library target.  It is *not* the requested number of
        physical rows for cis/trans projections.
    cpu : int
        Maximum number of files processed concurrently.  RNG streams are tied
        to stable record IDs, so output is independent of this value.
    mode : {"cis", "trans", "all"}
        Categories physically written from the same global sample.
    seed : int or None
        A non-negative seed.  ``None`` obtains system entropy and records the
        effective seed in ``Sampling`` metadata.
    """
    target_total = _as_nonnegative_int(target_total, "target_total")
    if (isinstance(cpu, (bool, np.bool_)) or
            not isinstance(cpu, (int, np.integer))):
        raise ValueError("cpu must be -1 or a positive integer")
    cpu = int(cpu)
    if cpu == 0 or cpu < -1:
        raise ValueError("cpu must be -1 or a positive integer")
    if mode not in ("cis", "trans", "all"):
        raise ValueError("mode must be one of cis, trans, or all")
    if seed is not None:
        seed = _as_nonnegative_int(seed, "seed")
        effective_seed = seed
    else:
        effective_seed = int(np.random.SeedSequence().entropy)

    predir = Path(predir).expanduser().resolve()
    outdir = Path(outdir).expanduser().resolve()
    if not predir.is_dir():
        raise FileNotFoundError("input directory does not exist: %s" % predir)
    if predir == outdir:
        raise ValueError("input and output directories must be different")
    meta_path = predir / "petMeta.json"
    if not meta_path.is_file():
        raise FileNotFoundError("missing input metadata: %s" % meta_path)
    if outdir.exists():
        if not outdir.is_dir():
            raise ValueError("output path exists and is not a directory")
        if any(outdir.iterdir()):
            raise ValueError("output directory exists and is not empty: %s" %
                             outdir)

    with meta_path.open() as handle:
        meta = json.load(handle)
    records = _collect_records(predir, meta)
    actual_counts = {
        record["record_id"]: int(record["source_rows"])
        for record in records
    }
    source_context = build_library_context(meta, actual_counts=actual_counts)
    if source_context.validity == "invalid":
        raise ValueError(
            "input metadata is structurally invalid: %s" %
            ("; ".join(source_context.reasons) or "unknown reason"))
    parent = _sampling_parent(meta)
    if parent is not None:
        if source_context.validity != "valid":
            raise ValueError(
                "input Sampling provenance is invalid: %s" %
                ("; ".join(source_context.reasons) or "unknown reason"))
    allocation_rng = _rng_for(effective_seed, "allocation")
    if parent is None:
        plan = _plan_full_source(records, target_total, mode, allocation_rng)
        if source_context.source_kind == "all_sample":
            plan["source_kind"] = "all_sample"
    elif parent.get("source_kind") == "projection":
        plan = _plan_projection(records, target_total, mode, parent,
                                allocation_rng)
    else:
        plan = _plan_materialized_sample(records, target_total, mode, parent,
                                         allocation_rng)
    plan["source_provenance_validity"] = source_context.validity
    plan["formal_inference_eligible"] = (
        CAP_FORMAL_INFERENCE in source_context.capabilities and
        not plan["replacement_this_step"])
    if parent is None and not plan["replacement_this_step"]:
        # Fresh/new pre roots and historical pre roots with their QC fields
        # establish a no-replacement source.  Bare Unique/data directories may
        # instead be old sample/combine/filter outputs, so their ancestry is
        # intentionally left unknown rather than certified as false.
        if source_context.replacement_ever is True:
            plan["replacement_ever"] = True
        elif (source_context.source_kind in ("root", "all_sample") and
                source_context.replacement_ever is False and
                source_context.validity in ("valid", "legacy_root_assumed")):
            plan["replacement_ever"] = False
        else:
            plan["replacement_ever"] = None
    plan["predir"] = str(predir)
    plan_records = plan.get("records", records)

    # All validation and allocation completes before the destination is made.
    outdir.mkdir(parents=True, exist_ok=True)
    emit_categories = set(_CATEGORIES if mode == "all" else (mode, ))
    jobs = []
    for record, selected in zip(plan_records, plan["allocations"]):
        write_record = record["category"] in emit_categories
        if write_record and int(selected) > 0:
            jobs.append((record, int(selected), write_record))
    if jobs:
        n_jobs = cpu if cpu == -1 else min(cpu, len(jobs))
        written = Parallel(n_jobs=n_jobs)(
            delayed(_sample_one_record)(
                record,
                str(outdir),
                selected,
                write_record,
                plan["replacement_this_step"],
                effective_seed,
            ) for record, selected, write_record in jobs)
        expected = sum(selected for _, selected, _ in jobs)
        if sum(written) != expected:
            raise RuntimeError("sample workers did not write the planned rows")

    return _write_sampling_metadata(
        outdir,
        meta,
        plan_records,
        plan,
        mode,
        target_total,
        seed,
        effective_seed,
    )
