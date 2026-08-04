#!/usr/bin/env python3
# --coding: utf-8 --
"""Replicate-aware differential analysis for directional trans rectangles.

The workflow counts one fixed candidate family in every biological sample,
keeps global/pair/marginal exposures separate, and exports raw matrices.  A
single sample per condition gets an explicitly exploratory conditional exact
rate test; replicated conditions use a negative-binomial GLM.  Replacement or
unknown provenance is rejected before counting.
"""

from __future__ import absolute_import

import csv
import json
import math
import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
from scipy import stats as _scipy_stats


def _two_sided_binomial(k, n, probability):
    """Support both modern scipy.binomtest and legacy scipy.binom_test."""
    modern = getattr(_scipy_stats, "binomtest", None)
    if modern is not None:
        return float(modern(k, n, probability).pvalue)
    legacy = getattr(_scipy_stats, "binom_test", None)
    if legacy is None:
        raise ImportError("SciPy provides neither binomtest nor binom_test")
    return float(legacy(k, n, probability))

from cLoops2.ds import TransContactIndex
from cLoops2.metadata import (CAP_FORMAL_INFERENCE,
                              CAP_GLOBAL_NORMALIZATION,
                              build_library_context,
                              inspect_actual_counts)
from cLoops2.trans_stats import adjust_pvalues


@dataclass(frozen=True)
class TransDiffCandidate(object):
    loop_id: str
    chrom_x: str
    x_start: int
    x_end: int
    chrom_y: str
    y_start: int
    y_end: int


@dataclass(frozen=True)
class TransDiffSample(object):
    sample: str
    condition: str
    directory: str


@dataclass(frozen=True)
class TransDiffResult(object):
    loop_id: str
    chrom_x: str
    x_start: int
    x_end: int
    chrom_y: str
    y_start: int
    y_end: int
    reference: str
    contrast: str
    reference_samples: int
    contrast_samples: int
    reference_total: int
    contrast_total: int
    reference_mean_cpm: float
    contrast_mean_cpm: float
    log2_fold_change: float
    offset_scope: str
    dispersion: object
    coefficient: object
    standard_error: object
    pvalue: object
    bh_adjusted_p: object
    by_adjusted_p: object
    adjustment_method: object
    alpha: object
    method: str
    model_status: str
    biological_inference: bool
    candidate_source: str
    significant: object


def _coordinate(value, label):
    try:
        value = int(float(value))
    except (OverflowError, TypeError, ValueError):
        raise ValueError("%s must be an integer coordinate" % label)
    if value < 0:
        raise ValueError("%s must be non-negative" % label)
    return value


def read_trans_diff_candidates(path):
    """Read fixed rectangles from the common first seven loop columns."""
    candidates, seen = [], set()
    with open(path) as handle:
        for line_number, row in enumerate(
                csv.reader(handle, delimiter="\t"), start=1):
            if not row or row[0].startswith("#") or row[0] == "loopId":
                continue
            if len(row) < 7:
                raise ValueError("candidate line %s has fewer than 7 columns" %
                                 line_number)
            loop_id, chrom_x, chrom_y = row[0], row[1], row[4]
            if not loop_id or loop_id in seen:
                raise ValueError("candidate loop ids must be unique and non-empty")
            if chrom_x == chrom_y:
                raise ValueError("candidate %s is not trans" % loop_id)
            xa, xb = _coordinate(row[2], "x_start"), _coordinate(row[3], "x_end")
            ya, yb = _coordinate(row[5], "y_start"), _coordinate(row[6], "y_end")
            if xa > xb or ya > yb:
                raise ValueError("candidate %s has a reversed interval" % loop_id)
            seen.add(loop_id)
            candidates.append(TransDiffCandidate(
                loop_id, chrom_x, xa, xb, chrom_y, ya, yb))
    if not candidates:
        raise ValueError("trans candidate file contains no candidates")
    return candidates


def read_trans_sample_sheet(path):
    """Read sample/condition/directory TSV; relative paths follow the sheet."""
    base = Path(path).expanduser().resolve().parent
    samples, seen, conditions = [], set(), []
    with open(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample", "condition", "directory"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("sample sheet requires sample, condition, directory")
        for row in reader:
            sample = str(row.get("sample", "")).strip()
            condition = str(row.get("condition", "")).strip()
            raw_directory = str(row.get("directory", "")).strip()
            if not sample or sample in seen or not condition or not raw_directory:
                raise ValueError("sample names must be unique and fields non-empty")
            directory = Path(raw_directory).expanduser()
            if not directory.is_absolute():
                directory = base / directory
            directory = directory.resolve()
            if not directory.is_dir():
                raise FileNotFoundError("sample directory does not exist: %s" %
                                        directory)
            seen.add(sample)
            if condition not in conditions:
                conditions.append(condition)
            samples.append(TransDiffSample(sample, condition, str(directory)))
    if not samples or len(conditions) != 2:
        raise ValueError("trans differential analysis requires two conditions")
    return samples, conditions


def _retention_signature(meta):
    retention = meta.get("Retention")
    if not isinstance(retention, dict):
        raise ValueError("formal trans differential analysis requires Retention")
    if not retention.get("retain trans", False) or \
            "trans" not in retention.get("retained categories", []):
        raise ValueError("sample metadata says trans PETs were not retained")
    return json.dumps(retention, sort_keys=True, separators=(",", ":"))


def _pair_records(meta):
    records = {}
    for key, entry in sorted(meta.get("data", {}).get("trans", {}).items()):
        if not isinstance(entry, dict) or not entry.get("ixy"):
            raise ValueError("trans metadata record %s has no ixy path" % key)
        chrom_x, chrom_y = entry.get("chromX"), entry.get("chromY")
        if chrom_x is None or chrom_y is None:
            fields = key.split("-")
            if len(fields) != 2:
                raise ValueError("ambiguous trans record %s" % key)
            chrom_x, chrom_y = fields
        pair = (str(chrom_x), str(chrom_y))
        if pair in records:
            raise ValueError("duplicate trans chromosome pair %s,%s" % pair)
        records[pair] = entry["ixy"]
    return records


def _structural_zero(meta, chrom_x, chrom_y):
    retention = meta.get("Retention", {})
    whitelist = retention.get("chromosome whitelist", [])
    return (bool(retention.get("retain trans", False)) and
            "trans" in retention.get("retained categories", []) and
            (not whitelist or
             (chrom_x in whitelist and chrom_y in whitelist)))


def _load_index(path):
    try:
        matrix = joblib.load(path, mmap_mode="r")
    except Exception:
        matrix = joblib.load(path)
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or
            matrix.shape[1] != 2 or
            (matrix.size and not np.issubdtype(matrix.dtype, np.integer))):
        raise ValueError("trans ixy %s must be an integer (n, 2) array" % path)
    return TransContactIndex.from_matrix(matrix)


def count_trans_candidates(samples, candidates, offset_scope="global"):
    """Strict-axis count the full candidate family in every raw sample."""
    if offset_scope not in ("global", "pair", "marginal"):
        raise ValueError("offset_scope must be global, pair, or marginal")
    shape = (len(candidates), len(samples))
    counts = np.zeros(shape, dtype=np.int64)
    pair_counts = np.zeros(shape, dtype=np.int64)
    reads_a = np.zeros(shape, dtype=np.int64)
    reads_b = np.zeros(shape, dtype=np.int64)
    exposures = np.zeros(shape, dtype=float)
    library_depths = np.zeros(len(samples), dtype=float)
    signatures = []
    grouped = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault((candidate.chrom_x, candidate.chrom_y), []).append(
            (index, candidate))

    for sample_index, sample in enumerate(samples):
        with open(os.path.join(sample.directory, "petMeta.json")) as handle:
            meta = json.load(handle)
        context = build_library_context(
            meta, actual_counts=inspect_actual_counts(meta))
        context.require(CAP_FORMAL_INFERENCE)
        context.require(CAP_GLOBAL_NORMALIZATION)
        signatures.append(_retention_signature(meta))
        library_depths[sample_index] = context.logical_total
        records = _pair_records(meta)
        for pair, items in grouped.items():
            path, reverse = records.get(pair), False
            if path is None and records.get((pair[1], pair[0])) is not None:
                path, reverse = records[(pair[1], pair[0])], True
            if path is None:
                if not _structural_zero(meta, pair[0], pair[1]):
                    raise ValueError("sample %s cannot prove zero pair %s,%s" %
                                     (sample.sample, pair[0], pair[1]))
                contact = TransContactIndex([], [])
            else:
                contact = _load_index(path)
            for loop_index, candidate in items:
                if reverse:
                    xa, xb, ya, yb = (candidate.y_start, candidate.y_end,
                                      candidate.x_start, candidate.x_end)
                else:
                    xa, xb, ya, yb = (candidate.x_start, candidate.x_end,
                                      candidate.y_start, candidate.y_end)
                stored_ra = contact.count_x(xa, xb)
                stored_rb = contact.count_y(ya, yb)
                # Export marginals in the candidate's chromX/chromY order,
                # even when metadata stores the chromosome pair reversed.
                ra, rb = ((stored_rb, stored_ra) if reverse else
                          (stored_ra, stored_rb))
                rab, n_pair = contact.count_rect(xa, xb, ya, yb), contact.number
                counts[loop_index, sample_index] = rab
                pair_counts[loop_index, sample_index] = n_pair
                reads_a[loop_index, sample_index] = ra
                reads_b[loop_index, sample_index] = rb
                if offset_scope == "global":
                    exposures[loop_index, sample_index] = context.logical_total
                elif offset_scope == "pair":
                    exposures[loop_index, sample_index] = n_pair
                elif n_pair > 0:
                    exposures[loop_index, sample_index] = float(ra) * rb / n_pair
            del contact
    if len(set(signatures)) != 1:
        raise ValueError("sample Retention scopes differ")
    return {"counts": counts, "pair_counts": pair_counts,
            "reads_a": reads_a, "reads_b": reads_b,
            "exposures": exposures, "library_depths": library_depths}


def _effect(counts, exposures, conditions, reference, contrast):
    ref = np.asarray([value == reference for value in conditions], dtype=bool)
    con = np.asarray([value == contrast for value in conditions], dtype=bool)
    ref_rates = np.divide(counts[ref], exposures[ref],
                          out=np.zeros(ref.sum()), where=exposures[ref] > 0)
    con_rates = np.divide(counts[con], exposures[con],
                          out=np.zeros(con.sum()), where=exposures[con] > 0)
    ref_cpm, con_cpm = float(ref_rates.mean() * 1e6), float(con_rates.mean() * 1e6)
    ref_rate = ((float(counts[ref].sum()) + 0.5) /
                max(float(exposures[ref].sum()), 1.0))
    con_rate = ((float(counts[con].sum()) + 0.5) /
                max(float(exposures[con].sum()), 1.0))
    return ref, con, ref_cpm, con_cpm, math.log(con_rate / ref_rate, 2)


def _raw_dispersion(counts, exposures, conditions, reference, contrast):
    fitted = np.zeros_like(exposures, dtype=float)
    for group in (reference, contrast):
        selected = np.asarray([value == group for value in conditions])
        total_exposure = float(exposures[selected].sum())
        rate = (float(counts[selected].sum()) / total_exposure
                if total_exposure > 0 else 0.0)
        fitted[selected] = exposures[selected] * rate
    denominator = float(np.square(fitted).sum())
    if denominator <= 0:
        return 0.0
    return max(0.0, float((np.square(counts - fitted) - fitted).sum()) /
               denominator)


def _exact_p(counts, exposures, ref, con):
    if np.any(exposures <= 0):
        return 1.0, "zero_exposure"
    y_ref, y_con = int(counts[ref].sum()), int(counts[con].sum())
    e_ref, e_con = float(exposures[ref].sum()), float(exposures[con].sum())
    if y_ref + y_con == 0:
        return 1.0, "all_zero"
    probability = e_con / (e_ref + e_con)
    return _two_sided_binomial(y_con, y_ref + y_con, probability), \
        "exact_technical_sampling"


def _fit_nb(counts, exposures, conditions, contrast, dispersion, ref, con):
    if np.any(exposures <= 0):
        return None, None, 1.0, "zero_exposure", False
    if int(counts.sum()) == 0:
        return 0.0, None, 1.0, "all_zero", True
    try:
        import statsmodels.api as sm
        design = np.column_stack([
            np.ones(len(conditions)),
            np.asarray([value == contrast for value in conditions], dtype=float)
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(
                counts, design,
                family=sm.families.NegativeBinomial(
                    alpha=max(float(dispersion), 1e-8)),
                offset=np.log(exposures)).fit(maxiter=100, disp=False)
        values = float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1])
        if not all(map(math.isfinite, values)):
            raise ValueError("non-finite NB fit")
        return values[0], values[1], values[2], "nb_glm", True
    except Exception as error:
        pvalue, _ = _exact_p(counts, exposures, ref, con)
        return None, None, pvalue, \
            "fallback_exact:%s" % type(error).__name__, False


def differential_test(candidates, samples, matrices, reference, contrast,
                      method="auto", offset_scope="global", adjustment="BH",
                      alpha=0.05, prior_df=10.0, candidate_source=""):
    """Test the complete family without count/effect-dependent filtering."""
    if not candidate_source or not str(candidate_source).strip():
        raise ValueError("candidate_source declaration is required")
    conditions = [sample.condition for sample in samples]
    if reference == contrast or reference not in conditions or \
            contrast not in conditions:
        raise ValueError("reference and contrast must name the two conditions")
    n_ref, n_con = conditions.count(reference), conditions.count(contrast)
    if method == "auto":
        method = "nb" if n_ref >= 2 and n_con >= 2 else "exact"
    if method not in ("exact", "nb", "export"):
        raise ValueError("method must be auto, exact, nb, or export")
    if method == "exact" and (n_ref != 1 or n_con != 1):
        raise ValueError("exact mode requires one sample per condition")
    if method == "nb" and (n_ref < 2 or n_con < 2):
        raise ValueError("NB mode requires at least two samples per condition")
    adjustment = str(adjustment).upper()
    if adjustment not in ("BH", "BY"):
        raise ValueError("adjustment must be BH or BY")
    alpha, prior_df = float(alpha), float(prior_df)
    if not 0 < alpha <= 1 or prior_df < 0:
        raise ValueError("alpha must be in (0,1] and prior_df non-negative")

    counts_matrix, exposure_matrix = matrices["counts"], matrices["exposures"]
    if method == "nb":
        raw_dispersions = [
            _raw_dispersion(counts_matrix[i].astype(float), exposure_matrix[i],
                            conditions, reference, contrast)
            for i in range(len(candidates))]
        positive = [value for value in raw_dispersions if value > 0]
        common_dispersion = float(np.median(positive)) if positive else 0.1
    else:
        raw_dispersions, common_dispersion = [], None

    results, pvalues = [], []
    for index, candidate in enumerate(candidates):
        counts = counts_matrix[index].astype(float)
        exposures = exposure_matrix[index].astype(float)
        ref, con, ref_cpm, con_cpm, log2fc = _effect(
            counts, exposures, conditions, reference, contrast)
        dispersion = coefficient = standard_error = pvalue = None
        biological, status, result_method = False, "export_only", "export"
        if method == "exact":
            pvalue, status = _exact_p(counts, exposures, ref, con)
            result_method = "exact_rate"
        elif method == "nb":
            if common_dispersion is None:
                raise RuntimeError("NB common dispersion was not initialized")
            degrees = max(0, len(samples) - 2)
            raw = raw_dispersions[index]
            dispersion = ((degrees * raw + prior_df * common_dispersion) /
                          (degrees + prior_df) if degrees + prior_df else raw)
            coefficient, standard_error, pvalue, status, biological = _fit_nb(
                counts, exposures, conditions, contrast, dispersion, ref, con)
            result_method = "nb_glm_wald"
        if pvalue is not None:
            pvalues.append(float(pvalue))
        results.append(TransDiffResult(
            candidate.loop_id, candidate.chrom_x, candidate.x_start,
            candidate.x_end, candidate.chrom_y, candidate.y_start,
            candidate.y_end, reference, contrast, n_ref, n_con,
            int(counts[ref].sum()), int(counts[con].sum()), ref_cpm, con_cpm,
            log2fc, offset_scope, dispersion, coefficient, standard_error,
            pvalue, None, None, None, None, result_method, status, biological,
            str(candidate_source).strip(), None))

    if method == "export":
        return results
    bh, by = adjust_pvalues(pvalues, "BH"), adjust_pvalues(pvalues, "BY")
    selected, output, pindex = (bh if adjustment == "BH" else by), [], 0
    for result in results:
        if result.pvalue is None:
            output.append(result)
            continue
        # With one sample per condition the exact rate test quantifies only
        # technical sampling uncertainty.  It must not look like a biological
        # differential call even when its adjusted p-value is small.
        significant = (bool(selected[pindex] <= alpha)
                       if method == "nb" and result.biological_inference
                       else None)
        output.append(replace(
            result, bh_adjusted_p=bh[pindex], by_adjusted_p=by[pindex],
            adjustment_method=adjustment, alpha=alpha,
            significant=significant))
        pindex += 1
    return output


def _write_matrix(path, candidates, samples, matrix):
    with open(path, "w") as handle:
        header = ["loopId", "chromX", "startX", "endX", "chromY",
                  "startY", "endY"] + [sample.sample for sample in samples]
        handle.write("\t".join(header) + "\n")
        for candidate, values in zip(candidates, matrix):
            row = [candidate.loop_id, candidate.chrom_x, candidate.x_start,
                   candidate.x_end, candidate.chrom_y, candidate.y_start,
                   candidate.y_end] + list(values)
            handle.write("\t".join(map(str, row)) + "\n")
    return path


def write_trans_diff_results(path, results):
    fields = list(TransDiffResult.__dataclass_fields__)
    with open(path, "w") as handle:
        handle.write("#cLoops2_trans_diff_schema=1\n")
        handle.write("\t".join(fields) + "\n")
        for result in results:
            handle.write("\t".join(
                "NA" if getattr(result, field) is None else
                str(getattr(result, field)) for field in fields) + "\n")
    return path


def _r_literal(value):
    """Return a JSON/R-compatible quoted scalar string."""
    return json.dumps(str(value))


def _write_edger(path, count_path, sample_path, exposure_path, result_path,
                 reference, contrast):
    """Write an edgeR quasi-likelihood script using the exact exposures."""
    lines = [
        "#!/usr/bin/env Rscript",
        "suppressPackageStartupMessages(library(edgeR))",
        "counts <- read.delim(%s, check.names=FALSE)" %
        _r_literal(os.path.abspath(count_path)),
        "samples <- read.delim(%s, stringsAsFactors=FALSE)" %
        _r_literal(os.path.abspath(sample_path)),
        "exposure <- read.delim(%s, check.names=FALSE)" %
        _r_literal(os.path.abspath(exposure_path)),
        "coord.cols <- 1:7",
        "count.mat <- as.matrix(counts[, -coord.cols, drop=FALSE])",
        "exposure.mat <- as.matrix(exposure[, -coord.cols, drop=FALSE])",
        "if (!identical(colnames(count.mat), samples$sample) || "
        "!identical(colnames(exposure.mat), samples$sample)) "
        "stop('sample order differs between matrices and design')",
        "if (any(!is.finite(exposure.mat)) || any(exposure.mat <= 0)) "
        "stop('edgeR QL requires finite positive exposure for every candidate/sample')",
        "rownames(count.mat) <- counts$loopId",
        "rownames(exposure.mat) <- exposure$loopId",
        "group <- factor(samples$condition, levels=c(%s, %s))" %
        (_r_literal(reference), _r_literal(contrast)),
        "design <- model.matrix(~group)",
        "y <- DGEList(counts=count.mat, group=group)",
        "y$offset <- log(exposure.mat)",
        "y <- estimateDisp(y, design, robust=TRUE)",
        "fit <- glmQLFit(y, design, robust=TRUE)",
        "test <- glmQLFTest(fit, coef=2)",
        "tab <- topTags(test, n=Inf, sort.by='none')$table",
        "tab$loopId <- rownames(tab)",
        "write.table(tab, %s, sep='\\t', quote=FALSE, row.names=FALSE)" %
        _r_literal(os.path.abspath(result_path)),
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def call_trans_diff(sample_sheet, candidate_file, output, candidate_source,
                    reference=None, contrast=None, method="auto",
                    offset_scope="global", adjustment="BH", alpha=0.05,
                    prior_df=10.0):
    """Count, test and export a complete trans differential analysis."""
    output = str(output).strip()
    if not output:
        raise ValueError("output prefix must be non-empty")
    if not candidate_source or not str(candidate_source).strip():
        raise ValueError("candidate_source declaration is required")
    paths = {
        "counts": output + "_trans_counts.tsv",
        "pair_counts": output + "_trans_pair_counts.tsv",
        "reads_a": output + "_trans_readsA.tsv",
        "reads_b": output + "_trans_readsB.tsv",
        "exposures": output + "_trans_exposures.tsv",
        "samples": output + "_trans_samples.tsv",
        "results": output + "_trans_dloops.tsv",
        "edger_script": output + "_trans_edgeR.R",
        "edger_results": output + "_trans_edgeR_results.tsv",
    }
    existing = [path for key, path in paths.items()
                if key != "edger_results" and os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "trans differential output exists: %s" % existing[0])
    samples, condition_order = read_trans_sample_sheet(sample_sheet)
    reference = condition_order[0] if reference is None else reference
    contrast = condition_order[1] if contrast is None else contrast
    candidates = read_trans_diff_candidates(candidate_file)
    matrices = count_trans_candidates(samples, candidates, offset_scope)
    results = differential_test(
        candidates, samples, matrices, reference, contrast, method=method,
        offset_scope=offset_scope, adjustment=adjustment, alpha=alpha,
        prior_df=prior_df, candidate_source=candidate_source)

    parent = os.path.dirname(os.path.abspath(output))
    os.makedirs(parent, exist_ok=True)
    _write_matrix(paths["counts"], candidates, samples, matrices["counts"])
    _write_matrix(paths["pair_counts"], candidates, samples,
                  matrices["pair_counts"])
    _write_matrix(paths["reads_a"], candidates, samples,
                  matrices["reads_a"])
    _write_matrix(paths["reads_b"], candidates, samples,
                  matrices["reads_b"])
    _write_matrix(paths["exposures"], candidates, samples,
                  matrices["exposures"])
    with open(paths["samples"], "w") as handle:
        handle.write("sample\tcondition\tdirectory\tlibraryDepth\n")
        for sample, depth in zip(samples, matrices["library_depths"]):
            handle.write("%s\t%s\t%s\t%s\n" % (
                sample.sample, sample.condition, sample.directory, int(depth)))
    write_trans_diff_results(paths["results"], results)
    _write_edger(paths["edger_script"], paths["counts"], paths["samples"],
                 paths["exposures"], paths["edger_results"], reference,
                 contrast)
    return results, paths
