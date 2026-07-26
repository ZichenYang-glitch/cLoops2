#!/usr/bin/env python3
#--coding:utf-8 --
"""
callTransLoops.py
"""

#sys
import os
import sys
import json
import csv
import hashlib
from glob import glob
from datetime import datetime
from collections import Counter

#3rd
import joblib
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn import linear_model
from scipy.stats import hypergeom, binom, poisson, combine_pvalues

#cLoops
from cLoops2.ds import Loop, XY, TransContactIndex
from cLoops2.io import parseIxy, ixy2pet, loops2juiceTxt, loops2washuTxt, ixy2pet, updateJson, loops2txt
from cLoops2.geo import checkLoopOverlap, combineLoops
from cLoops2.settings import *
from cLoops2.blockDBSCAN import blockDBSCAN as DBSCAN
from cLoops2.callCisLoops import getPerRegions, selSigLoops, estAnchorSig
from cLoops2.metadata import (CAP_FORMAL_INFERENCE,
                              build_library_context)
from cLoops2.trans_stats import (
    TransCandidate,
    describe_de_novo_candidates,
    test_fixed_candidates,
    test_split_validation_candidates,
    write_trans_loop_results,
)


def _run_trans_dbscan_matrix(mat, chrom_x, chrom_y, eps, minPts,
                             source_label="in-memory trans record"):
    """Run the trans candidate geometry on an explicit two-column matrix.

    Keeping this primitive independent of ``.ixy`` paths is essential for
    discovery/validation inference: discovery rows can be clustered in memory
    without ever materializing or exposing held-out validation rows.
    """
    if eps <= 0 or minPts <= 0:
        raise ValueError("trans DBSCAN eps and minPts must be positive")
    if not isinstance(chrom_x, str) or not chrom_x:
        raise ValueError("chrom_x must be a non-empty string")
    if not isinstance(chrom_y, str) or not chrom_y:
        raise ValueError("chrom_y must be a non-empty string")
    if chrom_x == chrom_y:
        raise ValueError("trans DBSCAN requires different chromosome axes")
    mat = np.asarray(mat)
    if (mat.ndim != 2 or mat.shape[1] != 2 or
            (mat.size > 0 and not np.issubdtype(mat.dtype, np.integer))):
        raise ValueError("trans DBSCAN requires an integer (n, 2) matrix")
    loops, loopReads = [], []
    if mat.shape[0] < minPts:
        return []
    # Give every PET an integer ID without converting genomic coordinates to
    # float64.  Stable IDs plus sorted labels make candidate output
    # deterministic for a fixed input row order.
    mat = np.column_stack((np.arange(mat.shape[0], dtype=np.int64),
                           np.asarray(mat, dtype=np.int64)))

    #data for interaction records, read for readId
    report = "%s \t Clustering %s and %s using eps %s, minPts %s\n" % (
        datetime.now(), chrom_x, chrom_y, eps, minPts)
    sys.stderr.write(report)
    db = DBSCAN(mat, eps, minPts)
    labels = db.labels
    nlabels = sorted(set(labels.values()))
    #collect clusters
    for label in nlabels:
        los = sorted([int(index) for index, value in labels.items()
                      if value == label])
        loopReads.extend(los)
        sub = mat[np.asarray(los, dtype=np.int64), 1:]
        if (int(np.min(sub[:, 0])) == int(np.max(sub[:, 0])) or
                int(np.min(sub[:, 1])) == int(np.max(sub[:, 1]))):
            continue
        #define loops
        loop = Loop()
        loop.rab = int(sub.shape[0])
        loop.chromX = chrom_x
        loop.chromY = chrom_y
        loop.x_start = int(np.min(sub[:, 0]))
        loop.x_end = int(np.max(sub[:, 0]))
        loop.x_center = (loop.x_start + loop.x_end) / 2
        loop.y_start = int(np.min(sub[:, 1]))
        loop.y_end = int(np.max(sub[:, 1]))
        loop.y_center = (loop.y_start + loop.y_end) / 2
        loop.distance = -1
        loop.cis = False
        loops.append(loop)
    report = "%s \t Clustering %s and %s finished. Estimated %s reads for %s candidate loops. \n" % (
        datetime.now(), chrom_x, chrom_y, len(loopReads), len(loops))
    sys.stderr.write(report)
    return loops


def runTransDBSCANLoops(fixy, eps, minPts):
    """
    Run DBSCAN to detect interactions for one .ixy file.
    @param fixy: str, .ixy file name
    @param eps: int, eps for DBSCAN
    @param minPts: int, minPts for DBSCAN
    """
    key, mat = parseIxy(fixy, cut=0)
    if len(key) != 2 or key[0] == key[1]:
        raise ValueError("runTransDBSCANLoops requires one trans .ixy file")
    loops = _run_trans_dbscan_matrix(
        mat, key[0], key[1], eps, minPts, source_label=fixy)
    return "-".join(key), loops


#related
def parallelRunTransDBSCANLoops(meta, eps, minPts, cpu=1):
    """
    Paralle version of runCisDBSCANLoops
    @param meta: meta information parsed form petMeta.json
    @param eps: int, eps for DBSCAN
    @param minPts: int, minPts for DBSCAN
    """
    keys = sorted(meta.get("data", {}).get("trans", {}))
    ds = Parallel(n_jobs=cpu,backend="multiprocessing")(delayed(runTransDBSCANLoops)(
        meta["data"]["trans"][key]["ixy"], eps, minPts)
                              for key in keys)
    loops = {}
    for d in ds:
        if d is not None and len(d[1]) > 0:
            key, di = d[0], d[1]
            loops[key] = di
    return loops


def estLoopSig(key,
               loops,
               fixy,
               minPts=5,
               pseudo=1,
               peakPcut=1e-5,
               peakFccut=2,
               countDiffCut=10):
    """
    Estimate the loop statstical significance for one chromosomal.
    @param loops: list of Loop object
    @param fixy: cLoops2 pre generated .ixy file
    @param minPts: int, minPts
    """
    xy = ixy2pet(fixy, cut=0)
    N = xy.number
    print("%s \t Estimate significance for %s candidate interactions in %s." %
          (datetime.now(), len(loops), key))
    nloops = []
    for loop in tqdm(loops):
        ra, rb, rab = xy.queryLoop(loop.x_start, loop.x_end, loop.y_start,
                                   loop.y_end)
        ra, rb, rab = len(ra), len(rb), len(rab)
        if rab < minPts:
            continue
        loop.ra = ra
        loop.rb = rb
        loop.rab = rab
        #unbalanced anchor density, to avoid lines, unknow reason for lines, maybe stripes
        if ra / float(rb) > countDiffCut or rb / float(ra) > countDiffCut:
            continue
        if (loop.x_end -
                loop.x_start) / (loop.y_end - loop.y_start) > countDiffCut or (
                    loop.y_end - loop.y_start) / (loop.x_end -
                                                  loop.x_start) > countDiffCut:
            continue
        lowerra, lowerrb, lowerrab = xy.queryLoop(
            loop.x_start - (loop.x_end - loop.x_start), loop.x_start,
            loop.y_start - (loop.y_end - loop.y_start), loop.y_start)  #p2ll
        loop.P2LL = float(rab) / max(len(lowerrab), pseudo)
        px, esx = estAnchorSig(xy, loop.x_start, loop.x_end)
        py, esy = estAnchorSig(xy, loop.y_start, loop.y_end)
        loop.x_peak_poisson_p_value = px
        loop.x_peak_es = esx
        loop.y_peak_poisson_p_value = py
        loop.y_peak_es = esy
        #hypergeometric p-value
        hyp = max([1e-300, hypergeom.sf(rab - 1.0, N, ra, rb)])
        #start caculate the permutated background
        rabs, nbps = [], []
        nas, nbs = getPerRegions(loop, xy)
        for na in nas:
            nac = float(len(na))
            for nb in nbs:
                nbc = float(len(nb))
                nrab = float(len(na.intersection(nb)))
                #collect the value for poisson test
                rabs.append(nrab)
                #collect the possibility for following binomial test
                if nac > 0 and nbc > 0:
                    den = nrab / (nac * nbc)
                    nbps.append(den)
                else:
                    nbps.append(0.0)
        rabs, nbps = np.array(rabs), np.array(nbps)
        mrabs = float(np.mean(rabs))
        mbps = np.mean(nbps)
        #local fdr
        fdr = len(rabs[rabs > rab]) / float(len(rabs))
        #enrichment score
        es = rab / max(mrabs, pseudo)
        #simple possion test
        pop = max([1e-300, poisson.sf(rab - 1.0, mrabs)])
        #simple binomial test
        nbp = max([
            1e-300, binom.sf(rab - 1.0, ra * rb, mbps)
        ])  #the p-value is quit similar to that of cLoops 1 binomial test
        loop.FDR = fdr
        loop.ES = es
        loop.density = float(
            loop.rab) / (loop.x_end - loop.x_start + loop.y_end -
                         loop.y_start) / N * 10.0**9
        loop.hypergeometric_p_value = hyp
        loop.poisson_p_value = pop
        loop.binomial_p_value = nbp
        nloops.append(loop)
        #print(ra,rb,rab,mrabs,es,fdr,hyp,pop,nbp,n,nbp2)
    return key, nloops


def markSigLoops(key, loops):
    """
    Mark the significance of different loops.
    """
    sig = lambda x: True if x.binomial_p_value <= 1e-10 and x.FDR <= 0.05 and loop.ES >= 2 else False
    for loop in loops:
        if sig(loop):
            loop.significant = 1
        else:
            loop.significant = 0
    return key, loops


def _removed_legacy_trans_inference(*args, **kwargs):
    """Hard-stop the historical axis-mixing trans significance API.

    The symbols remain importable for a clear compatibility error, but can no
    longer emit formal-looking p-values that bypass the v2 provenance and
    fixed-candidate safeguards.
    """
    raise RuntimeError(
        "legacy trans significance used axis-mixed XY queries and has been "
        "disabled; use callTransLoops() or cLoops2.trans_stats instead")


# Preserve import compatibility while making the invalid implementation
# unreachable to external Python callers.
estLoopSig = _removed_legacy_trans_inference
markSigLoops = _removed_legacy_trans_inference


def _load_ixy_matrix(path, mmap=True):
    """Load and validate one cLoops2 coordinate matrix."""
    try:
        mat = joblib.load(path, mmap_mode="r" if mmap else None)
    except Exception:
        mat = joblib.load(path)
    if (not isinstance(mat, np.ndarray) or mat.ndim != 2 or
            mat.shape[1] != 2 or
            (mat.size > 0 and not np.issubdtype(mat.dtype, np.integer))):
        raise ValueError("ixy file %s must contain an integer (n, 2) array" %
                         path)
    return mat


def _trans_resources(predir):
    """Return validated metadata context and structured trans records.

    Only row counts are read for cis files.  Trans indexes are built lazily by
    the caller for the chromosome pairs it actually needs.
    """
    meta_path = os.path.join(predir, "petMeta.json")
    with open(meta_path) as handle:
        meta = json.load(handle)
    data = meta.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("petMeta.json is missing data")
    actual_counts = {}
    trans_records = {}
    seen_record_ids = set()
    for category in ("cis", "trans"):
        entries = data.get(category, {})
        if not isinstance(entries, dict):
            raise ValueError("petMeta data.%s must be an object" % category)
        for key in sorted(entries):
            entry = entries[key]
            if not isinstance(entry, dict) or not entry.get("ixy"):
                raise ValueError("metadata record %s:%s has no ixy path" %
                                 (category, key))
            path = entry["ixy"]
            mat = _load_ixy_matrix(path)
            record_id = entry.get("record_id", "%s:%s" % (category, key))
            record_id = str(record_id)
            if record_id in seen_record_ids:
                raise ValueError("duplicate metadata record_id %r" % record_id)
            seen_record_ids.add(record_id)
            actual_counts[record_id] = int(mat.shape[0])
            del mat
            if category != "trans":
                continue
            chrom_x = entry.get("chromX")
            chrom_y = entry.get("chromY")
            if chrom_x is None or chrom_y is None:
                axes = key.split("-")
                if len(axes) != 2:
                    raise ValueError(
                        "ambiguous trans key %r; chromX/chromY metadata is "
                        "required" % key)
                chrom_x, chrom_y = axes
            pair = (str(chrom_x), str(chrom_y))
            if pair[0] == pair[1]:
                raise ValueError("trans metadata record has identical axes")
            if pair in trans_records:
                raise ValueError("duplicate trans chromosome pair %s,%s" % pair)
            trans_records[pair] = {
                "key": key,
                "record_id": record_id,
                "path": path,
                "chromX": pair[0],
                "chromY": pair[1],
            }
    context = build_library_context(meta, actual_counts=actual_counts)
    if context.validity == "invalid":
        raise ValueError("invalid PET metadata: %s" %
                         ("; ".join(context.reasons) or "unknown reason"))

    # A globally sampled pair can be a known member of the retained universe
    # yet receive zero rows, in which case no physical .ixy is written.  Keep
    # that pair in fixed-candidate families as an explicit empty index.  Do
    # not manufacture omitted categories from a cis-only projection.
    sampling = meta.get("Sampling")
    if (isinstance(sampling, dict) and
            sampling.get("emit", sampling.get("mode")) in ("trans", "all")):
        for record in sampling.get("records", []):
            if (not isinstance(record, dict) or
                    record.get("category") != "trans" or
                    int(record.get("selected_rows", -1)) != 0 or
                    int(record.get("written_rows", -1)) != 0):
                continue
            record_id = str(record.get("record_id", ""))
            chrom_x, chrom_y = record.get("chromX"), record.get("chromY")
            if not record_id or chrom_x is None or chrom_y is None:
                raise ValueError("invalid zero-row trans Sampling record")
            pair = (str(chrom_x), str(chrom_y))
            if pair in trans_records:
                continue
            if record_id in seen_record_ids:
                raise ValueError("duplicate metadata record_id %r" % record_id)
            seen_record_ids.add(record_id)
            trans_records[pair] = {
                "key": record.get("key", record_id.split(":", 1)[-1]),
                "record_id": record_id,
                "path": None,
                "chromX": pair[0],
                "chromY": pair[1],
                "known_zero": True,
            }
    if not trans_records:
        raise ValueError(
            "no retained trans PET records; rerun cLoops2 pre -trans and "
            "ensure the chromosome whitelist contains both PET ends")
    return meta, context, trans_records


def _index_for_record(record):
    if record.get("known_zero") and record.get("path") is None:
        return TransContactIndex([], [])
    mat = _load_ixy_matrix(record["path"])
    try:
        return TransContactIndex.from_matrix(mat)
    finally:
        del mat


def _split_record_indices(number, split_seed, record_id,
                          validation_fraction=0.5):
    """Return deterministic, disjoint discovery and validation row ids.

    A SHA-256 digest of ``record_id`` is mixed with the user seed through
    ``SeedSequence``.  Consequently adding/removing/reordering another
    chromosome pair does not change this record's partition, and worker
    scheduling cannot share or duplicate a mutable Generator.
    """
    if isinstance(number, bool) or not isinstance(number, (int, np.integer)):
        raise TypeError("number must be a non-negative integer")
    number = int(number)
    if number < 0:
        raise ValueError("number must be a non-negative integer")
    if (isinstance(split_seed, bool) or
            not isinstance(split_seed, (int, np.integer))):
        raise TypeError("split_seed must be a non-negative integer")
    split_seed = int(split_seed)
    if split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string")
    if (isinstance(validation_fraction, bool) or
            not isinstance(validation_fraction, (int, float))):
        raise TypeError("validation_fraction must be numeric")
    validation_fraction = float(validation_fraction)
    if (not np.isfinite(validation_fraction) or
            not 0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in (0, 1)")

    if number == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    digest_words = [
        int.from_bytes(digest[offset:offset + 4], "little")
        for offset in range(0, 32, 4)
    ]
    rng = np.random.default_rng(
        np.random.SeedSequence([split_seed] + digest_words))

    # Half-up rounding avoids Python's bankers-rounding ambiguity.  For a
    # record with at least two rows, both partitions are guaranteed non-empty.
    # A singleton is assigned to validation: it cannot form a two-dimensional
    # DBSCAN candidate, but remains honest held-out evidence.
    validation_n = int(
        np.floor(number * validation_fraction + 0.5))
    if number == 1:
        validation_n = 1
    else:
        validation_n = min(number - 1, max(1, validation_n))
    permutation = rng.permutation(number)
    validation = np.sort(permutation[:validation_n], kind="mergesort")
    discovery = np.sort(permutation[validation_n:], kind="mergesort")
    return (np.asarray(discovery, dtype=np.int64),
            np.asarray(validation, dtype=np.int64))


def _split_record_matrix(record, split_seed, validation_fraction):
    """Load one record and return source-order discovery/validation matrices."""
    if record.get("known_zero") and record.get("path") is None:
        mat = np.empty((0, 2), dtype=np.int64)
    else:
        mat = _load_ixy_matrix(record["path"])
    try:
        discovery_ids, validation_ids = _split_record_indices(
            int(mat.shape[0]), split_seed, record["record_id"],
            validation_fraction)
        discovery = np.asarray(mat[discovery_ids, :], dtype=np.int64)
        validation = np.asarray(mat[validation_ids, :], dtype=np.int64)
    finally:
        del mat
    return discovery, validation, discovery_ids, validation_ids


def _discover_matrix_candidates(mat, record, eps, minPts):
    """Run a deterministic parameter grid and deduplicate exact rectangles."""
    eps_values = sorted(set(int(value) for value in eps))
    min_values = sorted(set(int(value) for value in minPts), reverse=True)
    if not eps_values or any(value <= 0 for value in eps_values):
        raise ValueError("split-validation eps values must be positive")
    if not min_values or any(value <= 0 for value in min_values):
        raise ValueError("split-validation minPts values must be positive")
    by_geometry = {}
    for ep in eps_values:
        for minimum in min_values:
            for loop in _run_trans_dbscan_matrix(
                    mat, record["chromX"], record["chromY"], ep, minimum,
                    source_label=record["record_id"]):
                geometry = (loop.chromX, loop.x_start, loop.x_end,
                            loop.chromY, loop.y_start, loop.y_end)
                by_geometry.setdefault(geometry, loop)
    return [by_geometry[key] for key in sorted(by_geometry)]


def _write_split_provenance(path, split_seed, validation_fraction, eps,
                            minPts, test_scope, adjustment, alpha,
                            record_splits, candidate_count):
    """Write machine-readable provenance for strict sample-split inference."""
    payload = {
        "schema_version": 1,
        "inference_mode": "formal_split_validation",
        "split_algorithm": "sha256-record-seed-pcg64-permutation-v1",
        "split_seed": int(split_seed),
        "validation_fraction": float(validation_fraction),
        "eps": list(map(int, eps)),
        "minPts": list(map(int, minPts)),
        "test_scope": test_scope,
        "adjustment": str(adjustment).upper(),
        "alpha": float(alpha),
        "candidate_count": int(candidate_count),
        "record_splits": list(record_splits),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _candidate_from_loop(loop, record, loop_id, source):
    return TransCandidate(
        loop_id=loop_id,
        chrom_x=record["chromX"],
        x_start=int(loop.x_start),
        x_end=int(loop.x_end),
        chrom_y=record["chromY"],
        y_start=int(loop.y_start),
        y_end=int(loop.y_end),
        record_id=record["record_id"],
        candidate_source=source,
    )


def parseFixedTransCandidates(path, trans_records, candidate_source,
                              skip_cis=False):
    """Read fixed rectangles and map them to structured metadata records.

    The first eight columns follow the common cLoops2 loop layout.  A v2
    result marker/header or a conventional single header are both accepted.
    Reversed external chromosome pairs are canonicalized by swapping both
    chromosome names and their coordinate intervals.
    """
    if not candidate_source or not str(candidate_source).strip():
        raise ValueError(
            "formal trans testing requires a non-empty candidate source "
            "declaration independent of the current X-Y pairings")
    candidates = []
    with open(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#") or row[0] == "loopId":
                continue
            if len(row) < 7:
                raise ValueError("candidate line %s has fewer than 7 columns" %
                                 line_number)
            chrom_x, chrom_y = row[1], row[4]
            if chrom_x == chrom_y:
                if skip_cis:
                    continue
                raise ValueError(
                    "candidate %s is cis; formal trans candidate files must "
                    "contain only inter-chromosomal rectangles" % row[0])
            try:
                x_start, x_end = int(float(row[2])), int(float(row[3]))
                y_start, y_end = int(float(row[5])), int(float(row[6]))
            except (OverflowError, ValueError):
                raise ValueError("candidate line %s has invalid coordinates" %
                                 line_number)
            pair = (chrom_x, chrom_y)
            record = trans_records.get(pair)
            if record is None:
                reverse = (chrom_y, chrom_x)
                record = trans_records.get(reverse)
                if record is not None:
                    chrom_x, chrom_y = chrom_y, chrom_x
                    x_start, x_end, y_start, y_end = (
                        y_start, y_end, x_start, x_end)
            if record is None:
                raise ValueError(
                    "candidate %s references unavailable chromosome pair "
                    "%s,%s" % (row[0], pair[0], pair[1]))
            candidates.append(
                TransCandidate(
                    loop_id=row[0],
                    chrom_x=chrom_x,
                    x_start=x_start,
                    x_end=x_end,
                    chrom_y=chrom_y,
                    y_start=y_start,
                    y_end=y_end,
                    record_id=record["record_id"],
                    candidate_source=str(candidate_source).strip(),
                ))
    return candidates


def _write_trans_washu(results, path, formal=False):
    """Write directionally correct WashU records without legacy p columns."""
    with open(path, "w") as handle:
        for result in results:
            if formal and result.significant is not True:
                continue
            handle.write("%s:%s-%s\t%s:%s-%s\t%s\n" % (
                result.chrom_x, result.x_start, result.x_end,
                result.chrom_y, result.y_start, result.y_end, result.pets))


def _write_trans_juicebox(results, path, formal=False):
    """Write a minimal Juicebox 2D annotation with honest v2 fields."""
    with open(path, "w") as handle:
        handle.write(
            "chromosome1\tx1\tx2\tchromosome2\ty1\ty2\tcolor\tobserved\t"
            "loopId\tinferenceMode\tadjustedP\n")
        for result in results:
            if formal and result.significant is not True:
                continue
            adjusted = (result.bh_adjusted_p
                        if result.adjustment_method == "BH" else
                        result.by_adjusted_p)
            handle.write("\t".join(map(str, (
                result.chrom_x, result.x_start, result.x_end,
                result.chrom_y, result.y_start, result.y_end,
                '"0,255,255"', result.pets, result.loop_id,
                result.inference_mode,
                "NA" if adjusted is None else adjusted,
            ))) + "\n")


def callTransLoops(
        predir,
        fout,
        logger,
        eps=[2000, 5000],
        minPts=[5, 10],
        cpu=1,
        filter=False,
        washU=False,
        juicebox=False,
        candidate_file=None,
        candidate_source=None,
        local_pad=None,
        test_scope="both",
        adjustment="BH",
        alpha=0.05,
        split_validation=False,
        split_seed=None,
        validation_fraction=0.5,
):
    """Discover exploratory, test fixed, or sample-split trans rectangles.

    With no ``candidate_file`` this function runs the historical blockDBSCAN
    geometry as a *candidate generator*.  The resulting conditional tails are
    explicitly exploratory: adjusted p-values and significance are ``NA``
    because the same PETs selected and scored the rectangles.

    Supplying ``candidate_file`` switches to formal fixed-candidate testing.
    The caller must also provide a non-empty ``candidate_source`` declaration
    establishing that the rectangles are independent of the current X--Y
    pairings.  Sampling provenance with replacement is rejected by the formal
    statistics API.

    ``split_validation=True`` is the internally discoverable formal mode.  A
    deterministic, record-specific random partition is made from every trans
    matrix.  DBSCAN sees discovery rows only; all counts, p-values, and the
    single family-wide BH/BY correction use the disjoint validation rows only.
    ``split_seed`` is required and ``candidate_file`` is mutually exclusive.
    """
    meta, context, trans_records = _trans_resources(predir)
    fixed_formal = bool(candidate_file)
    split_formal = bool(split_validation)
    if fixed_formal and split_formal:
        raise ValueError(
            "candidate_file and split_validation are mutually exclusive")
    if split_formal and split_seed is None:
        raise ValueError("split_seed is required for split_validation")
    if not split_formal and split_seed is not None:
        raise ValueError("split_seed requires split_validation=True")
    if split_formal and candidate_source:
        raise ValueError(
            "candidate_source is only valid with an external candidate_file")
    split_seed_value = None
    if split_formal:
        if split_seed is None:
            raise RuntimeError("validated split mode has no seed")
        split_seed_value = int(split_seed)
    formal = fixed_formal or split_formal
    if local_pad is None:
        # Formal local windows must not depend on DBSCAN parameters that the
        # fixed-candidate path does not use.
        local_pad = (5000 if fixed_formal else
                     max([int(value) for value in eps] or [1]) * 5)
    if local_pad < 0:
        raise ValueError("local_pad must be non-negative")
    if filter:
        logger.warning(
            "-filter is not applied to trans v2 results yet; no PET output "
            "will be written by this option.")

    split_provenance = None
    record_splits = []
    if fixed_formal:
        # Fail before loading/sorting potentially large chromosome-pair files.
        context.require(CAP_FORMAL_INFERENCE)
        candidates = parseFixedTransCandidates(candidate_file, trans_records,
                                                candidate_source)
        if not candidates:
            raise ValueError("formal trans candidate file contains no candidates")
        needed = sorted(set(candidate.record_id for candidate in candidates))
        records_by_id = {
            record["record_id"]: record for record in trans_records.values()
        }
        indexes = {
            record_id: _index_for_record(records_by_id[record_id])
            for record_id in needed
        }
        results = test_fixed_candidates(
            candidates,
            indexes,
            context,
            local_pad=local_pad,
            test_scope=test_scope,
            adjustment=adjustment,
            alpha=alpha,
        )
        output = fout + "_trans_loops.txt"
        logger.info("Writing %s formally tested trans candidates to %s." %
                    (len(results), output))
    elif split_formal:
        # Fail before constructing DBSCAN matrices or sorted validation indexes.
        if split_seed_value is None:
            raise RuntimeError("validated split seed is unavailable")
        context.require(CAP_FORMAL_INFERENCE)
        candidate_entries = []
        validation_indexes = {}
        discovery_counts = {}
        for pair in sorted(trans_records):
            record = trans_records[pair]
            discovery, validation, discovery_ids, validation_ids = (
                _split_record_matrix(record, split_seed_value,
                                     validation_fraction))
            source_rows = context.get_npair(record["record_id"])
            if discovery.shape[0] + validation.shape[0] != source_rows:
                raise ValueError(
                    "split row closure failed for %s" % record["record_id"])
            record_splits.append({
                "record_id": record["record_id"],
                "chromX": record["chromX"],
                "chromY": record["chromY"],
                "source_rows": int(source_rows),
                "discovery_rows": int(discovery.shape[0]),
                "validation_rows": int(validation.shape[0]),
                "discovery_index_sha256": hashlib.sha256(
                    np.asarray(discovery_ids, dtype="<i8").tobytes()).hexdigest(),
                "validation_index_sha256": hashlib.sha256(
                    np.asarray(validation_ids, dtype="<i8").tobytes()).hexdigest(),
            })
            loops_for_record = _discover_matrix_candidates(
                discovery, record, eps, minPts)
            if loops_for_record:
                record_id = record["record_id"]
                validation_indexes[record_id] = TransContactIndex.from_matrix(
                    validation)
                discovery_counts[record_id] = int(discovery.shape[0])
                for loop in loops_for_record:
                    candidate_entries.append((record, loop,
                                              int(discovery.shape[0]),
                                              int(validation.shape[0])))
            del discovery, validation, discovery_ids, validation_ids

        candidate_entries.sort(key=lambda item: (
            item[0]["chromX"], item[0]["chromY"], item[1].x_start,
            item[1].x_end, item[1].y_start, item[1].y_end))
        candidates = []
        for number, (record, loop, discovery_n, validation_n) in enumerate(
                candidate_entries):
            source = (
                "split_validation:blockDBSCAN;seed=%s;validation_fraction=%s;"
                "discovery_pair_pets=%s;validation_pair_pets=%s;eps=%s;"
                "minPts=%s" %
                (split_seed_value, repr(float(validation_fraction)), discovery_n,
                 validation_n, ",".join(map(str, sorted(set(eps)))),
                 ",".join(map(str, sorted(set(minPts), reverse=True)))))
            candidates.append(
                _candidate_from_loop(
                    loop, record, "trans_split_candidate_%s" % number, source))

        results = test_split_validation_candidates(
            candidates,
            validation_indexes,
            discovery_counts,
            context,
            validation_fraction=validation_fraction,
            local_pad=local_pad,
            test_scope=test_scope,
            adjustment=adjustment,
            alpha=alpha,
        )
        output = fout + "_trans_loops.txt"
        split_provenance = fout + "_trans_split.json"
        logger.info(
            "Writing %s split-validation trans loops to %s; discovery and "
            "validation PETs were disjoint for every chromosome pair." %
            (len(results), output))
    else:
        loops = {}
        for ep in eps:
            for minPt in minPts:
                discovered = parallelRunTransDBSCANLoops(
                    meta, ep, minPt, cpu=cpu)
                loops = combineLoops(loops, discovered)

        candidates_by_record = {}
        counter = 0
        for key in sorted(loops):
            ordered = sorted(
                loops[key],
                key=lambda loop: (loop.chromX, loop.x_start, loop.x_end,
                                  loop.chromY, loop.y_start, loop.y_end),
            )
            for loop in ordered:
                record = trans_records.get((loop.chromX, loop.chromY))
                if record is None:
                    raise ValueError(
                        "DBSCAN candidate references unknown trans pair %s,%s" %
                        (loop.chromX, loop.chromY))
                loop_id = "trans_candidate_%s" % counter
                counter += 1
                candidate = _candidate_from_loop(
                    loop,
                    record,
                    loop_id,
                    "de_novo:blockDBSCAN;eps=%s;minPts=%s" %
                    (",".join(map(str, eps)), ",".join(map(str, minPts))),
                )
                candidates_by_record.setdefault(record["record_id"],
                                                []).append(candidate)

        records_by_id = {
            record["record_id"]: record for record in trans_records.values()
        }
        results = []
        # No family correction occurs in exploratory mode, so indexes can be
        # constructed and released one chromosome pair at a time.
        for record_id in sorted(candidates_by_record):
            index = _index_for_record(records_by_id[record_id])
            results.extend(
                describe_de_novo_candidates(
                    candidates_by_record[record_id],
                    {record_id: index},
                    context,
                    local_pad=local_pad,
                    test_scope=test_scope,
                ))
        results.sort(key=lambda result: (
            result.chrom_x, result.chrom_y, result.x_start, result.y_start,
            result.loop_id))
        output = fout + "_trans_candidates.txt"
        logger.info(
            "Writing %s exploratory trans candidates to %s; adjusted p-values "
            "and significance are NA because candidate discovery and scoring "
            "used the same PETs." % (len(results), output))

    write_trans_loop_results(output, results)
    if split_provenance is not None:
        if split_seed_value is None:
            raise RuntimeError("split provenance has no validated seed")
        _write_split_provenance(
            split_provenance,
            split_seed_value,
            validation_fraction,
            sorted(set(eps)),
            sorted(set(minPts), reverse=True),
            test_scope,
            adjustment,
            alpha,
            record_splits,
            len(results),
        )
    stem = output[:-4] if output.endswith(".txt") else output
    if washU:
        _write_trans_washu(results, stem + "_washU.txt", formal=formal)
    if juicebox:
        _write_trans_juicebox(results,
                              stem + "_juicebox.txt",
                              formal=formal)
    return results
