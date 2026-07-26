#!/usr/bin/env python
#--coding:utf-8--
"""
filter.py
cLoops2 PETs filtering related code.

2020-01-29: --invert-match added.
2020-02-17: filter singleton PETs added
2020-02-19: sample PETs added
2020-04-06: filterPETsbyLoops updated as requiring both ends in loops
2020-08-24: update np.random.choice, its default replace parameter is True, there fore the sampling not well for tot<true unique reads.
"""

__author__ = "CAO Yaqiang"
__date__ = "2020-01-28"
__modified__ = ""
__email__ = "caoyaqiang0410@gmail.com"

#sys
import copy
import json
import os
from glob import glob
from datetime import datetime
from pathlib import Path

#3rd
import joblib
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

#cLoops2
from cLoops2.ds import Peak, Loop, TransContactIndex, XY
from cLoops2.io import ixy2pet, parseIxy, writeNewJson, parseBed2Peaks, parseTxt2Loops
from cLoops2.metadata import (build_library_context,
                              inspect_actual_counts)


def _load_filter_source(predir):
    """Load and structurally validate the parent of a filter transform."""
    metaf = os.path.join(predir, "petMeta.json")
    with open(metaf) as handle:
        meta = json.load(handle)
    context = build_library_context(
        meta, actual_counts=inspect_actual_counts(meta))
    if context.validity == "invalid":
        raise ValueError("input metadata is structurally invalid: %s" %
                         ("; ".join(context.reasons) or "unknown reason"))
    return meta, context


def _finalize_filter_metadata(predir, outdir, source_meta, source_context,
                              filter_type, parameters):
    """Rebuild output metadata and preserve replacement ancestry.

    Filtering is a deterministic materialized transform, but it is not a
    random sample of the parent library.  It therefore gets a usable physical
    and descriptive logical depth while remaining ineligible for formal
    inference.  A later sample must not launder that eligibility boundary.
    """
    writeNewJson(outdir)
    metaf = Path(outdir) / "petMeta.json"
    with metaf.open() as handle:
        output_meta = json.load(handle)
    physical_total = int(output_meta["Unique PETs"])
    parent_valid = source_context.validity in ("valid", "legacy_root_assumed")
    transform = {
        "schema_version": 1,
        "operation": "filter",
        "filter_type": str(filter_type),
        "parameters": copy.deepcopy(parameters),
        "validity": "valid" if parent_valid else "unknown_ancestry",
        "integrity_level": "structural",
        "logical_total": physical_total,
        "physical_output_total": physical_total,
        "replacement_ever": source_context.replacement_ever,
        "retention_compatible": "Retention" in source_meta,
        "parents_fully_materialized": bool(
            source_context.physical_total == source_context.logical_total),
        "formal_inference_eligible": False,
        "parent": {
            "directory": str(Path(predir).expanduser().resolve()),
            "physical_total": source_context.physical_total,
            "logical_total": source_context.logical_total,
            "validity": source_context.validity,
            "source_kind": source_context.source_kind,
            "emit": source_context.emit,
            "replacement_ever": source_context.replacement_ever,
        },
    }
    output_meta["Transformation"] = transform
    if "Retention" in source_meta:
        output_meta["Retention"] = copy.deepcopy(source_meta["Retention"])
    temporary = metaf.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(output_meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(metaf))
    return transform


def stichRegions(rs, gap=1):
    """
    Stich 1D regions with specified gap size. 
    """
    cov = set()
    for r in rs:
        cov.update(range(r[0], r[1] + 1))
    cov = list(cov)
    cov.sort()
    nrs = []
    i = 0
    while i < len(cov) - 1:
        for j in range(i + 1, len(cov)):
            if cov[j] - cov[j - 1] > gap:
                break
            else:
                continue
        start = cov[i]
        end = cov[j - 1]
        nrs.append([start, end])
        i = j  #update search start
    return nrs


def filterPETs(rs, key, predir, fixy, iv=False):
    """
    Filter PETs, only keep those located at regions.
    """
    print("%s\t Filtering PETs of %s with %s regions." %
          (datetime.now(), key, len(rs)))
    key2, mat = parseIxy(fixy)
    xy = XY(mat[:,0],mat[:,1])
    rids = set()
    for r in tqdm(rs):
        r = xy.queryPeak(r[0], r[1])
        rids.update(r)
    if iv:
        aids = set(np.arange(mat.shape[0]))
        rids = aids.difference(rids)
    if len(rids) == 0:
        return
    rids = sorted(rids)
    mat = mat[rids, ]
    foixy = predir + "/" + "-".join(key2) + ".ixy"
    joblib.dump(mat, foixy)


def filterPETsByPeaks(predir, fbed, outdir, cpu=1, iv=False, gap=1,
                      mode="cis"):
    """
    Filter PETs according to peaks form .bed file. 
    """
    if mode not in ("cis", "trans", "all"):
        raise ValueError("mode must be cis, trans, or all")
    meta, context = _load_filter_source(predir)
    peaks = parseBed2Peaks(fbed)
    npeaks = {}
    cis_keys = (sorted(meta["data"]["cis"]) if iv else sorted(peaks))
    for key in cis_keys:
        if key not in meta["data"]["cis"]:
            continue
        rs = [[p.start, p.end] for p in peaks.get(key, [])]
        npeaks[key] = stichRegions(rs, gap)
    if mode in ("cis", "all"):
        Parallel(n_jobs=cpu,backend="multiprocessing")(delayed(filterPETs)(
            npeaks[key],
            key,
            outdir,
            meta["data"]["cis"][key]["ixy"],
            iv,
        ) for key in sorted(npeaks))
    if mode in ("trans", "all"):
        jobs = []
        for key in sorted(meta.get("data", {}).get("trans", {})):
            entry = meta["data"]["trans"][key]
            chrom_x, chrom_y = _entry_axes("trans", key, entry)
            x_regions = [(peak.start, peak.end)
                         for peak in peaks.get(
                             "%s-%s" % (chrom_x, chrom_x), [])]
            y_regions = [(peak.start, peak.end)
                         for peak in peaks.get(
                             "%s-%s" % (chrom_y, chrom_y), [])]
            if x_regions or y_regions or iv:
                jobs.append((x_regions, y_regions, key, entry["ixy"]))
        Parallel(n_jobs=cpu, backend="multiprocessing")(
            delayed(_filterPETsByTransRegions)(
                x_regions, y_regions, key, outdir, fixy,
                iv=iv, gap=gap)
            for x_regions, y_regions, key, fixy in jobs)
    return _finalize_filter_metadata(
        predir, outdir, meta, context, "peaks", {
            "regions": str(Path(fbed).expanduser().resolve()),
            "mode": mode,
            "invert": bool(iv),
            "gap": int(gap),
        })


def _filterPETsByLoops(loops,key,predir,fixy,iv=False):
    """
    Filter PETs by loops
    """
    print("%s\t Filtering PETs of %s with %s regions." %
          (datetime.now(), key, len(loops)))
    key2, mat = parseIxy(fixy)
    xy = XY(mat[:,0],mat[:,1]) 
    rids = set()
    for loop in tqdm(loops):
        a, b,r = xy.queryLoop(loop.x_start, loop.x_end, loop.y_start, loop.y_end)
        rids.update(r)
    if iv:
        aids = set(np.arange(mat.shape[0]))
        rids = aids.difference(rids)
    if len(rids) == 0:
        return
    rids = sorted(rids)
    mat = mat[rids, ]
    foixy = predir + "/" + "-".join(key2) + ".ixy"
    joblib.dump(mat, foixy)


def _merge_filter_intervals(intervals, gap):
    merged = []
    for start, end in sorted((int(start), int(end))
                             for start, end in intervals):
        if start > end:
            raise ValueError("filter interval start exceeds end")
        if not merged or start - merged[-1][1] > int(gap):
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _filterPETsByTransRegions(x_regions, y_regions, key, outdir, fixy,
                              iv=False, gap=1):
    """Filter a trans record by chromosome-specific one-dimensional regions."""
    _, mat = parseIxy(fixy)
    selected = np.zeros(int(mat.shape[0]), dtype=bool)
    for start, end in _merge_filter_intervals(x_regions, gap):
        selected |= ((mat[:, 0] >= start) & (mat[:, 0] <= end))
    for start, end in _merge_filter_intervals(y_regions, gap):
        selected |= ((mat[:, 1] >= start) & (mat[:, 1] <= end))
    if iv:
        selected = ~selected
    ids = np.flatnonzero(selected)
    if ids.size == 0:
        return 0
    output = mat[ids, :]
    joblib.dump(output, os.path.join(outdir, key + ".ixy"))
    return int(output.shape[0])


def _filterPETsByTransLoops(rectangles, key, outdir, fixy, iv=False,
                            gap=1, both=False):
    """Filter one trans record without exchanging its X/Y axes."""
    print("%s\t Filtering trans PETs of %s with %s rectangles." %
          (datetime.now(), key, len(rectangles)))
    _, mat = parseIxy(fixy)
    index = TransContactIndex.from_matrix(mat)
    selected = set()
    if both:
        for x_start, x_end, y_start, y_end in rectangles:
            selected.update(index.query_rect_ids(
                int(x_start), int(x_end), int(y_start), int(y_end)).tolist())
    else:
        x_intervals = _merge_filter_intervals(
            [(rectangle[0], rectangle[1]) for rectangle in rectangles], gap)
        y_intervals = _merge_filter_intervals(
            [(rectangle[2], rectangle[3]) for rectangle in rectangles], gap)
        for start, end in x_intervals:
            selected.update(np.flatnonzero(
                (mat[:, 0] >= start) & (mat[:, 0] <= end)).tolist())
        for start, end in y_intervals:
            selected.update(np.flatnonzero(
                (mat[:, 1] >= start) & (mat[:, 1] <= end)).tolist())
    if iv:
        selected = set(range(int(mat.shape[0]))).difference(selected)
    if not selected:
        return 0
    ids = np.asarray(sorted(selected), dtype=np.int64)
    output = mat[ids, :]
    joblib.dump(output, os.path.join(outdir, key + ".ixy"))
    return int(output.shape[0])


def _entry_axes(category, key, entry):
    chrom_x = entry.get("chromX") or entry.get("chrom_x")
    chrom_y = entry.get("chromY") or entry.get("chrom_y")
    if chrom_x is None or chrom_y is None:
        fields = key.split("-")
        if len(fields) != 2:
            raise ValueError(
                "metadata record %s:%s lacks unambiguous chromX/chromY" %
                (category, key))
        chrom_x, chrom_y = fields
    return str(chrom_x), str(chrom_y)


def _trans_rectangles_for_record(loop_values, chrom_x, chrom_y):
    rectangles = []
    for loop in loop_values:
        if (loop.chromX, loop.chromY) == (chrom_x, chrom_y):
            rectangles.append((loop.x_start, loop.x_end,
                               loop.y_start, loop.y_end))
        elif (loop.chromX, loop.chromY) == (chrom_y, chrom_x):
            rectangles.append((loop.y_start, loop.y_end,
                               loop.x_start, loop.x_end))
    return rectangles



def filterPETsByLoops(predir, floop, outdir, cpu=1, iv=False, gap=1,
                      both=False, mode="cis"):
    """
    Filter PETs according to loops from _loop.txt file. 
    """
    if mode not in ("cis", "trans", "all"):
        raise ValueError("mode must be cis, trans, or all")
    meta, context = _load_filter_source(predir)
    loops = parseTxt2Loops(floop, mode=mode)
    if mode in ("cis", "all") and both:
        #filter PETs, only keep those both ends overlapped with target loop anchors
        Parallel(n_jobs=cpu,backend="multiprocessing")(delayed(_filterPETsByLoops)(
            loops.get(key, []),
            key,
            outdir,
            meta["data"]["cis"][key]["ixy"],
            iv,
        ) for key in sorted(meta["data"]["cis"])
        if iv or key in loops)
    elif mode in ("cis", "all"):
        #filter PETs, keep those any end overlapped with target loop anchors
        npeaks = {}
        keys = (sorted(meta["data"]["cis"]) if iv else sorted(loops))
        for key in keys:
            if key not in meta["data"]["cis"]:
                continue
            rs = []
            for loop in loops.get(key, []):
                rs.append([loop.x_start, loop.x_end])
                rs.append([loop.y_start, loop.y_end])
            npeaks[key] = stichRegions(rs, gap)
        Parallel(n_jobs=cpu,backend="multiprocessing")(delayed(filterPETs)(
            npeaks[key],
            key,
            outdir,
            meta["data"]["cis"][key]["ixy"],
            iv,
        ) for key in npeaks.keys())

    if mode in ("trans", "all"):
        loop_values = [loop for values in loops.values() for loop in values
                       if not loop.cis]
        jobs = []
        for key in sorted(meta.get("data", {}).get("trans", {})):
            entry = meta["data"]["trans"][key]
            chrom_x, chrom_y = _entry_axes("trans", key, entry)
            rectangles = _trans_rectangles_for_record(
                loop_values, chrom_x, chrom_y)
            if rectangles or iv:
                jobs.append((rectangles, key, entry["ixy"]))
        Parallel(n_jobs=cpu, backend="multiprocessing")(
            delayed(_filterPETsByTransLoops)(
                rectangles, key, outdir, fixy, iv=iv, gap=gap, both=both)
            for rectangles, key, fixy in jobs)

    return _finalize_filter_metadata(
        predir, outdir, meta, context, "loops", {
            "loops": str(Path(floop).expanduser().resolve()),
            "mode": mode,
            "invert": bool(iv),
            "both": bool(both),
            "gap": int(gap),
        })


def _filterPETsBySingletons(f, outdir, binSize):
    """
    @param f:str .ixy file
    @param outir: str,
    @param bs: int, binSize
    """
    key, mat = parseIxy(f)
    print("%s\t Filtering %s singleton PETs in contact matrix bins." %
          (datetime.now(), key))
    minC = np.min(mat)
    ss = {}
    i = 0
    for x,y in tqdm(mat):
        x = int((x - minC) / binSize)
        y = int((y - minC) / binSize)
        if x not in ss:
            ss[x] = {}
        if y not in ss[x]:
            ss[x][y] = []
        ss[x][y].append(i)
        i += 1
    rs = []
    for nx in ss.keys():
        for ny in ss[nx].keys():
            if len(ss[nx][ny]) > 1:
                rs.extend(ss[nx][ny])
    if len(rs) > 0:
        mat = mat[rs, ]
        foixy = outdir + "/" + "-".join(key) + ".ixy"
        joblib.dump(mat, foixy)


def filterPETsBySingletons(predir, outdir, bs, cpu=1):
    """
    Filter singleton PETs in contact matrix bins.
    """
    meta, context = _load_filter_source(predir)
    Parallel(n_jobs=cpu,backend="multiprocessing")(delayed(_filterPETsBySingletons)(
        meta["data"]["cis"][key]["ixy"],
        outdir,
        bs,
    ) for key in meta["data"]["cis"].keys())
    return _finalize_filter_metadata(
        predir, outdir, meta, context, "singletons", {
            "bin_size": int(bs),
        })


def _getNearbyGrids(Gs, cell):
    x, y = cell[0], cell[1]
    keys = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y), (x - 1, y - 1),
            (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]
    ncells = []
    for key in keys:
        if key in Gs:
            ncells.append(key)
    return ncells


def _filterPETsByKNNs(f, outdir,eps,minPts):
    """
    @param f:str .ixy file
    @param outir: str,
    """
    key, mat = parseIxy(f)
    print("%s\t Filtering %s PETs based on KNN." % (datetime.now(), key))
    
    #build grids
    minX, minY = np.min(mat[:,0]),np.min(mat[:,1])
    Gs = {}
    ps = {}
    for i,(x,y) in enumerate(mat):
        nx = int((x - minX) / eps) + 1
        ny = int((y - minY) / eps) + 1
        Gs.setdefault((nx, ny), []).append(i)
        #last elements marks the class, initially -1 as noise
        ps[ i ] = [x, y, nx, ny, -1]

    #Grid index with all neighbor points.
    Gs2 = {}
    for cell in Gs.keys():
        nps = []
        nps.extend(Gs[cell])
        for cellj in _getNearbyGrids(Gs,cell):
            nps.extend(Gs[cellj])
        Gs2[cell] = nps
    
    #remove noise 
    #: noise cells without neighbors
    tode = set()
    #: noise cells with neighbors
    tode2 = set()
    for cell in Gs.keys():
        if len(Gs2[cell]) < minPts:
            tode2.add(cell)
    #KNN to noise cells with neighbors
    for cell in tode2:
        cells = _getNearbyGrids(Gs,cell)
        ncells = set(cells) & tode2
        #all neighbor cells are noise
        if len(cells) == len(ncells):
            tode.add(cell)
    for cell in tode:
        for p in Gs[cell]:
            del ps[p]
        del Gs[cell]
    
    nps = []
    for cell in Gs.keys():
        nps.extend( Gs[cell] )

    if len(nps) > 0:
        mat = mat[nps,]
        foixy = outdir + "/" + "-".join(key) + ".ixy"
        joblib.dump(mat, foixy)


def filterPETsByKNNs(predir, outdir, eps=1000, minPts=5, cpu=1):
    """
    Filter PETs based on blockDBSCAN noise-removing processing.
    """
    meta, context = _load_filter_source(predir)
    Parallel(n_jobs=cpu,backend="multiprocessing")(delayed(_filterPETsByKNNs)(
        meta["data"]["cis"][key]["ixy"],
        outdir,
        eps,
        minPts,
    ) for key in meta["data"]["cis"].keys())
    return _finalize_filter_metadata(
        predir, outdir, meta, context, "knn", {
            "eps": int(eps),
            "min_points": int(minPts),
        })


def samplePETs(predir, outdir, tot, cpu=1, mode="cis", seed=None):
    """Sample the retained whole library and emit cis, trans, or all PETs.

    ``tot`` is the logical target for the complete retained library.  In
    ``cis`` and ``trans`` modes the physical output is the corresponding
    projection of that same global sample, so ``Unique PETs`` need not equal
    ``tot``.  See :mod:`cLoops2.sampling` for the allocation algorithm and
    Sampling metadata contract.

    The original four positional arguments remain valid; ``mode`` and ``seed``
    are optional additions for API compatibility.
    """
    from cLoops2.sampling import sample_pets

    return sample_pets(
        predir,
        outdir,
        tot,
        cpu=cpu,
        mode=mode,
        seed=seed,
    )
