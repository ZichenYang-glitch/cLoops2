#!/usr/bin/env python
#--coding:utf-8--
"""
cLoops2: dump
cLoops2 major file conversion functions
2020-03-04: to add new washU/UCSC support bigInteract track, according to https://genome.ucsc.edu/goldenPath/help/interact.html 
2020-06-25: to add dump to BEDPE
2020-07-01: refine ixy2bdg
2020-07-28: ixy2bed added
2021-09-28: ixy2virtual4C added
"""

__author__ = "CAO Yaqiang"
__date__ = ""
__modified__ = ""
__email__ = "caoyaqiang0410@gmail.com"

#general library
import warnings
warnings.filterwarnings("ignore")
import os
import gzip 
import json
import random
import subprocess
from glob import glob

#3rd library
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

#cLoops2
from cLoops2.ds import XY
from cLoops2.io import parseIxy
from cLoops2.cmat import (getObsMat, getExpMat, getVirtual4CSig,
                          getTransObsMat, getTransObsMatCOO)
from cLoops2.utils import isTool, callSys
from cLoops2.settings import *


def _dump_record_axes(category, key, entry):
    chrom_x = entry.get("chromX") or entry.get("chrom_x")
    chrom_y = entry.get("chromY") or entry.get("chrom_y")
    if chrom_x is None or chrom_y is None:
        fields = str(key).split("-")
        if len(fields) != 2:
            raise ValueError(
                "record %s:%s requires structured chromX/chromY metadata" %
                (category, key))
        chrom_x, chrom_y = fields
    chrom_x, chrom_y = str(chrom_x), str(chrom_y)
    if (category == "cis") != (chrom_x == chrom_y):
        raise ValueError("metadata category and chromosome axes disagree")
    return chrom_x, chrom_y


def _select_dump_records(directory, mode):
    """Return stable structured records without decoding chromosome basenames."""
    if mode not in ("cis", "trans", "all"):
        raise ValueError("mode must be cis, trans, or all")
    with open(os.path.join(directory, "petMeta.json")) as handle:
        meta = json.load(handle)
    records = []
    categories = (("cis", "trans") if mode == "all" else (mode,))
    for category in categories:
        entries = meta.get("data", {}).get(category, {})
        if not isinstance(entries, dict):
            raise ValueError("petMeta data.%s must be an object" % category)
        for key in sorted(entries):
            entry = entries[key]
            if not isinstance(entry, dict) or not entry.get("ixy"):
                raise ValueError("metadata record %s:%s has no ixy" %
                                 (category, key))
            path = str(entry["ixy"])
            if not os.path.isabs(path) and not os.path.exists(path):
                path = os.path.join(directory, path)
            chrom_x, chrom_y = _dump_record_axes(category, key, entry)
            records.append((category, str(key), chrom_x, chrom_y, path))
    if not records:
        raise ValueError("no %s PET records are available for export" % mode)
    return records


def _load_dump_record(record, cut=0, mcut=-1):
    category, key, chrom_x, chrom_y, path = record
    try:
        matrix = joblib.load(path, mmap_mode="r")
    except Exception:
        matrix = joblib.load(path)
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or
            matrix.shape[1] != 2):
        raise ValueError("ixy %s must have shape (n, 2)" % path)
    # Genomic distance is undefined across chromosomes.  cut/mcut remain the
    # historical cis filters and are deliberately ignored for trans records.
    if category == "cis" and (cut > 0 or mcut > 0):
        distance = matrix[:, 1] - matrix[:, 0]
        keep = np.ones(matrix.shape[0], dtype=bool)
        if cut > 0:
            keep &= distance >= cut
        if mcut > 0:
            keep &= distance <= mcut
        matrix = matrix[keep, :]
    return chrom_x, chrom_y, matrix


def _iter_bidirectional_intervals(records, cut=0, mcut=-1, ext=50):
    """Yield both browser-query directions with chromosomes swapped too."""
    for record in records:
        chrom_x, chrom_y, matrix = _load_dump_record(
            record, cut=cut, mcut=mcut)
        for x, y in matrix:
            x, y = int(x), int(y)
            xa, xb = max(0, x - ext), x + ext
            ya, yb = max(0, y - ext), y + ext
            yield chrom_x, xa, xb, chrom_y, ya, yb
            yield chrom_y, ya, yb, chrom_x, xa, xb


def _sort_bed_like(source, destination):
    """Disk-backed stable coordinate sort used before bgzip/bigBed tools."""
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    with open(destination, "w") as output:
        subprocess.run(
            ["sort", "-k1,1", "-k2,2n", "-k3,3n", source],
            stdout=output, check=True, env=environment)

def ixy2bed(
            d, 
            fout, 
            logger,
            cut=0, 
            mcut=-1,
            ext=50,
            mode="cis",
    ):
    """
    Convert PETs to sorted BED file.
    @param d: str,cLoops2 pre data directory
    @param fout: str,prefix of output files
    @param logger: logger
    @param cut: int, > cut PETs kept
    @param mcut: int, <mcut PETs kept
    @param ext: int, extension from the PET center
    """
    if not os.path.exists(d):
        logger.error("%s not exists. return." % d)
        return
    fout = fout + "_reads.bed.gz"
    if os.path.isfile(fout):
        logger.error("Traget output file %s has been generated, return."%fout)
        return

    logger.info("Converting %s to BED file." % d)
    records = _select_dump_records(d, mode)
    with gzip.open(fout, "wt") as f:
        for record in records:
            chrom_x, chrom_y, mat = _load_dump_record(
                record, cut=cut, mcut=mcut)
            axes = ((chrom_x, mat[:, 0]), (chrom_y, mat[:, 1]))
            if chrom_x == chrom_y:
                axes = ((chrom_x, np.concatenate((mat[:, 0], mat[:, 1]))),)
            for chrom, values in axes:
                for coordinate in tqdm(np.unique(values)):
                    line = [chrom, max(0, coordinate - ext), coordinate + ext]
                    f.write("\t".join(map(str, line)) + "\n")
    logger.info("Converting to BED file %s finished." % fout)


def ixy2bedpe(
            d, 
            fout, 
            logger,
            cut=0, 
            mcut=-1,
            ext=50,
            mode="cis",
    ):
    """
    Convert PETs to BEDPE file.
    @param d: str,cLoops2 pre data directory
    @param fout: str,prefix of output files
    @param logger: logger
    @param cut: int, > cut PETs kept
    @param mcut: int, <mcut PETs kept
    @param ext: int, extension from the PET center
    """
    if not os.path.exists(d):
        logger.error("%s not exists. return." % d)
        return
    fout = fout + "_PETs.bedpe.gz"
    if os.path.isfile(fout):
        logger.error("Traget output file %s has been generated, return."%fout)
        return

    logger.info("Converting %s to BEDPE file." % d)
    records = _select_dump_records(d, mode)
    i = 0
    with gzip.open(fout, "wt") as f:
        for record in records:
            chrom_x, chrom_y, mat = _load_dump_record(
                record, cut=cut, mcut=mcut)
            for t in tqdm(mat):
                a = (chrom_x, max([0, t[0] - ext]), t[0] + ext)
                b = (chrom_y, max([0, t[1] - ext]), t[1] + ext)
                line = [
                    a[0], a[1], a[2],
                    b[0], b[1], b[2], 
                    i, ".", "+","-",
                ]
                f.write("\t".join(map(str, line)) + "\n")
                i += 1
    logger.info("Converting to BEDPE file %s finished." % fout)



def ixy2hic(
            d,
            fout,
            logger,
            org="hg38",
            resolution="1000,5000,10000,50000,100000,200000",
            cut=0,
            mcut=-1,
    ):
    """
    Convert reads level bedpe to HIC.
    Track format according to https://github.com/theaidenlab/juicer/wiki/Pre#file-format
    @param d: str,cLoops2 pre data directory
    @param fout: str,prefix of output files
    @param logger: logger
    @param org: str, species for the data
    @param resolution: str, series of resolutions
    @param cut: int, > cut PETs kept
    @param mcut: int, <mcut PETs kept
    """
    if not isTool("juicer_tools"):
        logger.error( "juicer_tools is not available in the executative enviroment! Please install and re-try.")
        return
    if not os.path.exists(d):
        logger.error("%s not exists,return." % d)
        return
    if os.path.isfile(fout+".hic"):
        logger.error("Traget output file %s.hic has been generated, return."%fout)
        return

    logger.info("Converting %s to .hic file which could be loaded in juicebox" % d)
    fs = glob(d + "/*.ixy")
    tmp = str(random.random())
    ss = {"+": 0, "-": 1}
    with open(tmp, "w") as fo:
        for fin in fs:
            print("converting %s" % fin)
            key, mat = parseIxy(fin, cut=cut,mcut=mcut)
            for t in tqdm(mat):
                line = [0, key[0], t[0], 0, 1, key[1], t[1], 1]
                fo.write("\t".join(map(str, line)) + "\n")
    # -n option from juicer_tools will cause trouble for trac-looping data
    #c1 = "juicer_tools pre -n -t ./ -r {resolution} {fin} {fout} {org}".format( 
    c1 = "juicer_tools pre -t ./ -r {resolution} {fin} {fout} {org}".format(
        resolution=resolution, fin=tmp, fout=fout+".hic", org=org)
    c2 = "rm %s" % tmp
    callSys([c1, c2])
    logger.info("Converting to juicer's hic file %s finished." % fout)



def ixy2washU(
            d, 
            fout, 
            logger,
            cut=0, 
            mcut=-1,
            ext=50,
            mode="cis",
    ):
    """
    Convert PETs to washU long range interactions. 
    Track format according to http://wiki.wubrowse.org/Long-range
    @param d: str,cLoops2 pre data directory
    @param fout: str,prefix of output files
    @param logger: logger
    @param cut: int, > cut PETs kept
    @param mcut: int, <mcut PETs kept
    @param ext: int, extension from the PET center
    """
    for t in ["bgzip", "tabix"]:
        if not isTool(t):
            logger.error(
                "%s is not available in the executative enviroment! Please install and re-try."
                % t)
            return
    if not os.path.exists(d):
        logger.error("%s not exists. return." % d)
        return
    fout = fout + "_PETs_washU.txt"
    if os.path.isfile(fout+".gz"):
        logger.error("Traget output file %s.gz has been generated, return."%fout)
        return

    logger.info("Converting %s to washU track." % d)
    records = _select_dump_records(d, mode)
    i = 0
    unsorted = fout + ".unsorted"
    with open(unsorted, "w") as f:
        for a_chrom, a_start, a_end, b_chrom, b_start, b_end in \
                _iter_bidirectional_intervals(
                    records, cut=cut, mcut=mcut, ext=ext):
            line = [a_chrom, a_start, a_end,
                    "%s:%s-%s,1" % (b_chrom, b_start, b_end), i, "."]
            f.write("\t".join(map(str, line)) + "\n")
            i += 1
    _sort_bed_like(unsorted, fout)
    os.unlink(unsorted)
    c1 = "bgzip %s" % fout
    c2 = "tabix -p bed %s.gz" % fout
    callSys([c1, c2])
    logger.info("Converting to washU long-range track %s finished." % fout)



def ixy2ucsc(
            d, 
            fout, 
            chromSizeF,
            logger,
            cut=0, 
            mcut=-1,
            ext=50,
            mode="cis",
    ):
    """
    Convert PETs to UCSC bigInteract track. 
    Track format according to https://genome.ucsc.edu/goldenPath/help/interact.html
    @param d: str,cLoops2 pre data directory
    @param fout: str,prefix of output files
    @param chromSizeF: str, file of chrom sizes, can be obtained through fetchChromSizes
    @param logger: logger
    @param cut: int, > cut PETs kept
    @param mcut: int, <mcut PETs kept
    @param ext: int, extension from the PET center

    """
    for t in ["bedToBigBed"]:
        if not isTool(t):
            logger.error(
                "%s is not available in the executative enviroment! Please install and re-try."
                % t)
            return
    if not os.path.exists(d):
        logger.error("%s not exists. return." % d)
        return
    if not os.path.exists(chromSizeF):
        logger.error("%s not exists. return." % chromSizeF)
        return
    if os.path.isfile(fout+".bb"):
        logger.error("Traget output file %s.bb has been generated, return."%fout)
        return

    fildes="""
table interact
"Interaction between two regions"
    (
    string chrom;      "Chromosome (or contig, scaffold, etc.). For interchromosomal, use 2 records"
    uint chromStart;   "Start position of lower region. For interchromosomal, set to chromStart of this region"
    uint chromEnd;     "End position of upper region. For interchromosomal, set to chromEnd of this region"
    string name;       "Name of item, for display.  Usually 'sourceName/targetName' or empty"
    uint score;        "Score from 0-1000."
    double value;      "Strength of interaction or other data value. Typically basis for score"
    string exp;        "Experiment name (metadata for filtering). Use . if not applicable"
    string color;      "Item color.  Specified as r,g,b or hexadecimal #RRGGBB or html color name, as in //www.w3.org/TR/css3-color/#html4."
    string sourceChrom;  "Chromosome of source region (directional) or lower region. For non-directional interchromosomal, chrom of this region."
    uint sourceStart;  "Start position source/lower/this region"
    uint sourceEnd;    "End position in chromosome of source/lower/this region"
    string sourceName;  "Identifier of source/lower/this region"
    string sourceStrand; "Orientation of source/lower/this region: + or -.  Use . if not applicable"
    string targetChrom; "Chromosome of target region (directional) or upper region. For non-directional interchromosomal, chrom of other region"
    uint targetStart;  "Start position in chromosome of target/upper/this region"
    uint targetEnd;    "End position in chromosome of target/upper/this region"
    string targetName; "Identifier of target/upper/this region"
    string targetStrand; "Orientation of target/upper/this region: + or -.  Use . if not applicable"
    )
    """
    #tmp file
    with open(fout+".tmp.as","w") as fo:
        fo.write(fildes)

    logger.info("Converting %s to UCSC track." % d)
    records = _select_dump_records(d, mode)
    i = 0
    unsorted = fout + ".tmp.unsorted.bed"
    with open(unsorted, "w") as f:
        for a_chrom, a_start, a_end, b_chrom, b_start, b_end in \
                _iter_bidirectional_intervals(
                    records, cut=cut, mcut=mcut, ext=ext):
            line = [a_chrom, a_start, a_end, ".", 1, 1, ".", 0,
                    a_chrom, a_start, a_end, ".", ".",
                    b_chrom, b_start, b_end, ".", "."]
            f.write("\t".join(map(str, line)) + "\n")
            i += 1
    _sort_bed_like(unsorted, fout + ".tmp.bed")
    os.unlink(unsorted)
    c1 = "bedToBigBed -tab -as=%s.tmp.as -type=bed5+13 %s.tmp.bed %s %s.bb"%( fout,fout,chromSizeF,fout )
    c2 = "rm %s.tmp.bed %s.tmp.as"%(fout,fout)
    callSys([c1,c2])
    logger.info("Converting to UCSC bigInteract track %s finished." % fout)



def addCov(cov, iv):
    """
    Add the coverage for a array. No value region is marked as False.
    """
    if len(cov) < iv[1]:
        cov.extend([False] * (iv[1] - len(cov) + 1))
    for i in range(iv[0], iv[1]):
        if cov[i] == False:
            cov[i] = 0
        cov[i] += 1
    return cov


def ixy2bdg(    
        d, 
        fout, 
        logger,
        cut=0, 
        mcut=-1,
        ext=50,
        pe=False,
    ):
    """
    Convert PETs to 1D bedGraph file with intrac-chromosomal PETs.

    @param d: str,cLoops2 pre data directory
    @param fout: str,prefix of output files
    @param logger: logger
    @param cut: int, > cut PETs kept
    @param mcut: int, <mcut PETs kept
    @param pe: bool, whether to treat paired-end tags as single end, set to True for ChIP-seq, ATAC-seq
    """
    if not os.path.exists(d):
        logger.error("%s not exists. return." % op.dir)
        return
    fout = fout + ".bdg"
    if os.path.isfile(fout):
        logger.error("Traget output file %s has been generated, return."%fout)
        return

    logger.info("Converting %s to bedGraph, normalized as RPM." % d)
    fs = glob(d + "/*.ixy")
    nfs = []
    for f in fs:
        chrom = f.split("/")[-1].split(".ixy")[0].split("-")
        #only cis PETs used
        if chrom[0] == chrom[1]:
            nfs.append(f)
    fs = nfs
    fs.sort()  #sort as lexicograhicly

    metaf = d + "/petMeta.json"
    meta = json.loads(open(metaf).read())
    tot = meta["Unique PETs"] * 2
    
    with open(fout, "w") as f:
        for fin in fs:
            print("converting %s" % fin)
            key, mat = parseIxy(fin, cut=cut,mcut=mcut)
            #obtain coverage
            cov = []
            for t in tqdm(mat):
                if pe: #not interacting data, only normal paired-end, used for ChIC-seq, ATAC-seq.
                    a = (max([0, t[0] - ext]), t[1] + ext)
                    cov = addCov(cov, a)
                else:
                    a = (max([0, t[0] - ext]), t[0] + ext)
                    cov = addCov(cov, a)
                    b = (max([0, t[1] - ext]), t[1] + ext)
                    cov = addCov(cov, b)
            #write the bedGraph file use the step vector
            print("writting %s to bedGraph" % key[0])
            i = 0
            while i < len(cov) - 1:
                if cov[i] == False:  #find the non 0 start
                    i += 1
                    continue
                j = i+1
                while j < len(cov) - 1:
                    if cov[j] != cov[i]:
                        break
                    j += 1
                """
                for j in range(i + 1, len(cov)):  #find the same value stop
                    if cov[j] != cov[i]:
                        break
                """
                v = cov[i] / 1.0 / tot * 10**6
                v = "%.3f"%v
                line = [key[0], i, j - 1, v]
                f.write("\t".join(list(map(str, line))) + "\n")
                if j == len(cov) - 1:
                    break
                i = j
            del cov
    logger.info("Converting to bedGraph track %s finished." % fout)


def ixy2mat(
        d,
        fout,
        logger,
        chrom="",
        start=-1,
        end=-1,
        r=5000,
        cut=0,
        mcut=-1,
        log=False,
        method="obs",
        corr=False,
        norm=False,
):
    """
    Get the contact matrix.
    @param d: str,cLoops2 pre data directory
    @param fout: str,prefix of output files
    @param logger: logger
    @param chrom: str, such as "chr1-chr1"
    @param start: int, start location
    @param end: int, end location
    @param r: int, resolution bin size
    @param cut: int, > cut PETs kept
    @param mcut: int, <mcut PETs kept
    @param log: bool, whether do log transformation
    @param method: choice, available are obs, obs/exp
    @param corr: bool, whehter to get the correlation matrix
    @param norm: bool, whether to normalize the matrix with z-score
    """
    if start != -1 and end != -1 and end < start:
        logger.error("End %s is smaller than %s start." % (end, start))
        return
    f = os.path.join(d,chrom+".ixy")
    if not os.path.isfile(f):
        logger.error("%s not exists, please check the input -mat_chrom"%f)
        return

    chrom, xy = parseIxy(f, cut=cut,mcut=mcut)
    if start == -1:
        start = np.min(xy)
    if end == -1:
        end = np.max(xy)
    mat = getObsMat(xy, start, end, r)
    bgmat = None
    if method == "obs/exp":
        bgmat = getExpMat(xy, mat.shape, start, end, r)
    if log:
        if bgmat is None:
            mat = np.log10(mat + 1)
        else:
            mat = np.log10(mat + 1) - np.log10(bgmat + 1)
    else:
        if bgmat is not None:
            mat = (mat + 1) / (bgmat + 1)
    if corr:
        mat = np.corrcoef(mat)
        mat = np.nan_to_num(mat)
    if norm:
        m = np.mean(mat)
        s = np.std(mat)
        mat = (mat - m) / s
    rs = []
    bs = int((end - start) / r) + 1
    for i in range(bs):
        nr = (chrom[0], start + i * r, start + i * r + r)
        rs.append("|".join(list(map(str, nr))))
    mat = pd.DataFrame(mat, index=rs, columns=rs)
    mat.to_csv(fout + "_cmat.txt", sep="\t", index_label="pos")
    logger.info("Converting to contact matrix txt %s_cmat.txt finished." % fout)


def ixy2transmat(
        d,
        fout,
        logger,
        chrom="",
        x_start=-1,
        x_end=-1,
        y_start=-1,
        y_end=-1,
        x_res=5000,
        y_res=None,
        method="obs",
        log=False,
        sparse=False,
        max_dense_cells=10000000,
):
    """Export a directional ``chromX bins x chromY bins`` trans matrix.

    ``pair_oe`` uses endpoint marginals from the complete chromosome-pair
    file, whereas ``window_oe`` conditions on the selected rectangular
    window.  These are different estimands and are named explicitly.  Sparse
    output currently represents observed counts and is suitable for a full
    chromosome pair; dense output is bounded before allocation.
    """
    if method not in ("obs", "pair_oe", "window_oe"):
        raise ValueError("trans matrix method must be obs, pair_oe, or window_oe")
    if sparse and log:
        raise ValueError(
            "-log is not supported for sparse trans COO output; transform "
            "the explicit count column downstream")
    if y_res is None:
        y_res = x_res
    meta_path = os.path.join(d, "petMeta.json")
    with open(meta_path) as handle:
        meta = json.load(handle)
    entry = meta.get("data", {}).get("trans", {}).get(chrom)
    if not isinstance(entry, dict) or not entry.get("ixy"):
        raise ValueError(
            "trans chromosome pair %r is not present in petMeta.json" % chrom)
    chrom_x = entry.get("chromX")
    chrom_y = entry.get("chromY")
    if chrom_x is None or chrom_y is None:
        axes = chrom.split("-")
        if len(axes) != 2:
            raise ValueError("ambiguous trans pair; chromX/chromY are required")
        chrom_x, chrom_y = axes
    try:
        xy = joblib.load(entry["ixy"], mmap_mode="r")
    except Exception:
        xy = joblib.load(entry["ixy"])
    if (not isinstance(xy, np.ndarray) or xy.ndim != 2 or
            xy.shape[1] != 2):
        raise ValueError("trans ixy must have shape (n, 2)")
    if xy.shape[0] == 0:
        raise ValueError("cannot infer/export a matrix from an empty trans pair")
    x_start = int(np.min(xy[:, 0])) if x_start == -1 else int(x_start)
    x_end = int(np.max(xy[:, 0])) if x_end == -1 else int(x_end)
    y_start = int(np.min(xy[:, 1])) if y_start == -1 else int(y_start)
    y_end = int(np.max(xy[:, 1])) if y_end == -1 else int(y_end)

    if sparse:
        if method != "obs":
            raise ValueError(
                "sparse trans export currently supports method=obs only")
        matrix = getTransObsMatCOO(xy, x_start, x_end, y_start, y_end,
                                   x_res, y_res)
        path = fout + "_trans_cmat_coo.txt"
        with open(path, "w") as handle:
            handle.write(
                "rowBin\tchromX\txStart\txEnd\tcolBin\tchromY\tyStart\t"
                "yEnd\tcount\n")
            order = np.lexsort((matrix.col, matrix.row))
            for position in order:
                row = int(matrix.row[position])
                col = int(matrix.col[position])
                handle.write("\t".join(map(str, (
                    row, chrom_x, x_start + row * x_res,
                    min(x_end, x_start + (row + 1) * x_res - 1),
                    col, chrom_y, y_start + col * y_res,
                    min(y_end, y_start + (col + 1) * y_res - 1),
                    int(matrix.data[position]),
                ))) + "\n")
        logger.info("Wrote sparse trans contact matrix to %s." % path)
        return path

    observed = getTransObsMat(
        xy, x_start, x_end, y_start, y_end, x_res, y_res,
        max_dense_cells=max_dense_cells)
    matrix = observed.astype(float) if method != "obs" else observed
    if method == "pair_oe":
        x_bins = ((np.asarray(xy[:, 0]) - x_start) // x_res).astype(int)
        y_bins = ((np.asarray(xy[:, 1]) - y_start) // y_res).astype(int)
        valid_x = (np.asarray(xy[:, 0]) >= x_start) & \
                  (np.asarray(xy[:, 0]) <= x_end)
        valid_y = (np.asarray(xy[:, 1]) >= y_start) & \
                  (np.asarray(xy[:, 1]) <= y_end)
        row_marginal = np.bincount(
            x_bins[valid_x], minlength=observed.shape[0])[:observed.shape[0]]
        column_marginal = np.bincount(
            y_bins[valid_y], minlength=observed.shape[1])[:observed.shape[1]]
        expected = np.outer(row_marginal, column_marginal) / float(xy.shape[0])
        matrix = np.divide(observed,
                           expected,
                           out=np.zeros_like(expected, dtype=float),
                           where=expected > 0)
    elif method == "window_oe":
        total = float(observed.sum())
        expected = (np.outer(observed.sum(axis=1), observed.sum(axis=0)) /
                    total if total > 0 else np.zeros_like(observed,
                                                         dtype=float))
        matrix = np.divide(observed,
                           expected,
                           out=np.zeros_like(expected, dtype=float),
                           where=expected > 0)
    if log:
        matrix = np.log2(np.asarray(matrix, dtype=float) + 1.0)
    rows = ["%s:%s-%s" % (
        chrom_x, x_start + index * x_res,
        min(x_end, x_start + (index + 1) * x_res - 1))
            for index in range(matrix.shape[0])]
    columns = ["%s:%s-%s" % (
        chrom_y, y_start + index * y_res,
        min(y_end, y_start + (index + 1) * y_res - 1))
               for index in range(matrix.shape[1])]
    path = fout + "_trans_cmat.txt"
    pd.DataFrame(matrix, index=rows, columns=columns).to_csv(
        path, sep="\t", index_label="pos")
    logger.info("Wrote rectangular trans contact matrix to %s." % path)
    return path


def ixy2virtual4C(
        d,
        fout,
        logger,
        chrom="",
        start=-1,
        end=-1,
        viewStart=-1,
        viewEnd=-1,
        cut=0,
        mcut=-1,
    ):
    """
    Get the virtual 4C signal for a specific view point.
    @param d: str,cLoops2 pre data directory
    @param fout: str,prefix of output files
    @param logger: logger
    @param chrom: str, such as "chr1-chr1"
    @param start: int, start location
    @param end: int, end location
    @param cut: int, > cut PETs kept
    @param mcut: int, <mcut PETs kept
    """
    if start != -1 and end != -1 and end < start:
        logger.error("End %s is smaller than %s start." % (end, start))
        return
    if viewStart == -1 or viewEnd == -1:
        logger.error("viewStart or viewEnd not assigned.")
        return
    if viewStart != -1 and viewEnd != -1 and viewEnd < viewStart:
        logger.error("viewEnd %s is smaller than %s viewStart." % (viewEnd, viewStart))
        return
    f = os.path.join(d,chrom+".ixy")
    if not os.path.isfile(f):
        logger.error("%s not exists, please check the input -virtual4C_chrom"%f)
        return
    metaf = d + "/petMeta.json"
    meta = json.loads(open(metaf).read())
    tot = meta["Unique PETs"] 
    chrom, xy = parseIxy(f, cut=cut, mcut=mcut)
    if start == -1:
        start = np.min(xy)
    if end == -1:
        end = np.max(xy)
    ps = np.where((xy[:, 0] >= start) & (xy[:, 1] <= end))[0]
    xy = xy[ps, ]
    xy2 = XY(xy[:, 0], xy[:, 1])  #XY object
    virtual4Csig = getVirtual4CSig(xy2, start, end, viewStart, viewEnd)
    with open( fout + "_4C.bdg","w") as fo:
        i = 0
        while i < len(virtual4Csig) -1 :
            if virtual4Csig[i] == 0 :
                i += 1
                continue
            for j in range(i+1, len(virtual4Csig)):
                if virtual4Csig[j] != virtual4Csig[i]:
                    break
            #v = np.log2(virtual4Csig[i] / 1.0 / tot * 10**6)
            #v = virtual4Csig[i] / 1.0 / tot * 10**6
            v = np.log2(virtual4Csig[i])
            line = [chrom[0],start+i, start+j-1,v]
            fo.write("\t".join(list(map(str, line))) + "\n")
            if j == len(virtual4Csig) - 1:
                break
            i = j
