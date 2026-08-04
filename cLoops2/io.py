#!/usr/bin/env python
#--coding:utf-8 --
"""
cLoops IO module.
2020-01-21: updated parseBedpe, small trick to handle too big files, such as HiC, hash(tuple)
2020-01-23: parse bed added
2020-02-19: updated json write
2020-03-04: loops to new washU long range chromatin interaction s added.
2020-04-09: updated parseIxy to add <=dist cutoff.
2020-07-30: DiffLoop object to washU, UCSC, Juicebox added.
2020-11-08: improving parseBedpe, to get the unique PETs from each txt file by parallel.
2021-08-20: adding parsePairs, pairs defined as https://pairtools.readthedocs.io/en/latest/formats.html#pairs
2023-10-16: updated dparsePairs in case there is no 8th columns
"""

__author__ = "CAO Yaqiang"
__date__ = ""
__modified__ = ""
__email__ = "caoyaqiang0410@gmail.com"

#sys
import os
import gzip
import json
import random
import warnings
from glob import glob

#3rd
import joblib
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

#cLoops2
from cLoops2.ds import PET, Pair, XY, Loop, Peak, Domain
from cLoops2.utils import callSys, cFlush, getLogger


## PETs level operation releated IO
def getUniqueTxt(f):
    """
    Get the unique reads from txt file
    """
    redu = set()
    with open(f) as handle:
        for line in handle:
            line = line.split("\n")[0].split("\t")
            line = tuple(line)
            redu.add(line)
    with open(f,"w") as fo:
        # Stable output makes downstream seeded sampling reproducible across
        # independent preprocessing runs with identical input.
        for t in sorted(redu):
            fo.write("\t".join(t)+"\n")
    return len(redu)


def txt2ixy(f):
    """
    Dump the np.ndarray using joblib.dump for fast access.
    .ixy is cLoops2 specific format.
    """
    data = []
    with open(f) as handle:
        for line in handle:
            line = line.split("\n")[0].split("\t")
            data.append(list(map(int, line)))
    data = np.array(data)
    joblib.dump(data, f.replace(".txt", ".ixy"))
    os.remove(f)
    return f.replace(".txt", ".ixy")



def parseBedpe(fs, fout, logger, mapq=1, cs=[], cut=0,mcut=-1, cis=False,cpu=1):
    """
    Pre-processing PETs, organized by chromosomes. Input could be mixed PETs in bedpe.gz or bedpe. Also change read id to numbers to minize memory usage.
    @param fs: bedpe files of replicates, could be .bedpe or .bedpe.gz
    @param fout: output prefix, the name for directory
    @param logger: logger object
    @param cs: chroms that wanted, list like ["chr1","chr2"]
    """
    #chroms data, used to keep
    chroms = {}
    total, hiq, t, c, closeCis,farCis = 0, 0, 0, 0, 0, 0
    for f in fs:
        r = "Parsing PETs from %s, requiring initial distance cutoff >%s and <%s" % (
            f, cut,mcut)
        logger.info(r)
        if f.endswith(".gz"):
            #of = gzip.open(f, "rb") #python2
            of = gzip.open(f, "rt")  #python3
        else:
            of = open(f)
        for i, line in enumerate(of):
            if i % 100000 == 0:
                cFlush("%s PETs processed from %s" % (i, f))
            line = line.split("\n")[0].split("\t")
            #if "*" in line and "-1" in line:
            #    continue
            try:
                pet = PET(line)
            except:
                continue
            #filtering unwanted PETs in chroms
            if len(cs) > 0 and (not (pet.chromA in cs and pet.chromB in cs)):
                continue
            total += 1  #total read in PETs
            if pet.mapq < mapq:
                continue
            hiq += 1  #high quality PETs
            if pet.cis == True:
                c += 1
            else:
                t += 1
            #skip trans PETs for furthur processing
            if cis and pet.cis == False:
                continue
            #filtering too close cis PETs
            if cut > 0 and pet.cis and pet.distance < cut:
                closeCis += 1
                continue
            if mcut > 0 and pet.cis and pet.distance > mcut:
                farCis += 1
                continue
            #pet.chromA and pet.chromB is sorted in PET object
            key = (pet.chromA, pet.chromB)
            if key not in chroms:
                cf = os.path.join(fout,
                                  "%s-%s" % (pet.chromA, pet.chromB) + ".txt")
                chroms[key] = {
                    "f": cf,
                    "of": open(cf, "w"),
                }
            nline = [pet.cA, pet.cB]
            chroms[key]["of"].write("\t".join(list(map(str, nline))) + "\n")
        of.close()
    print("\n" * 2)
    #get unique PETs
    for value in chroms.values():
        value["of"].close()
    records = [(key, value["f"]) for key, value in chroms.items()]
    nfs = [record[1] for record in records]
    del (chroms)
    unique_counts = Parallel(n_jobs=cpu,backend="multiprocessing")(
        delayed(getUniqueTxt)(f) for f in nfs)
    unique_cis = int(
        np.sum([n for ((chrom_a, chrom_b), _), n in zip(records,
                                                        unique_counts)
                if chrom_a == chrom_b]))
    unique_trans = int(
        np.sum([n for ((chrom_a, chrom_b), _), n in zip(records,
                                                        unique_counts)
                if chrom_a != chrom_b]))
    uniques = unique_cis + unique_trans
    r = "Totaly %s PETs in target chromosomes from %s, in which %s high quality unqiue PETs" % (
        total, ",".join(fs), uniques)
    logger.info(r)
    retained_cis = c - closeCis - farCis
    if retained_cis > 0:
        nr = 1 - unique_cis / float(retained_cis)
    else:
        nr = 0
    ds = {
        "Total PETs": total,
        "High Mapq(>=%s) PETs" % mapq: hiq,
        "Total Trans PETs": t,
        "Total Cis PETs": c,
        "Filtered too close (<%s) PETs" % cut: closeCis,
        "Filtered too distant (>%s) PETs" % mcut: farCis,
        "Unique PETs": uniques,
        "Unique Cis PETs": unique_cis,
        "Unique Trans PETs": unique_trans,
        "Physical Cis PET Rows": unique_cis,
        "Physical Trans PET Rows": unique_trans,
        "Retention": {
            "retain trans": not cis,
            "retained categories": ["cis", "trans"] if not cis else ["cis"],
            "chromosome whitelist": sorted(cs),
            "cut": cut,
            "mcut": mcut,
            "mapq": mapq,
        },
        "Cis PETs Redundancy": nr
    }
    with open(fout + "/petMeta.json", "w") as fo:
        json.dump(ds, fo)
    #convert to .ixy files
    logger.info("writing .ixy files")
    Parallel(n_jobs=cpu,backend="multiprocessing")(delayed(txt2ixy)(f) for f in nfs)
    nfs = glob(fout + "/*.ixy")
    #collect files and update meta information
    ixyfs = glob(fout + "/*.ixy")
    metaf = fout + "/petMeta.json"
    updateJson(ixyfs, metaf)


def parsePairs(fs, fout, logger, cs=[], cut=0,mcut=-1, cis=False,cpu=1):
    """
    Pre-processing PETs, organized by chromosomes. Input could be mixed PETs in pairs.gz or pairs. 
    Also change read id to numbers to minize memory usage.
    @param fs: pairs files of replicates, could be .pairs or .pairs.gz
    @param fout: output prefix, the name for directory
    @param logger: logger object
    @param cs: chroms that wanted, list like ["chr1","chr2"]
    """
    #chroms data, used to keep
    chroms = {}
    total, t, c, closeCis,farCis = 0, 0, 0, 0, 0
    for f in fs:
        r = "Parsing PETs from %s, requiring initial distance cutoff >%s and <%s" % (
            f, cut,mcut)
        logger.info(r)
        if f.endswith(".gz"):
            #of = gzip.open(f, "rb") #python2
            of = gzip.open(f, "rt")  #python3
        else:
            of = open(f)
        for i, line in enumerate(of):
            if i % 100000 == 0:
                cFlush("%s PETs processed from %s" % (i, f))
            if line.startswith("#"):
                continue
            line = line.split("\n")[0].split("\t")
            if len(line) == 8:
                if line[7] not in [ "UU", "MR","MU","RU","UR" ]:
                    continue
                try:
                    pet = Pair(line)
                except:
                    continue
            elif len(line) == 7:
                try:
                    pet = Pair(line)
                except:
                    continue
            else:
                print("Missing information for record as .pair file: %s"%"\t".join(line)) 
                continue
            #filtering unwanted PETs in chroms
            if len(cs) > 0 and (not (pet.chromA in cs and pet.chromB in cs)):
                continue
            total += 1  #total read in PETs
            if pet.cis == True:
                c += 1
            else:
                t += 1
            #skip trans PETs for furthur processing
            if cis and pet.cis == False:
                continue
            #filtering too close cis PETs
            if cut > 0 and pet.cis and pet.distance < cut:
                closeCis += 1
                continue
            if mcut > 0 and pet.cis and pet.distance > mcut:
                farCis += 1
                continue
            #pet.chromA and pet.chromB is sorted in PET object
            key = (pet.chromA, pet.chromB)
            if key not in chroms:
                cf = os.path.join(fout,
                                  "%s-%s" % (pet.chromA, pet.chromB) + ".txt")
                chroms[key] = {
                    "f": cf,
                    "of": open(cf, "w"),
                }
            nline = [pet.cA, pet.cB]
            chroms[key]["of"].write("\t".join(list(map(str, nline))) + "\n")
        of.close()
    print("\n" * 2)
    #get unique PETs
    for value in chroms.values():
        value["of"].close()
    records = [(key, value["f"]) for key, value in chroms.items()]
    nfs = [record[1] for record in records]
    del (chroms)
    unique_counts = Parallel(n_jobs=cpu,backend="multiprocessing")(
        delayed(getUniqueTxt)(f) for f in nfs)
    unique_cis = int(
        np.sum([n for ((chrom_a, chrom_b), _), n in zip(records,
                                                        unique_counts)
                if chrom_a == chrom_b]))
    unique_trans = int(
        np.sum([n for ((chrom_a, chrom_b), _), n in zip(records,
                                                        unique_counts)
                if chrom_a != chrom_b]))
    uniques = unique_cis + unique_trans
    r = "Totaly %s PETs in target chromosomes from %s, in which %s high quality unqiue PETs" % (
        total, ",".join(fs), uniques)
    logger.info(r)
    retained_cis = c - closeCis - farCis
    if retained_cis > 0:
        nr = 1 - unique_cis / float(retained_cis)
    else:
        nr = 0
    ds = {
        "Total PETs": total,
        "Total Trans PETs": t,
        "Total Cis PETs": c,
        "Filtered too close (<%s) PETs" % cut: closeCis,
        "Filtered too distant (>%s) PETs" % mcut: farCis,
        "Unique PETs": uniques,
        "Unique Cis PETs": unique_cis,
        "Unique Trans PETs": unique_trans,
        "Physical Cis PET Rows": unique_cis,
        "Physical Trans PET Rows": unique_trans,
        "Retention": {
            "retain trans": not cis,
            "retained categories": ["cis", "trans"] if not cis else ["cis"],
            "chromosome whitelist": sorted(cs),
            "cut": cut,
            "mcut": mcut,
            "format": "pairs",
        },
        "Cis PETs Redundancy": nr
    }
    with open(fout + "/petMeta.json", "w") as fo:
        json.dump(ds, fo)
    #convert to .ixy files
    logger.info("writing .ixy files")
    Parallel(n_jobs=cpu,backend="multiprocessing")(delayed(txt2ixy)(f) for f in nfs)
    nfs = glob(fout + "/*.ixy")
    #collect files and update meta information
    ixyfs = glob(fout + "/*.ixy")
    metaf = fout + "/petMeta.json"
    updateJson(ixyfs, metaf)



def parseIxy(f, cut=0,mcut=-1):
    """
    Read data from .ixy file 
    @param cut: dist >= cut
    @param mcut: dist <= mcut
    """
    key = os.path.split(f)[1].replace(".ixy", "")
    key = tuple(key.split("-"))
    mat = joblib.load(f)
    if cut > 0:
        d = mat[:, 1] - mat[:, 0]
        p = np.where(d >= cut)[0]
        mat = mat[p, :]
    if mcut >0:
        d = mat[:, 1] - mat[:, 0]
        p = np.where(d <= mcut)[0]
        mat = mat[p, :]
    return key, mat


def ixy2pet(f, cut=0,mcut=-1):
    """
    Convert .ixy file to sorted regions
    """
    key, mat = parseIxy(f, cut=cut,mcut=mcut)
    xy = XY(mat[:, 0], mat[:, 1])
    return xy
    #joblib.dump(xy, f.replace(".ixy", ".pet"))


def parsePet(f):
    """
    Parse .pet file.
    @param f: str, .pet file name
    """
    key = os.path.split(f)[1].replace(".pet", "")
    key = tuple(key.split("-"))
    xy = joblib.load(f)
    return key, xy


def combineIxys(key,fixys,outdir,keep=1):
    """
    Combine multiple ixy files.
    @param key: str, such as chr1-chr1
    @param fixys: list of str, .ixy file paths
    @param outdir: str, output directory
    @param keep: int, how many to keep for same location
    """
    if keep < 0:
        raise ValueError("combine keep must be non-negative")
    mat = {}
    for f in fixys:
        k, nmat = parseIxy(f)
        for t in nmat:
            nt = (t[0],t[1])
            if nt not in mat:
                mat[nt] = 0
            mat[nt] += 1
    nmat = []
    for k, v in sorted(mat.items()):
        copies = v if keep == 0 else min(v, keep)
        for i in range(copies):
            nmat.append(k)
    del mat
    nmat = np.asarray(nmat, dtype=np.int64).reshape((-1, 2))
    fout = os.path.join(outdir,key+".ixy")
    joblib.dump(nmat, fout)


def combineDirs(dirs,fout,logger,keep=1,cpu=1):
    """
    Combine multiple cLoops2 pre directories. 
    @param ds: list of str, directories
    @param fout: str, output directory
    @param logger: logger object
    @param keep: how many to keep for the same location, 0 means all
    @pram cpu: cpu numbers to run the job
    """
    # Validate parents before creating output.  A combine is a materialized
    # transform, not a fresh ``pre`` root, so its replacement ancestry must be
    # propagated explicitly instead of being lost by writeNewJson().
    from cLoops2.metadata import build_library_context

    ds = {"cis":{},"trans":{}}
    parent_summaries = []
    parent_retentions = []
    for dir in dirs:
        metaf = dir + "/petMeta.json"
        with open(metaf) as handle:
            meta = json.load(handle)
        actual_counts = {}
        for category in ("cis", "trans"):
            entries = meta.get("data", {}).get(category, {})
            for key, entry in entries.items():
                record_id = entry.get(
                    "record_id", "%s:%s" % (category, key))
                try:
                    matrix = joblib.load(entry["ixy"], mmap_mode="r")
                except Exception:
                    matrix = joblib.load(entry["ixy"])
                actual_counts[str(record_id)] = int(matrix.shape[0])
                del matrix
        context = build_library_context(meta, actual_counts=actual_counts)
        parent_summaries.append({
            "directory": os.path.abspath(dir),
            "physical_total": context.physical_total,
            "logical_total": context.logical_total,
            "validity": context.validity,
            "source_kind": context.source_kind,
            "emit": context.emit,
            "replacement_ever": context.replacement_ever,
        })
        parent_retentions.append(meta.get("Retention"))
        for k, v in meta["data"]["cis"].items():
            if k not in ds["cis"]:
                ds["cis"][k] = []
            ds["cis"][k].append(v["ixy"])
        for k, f in meta["data"]["trans"].items():
            if k not in ds["trans"]:
                ds["trans"][k] = []
            ds["trans"][k].append(f["ixy"])
    os.mkdir(fout)
    #combine files
    logger.info("Combining intra-chromosomal PETs.")
    Parallel(n_jobs=cpu, backend="multiprocessing")(delayed(combineIxys)(
        k,
        v,
        fout, 
        keep,
    ) for k,v in tqdm(ds["cis"].items()))
    logger.info("Combining inter-chromosomal PETs.")
    Parallel(n_jobs=cpu, backend="multiprocessing")(delayed(combineIxys)(
        k,
        v,
        fout, 
        keep,
    ) for k,v in tqdm(ds["trans"].items()))
    #update meta information
    writeNewJson(fout)
    metaf = os.path.join(fout, "petMeta.json")
    with open(metaf) as handle:
        output_meta = json.load(handle)
    known_replacements = [
        parent["replacement_ever"] for parent in parent_summaries
    ]
    if any(value is True for value in known_replacements):
        replacement_ever = True
    elif all(value is False for value in known_replacements):
        replacement_ever = False
    else:
        replacement_ever = None
    parents_structurally_valid = all(
        parent["validity"] in ("valid", "legacy_root_assumed")
        for parent in parent_summaries)
    parents_fully_materialized = all(
        parent["source_kind"] in ("root", "all_sample") and
        parent["physical_total"] == parent["logical_total"]
        for parent in parent_summaries)
    retention_compatible = (
        bool(parent_retentions) and parent_retentions[0] is not None and
        all(retention == parent_retentions[0]
            for retention in parent_retentions[1:]))
    transform_valid = (parents_structurally_valid and
                       parents_fully_materialized and
                       retention_compatible)
    output_meta["Transformation"] = {
        "schema_version": 1,
        "operation": "combine",
        "keep": int(keep),
        "validity": "valid" if transform_valid else "unknown_ancestry",
        "integrity_level": "structural",
        "logical_total": int(output_meta["Unique PETs"]),
        "physical_output_total": int(output_meta["Unique PETs"]),
        "replacement_ever": replacement_ever,
        "retention_compatible": bool(retention_compatible),
        "parents_fully_materialized": bool(parents_fully_materialized),
        "formal_inference_eligible": False,
        "parents": parent_summaries,
    }
    if retention_compatible:
        output_meta["Retention"] = parent_retentions[0]
    with open(metaf, "w") as handle:
        json.dump(output_meta, handle, indent=2, sort_keys=True)
        handle.write("\n")


## meta file related functions
def updateJson(ixyfs, metaf):
    """
    Update the meta information as add the files. 
    """
    with open(metaf) as handle:
        meta = json.load(handle)
    meta["data"] = {"cis": {}, "trans": {}}
    #add .ixy files information
    for f in sorted(ixyfs):
        key = os.path.split(f)[1].replace(".ixy", "")
        key2 = tuple(key.split("-"))
        if len(key2) != 2 or not key2[0] or not key2[1]:
            raise ValueError(
                "ambiguous chromosome-pair filename %r; v0.2 metadata "
                "requires an unambiguous legacy chromX-chromY basename" %
                os.path.basename(f))
        entry = {
            "ixy": os.path.abspath(f),
            "chromX": key2[0],
            "chromY": key2[1],
            "record_id": "%s:%s" %
            ("cis" if key2[0] == key2[1] else "trans", key),
        }
        if key2[0] == key2[1]:
            meta["data"]["cis"][key] = entry
        else:
            meta["data"]["trans"][key] = entry
    with open(metaf, "w") as fo:
        json.dump(meta, fo)


def writeNewJson(fdir):
    """
    Rebuild physical metadata while preserving auditable provenance.

    Historical versions replaced ``petMeta.json`` with only ``Unique PETs``
    and ``data``, silently deleting Sampling metadata on ``cLoops2 update``.
    v0.2 preserves existing top-level fields.  Structural Sampling validation
    is still performed by :mod:`cLoops2.metadata`; if physical counts changed,
    the old manifest and rebuilt totals deliberately disagree and the
    validator marks the context invalid rather than inventing a new logical
    library depth.
    """
    ixyfs = sorted(glob(fdir + "/*.ixy"))
    tot = 0
    category_totals = {"cis": 0, "trans": 0}
    observed_records = {}
    for f in ixyfs:
        key = os.path.basename(f).replace(".ixy", "")
        axes = key.split("-")
        if len(axes) != 2 or not axes[0] or not axes[1]:
            raise ValueError(
                "ambiguous chromosome-pair filename %r" % os.path.basename(f))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                mat = joblib.load(f, mmap_mode="r")
        except Exception:
            mat = joblib.load(f)
        if not isinstance(mat, np.ndarray) or mat.ndim != 2 or mat.shape[1] != 2:
            raise ValueError("ixy file %s must have shape (n, 2)" % f)
        rows = int(mat.shape[0])
        tot += rows
        category = "cis" if axes[0] == axes[1] else "trans"
        category_totals[category] += rows
        observed_records["%s:%s" % (category, key)] = {
            "rows": rows,
            "dtype": str(mat.dtype),
        }
        del mat
    nmetaf = fdir + "/petMeta.json"
    meta = {}
    existing = None
    if os.path.isfile(nmetaf):
        with open(nmetaf) as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            meta.update(existing)
    old_totals = None
    if isinstance(existing, dict):
        old_totals = (
            existing.get("Unique PETs"),
            existing.get("Unique Cis PETs",
                         existing.get("Physical Cis PET Rows")),
            existing.get("Unique Trans PETs",
                         existing.get("Physical Trans PET Rows")),
        )
    meta["Unique PETs"] = tot
    meta["Unique Cis PETs"] = category_totals["cis"]
    meta["Unique Trans PETs"] = category_totals["trans"]
    meta["Physical Cis PET Rows"] = category_totals["cis"]
    meta["Physical Trans PET Rows"] = category_totals["trans"]
    sampling = meta.get("Sampling")
    if isinstance(sampling, dict):
        # Relocating a directory changes absolute output paths but not the
        # sampling operation.  Refresh paths by stable legacy key; all count
        # fields remain untouched for independent structural validation.
        paths_by_key = {
            os.path.basename(path).replace(".ixy", ""): os.path.abspath(path)
            for path in ixyfs
        }
        for record in sampling.get("records", []):
            if not isinstance(record, dict):
                continue
            key = record.get("key")
            if key in paths_by_key and int(record.get("written_rows", 0)) > 0:
                record["output_path"] = paths_by_key[key]
        expected_records = {}
        for record in sampling.get("records", []):
            if not isinstance(record, dict):
                continue
            record_id = record.get("record_id")
            if record_id:
                expected_records[str(record_id)] = record
        invalidation = []
        for record_id, observed in observed_records.items():
            expected = expected_records.get(record_id)
            if expected is None:
                invalidation.append(
                    "physical record %s is absent from Sampling manifest" %
                    record_id)
                continue
            if int(expected.get("written_rows", -1)) != observed["rows"]:
                invalidation.append(
                    "%s has %s rows, manifest expects %s" %
                    (record_id, observed["rows"],
                     expected.get("written_rows")))
            expected_dtype = expected.get("dtype")
            if (expected_dtype is not None and
                    str(expected_dtype) != observed["dtype"]):
                invalidation.append(
                    "%s dtype is %s, manifest expects %s" %
                    (record_id, observed["dtype"], expected_dtype))
        for record_id, expected in expected_records.items():
            if (int(expected.get("written_rows", 0)) > 0 and
                    record_id not in observed_records):
                invalidation.append(
                    "manifest record %s is missing its physical ixy" %
                    record_id)
        if invalidation:
            sampling["validity"] = "invalid"
            sampling["invalidation_reasons"] = invalidation
    elif ("Transformation" not in meta and old_totals is not None and
          old_totals[0] is not None and
          (int(old_totals[0]) != tot or
           (old_totals[1] is not None and
            int(old_totals[1]) != category_totals["cis"]) or
           (old_totals[2] is not None and
            int(old_totals[2]) != category_totals["trans"]))):
        # An update may repair paths, but a changed physical root is no longer
        # demonstrably the original no-replacement pre output.  Record the
        # uncertainty explicitly so rebuilding metadata cannot launder copied
        # rows into formal-inference eligibility.
        meta["Transformation"] = {
            "schema_version": 1,
            "operation": "external_update",
            "validity": "unknown_ancestry",
            "integrity_level": "structural",
            "logical_total": tot,
            "physical_output_total": tot,
            "replacement_ever": None,
            "parents_fully_materialized": False,
            "formal_inference_eligible": False,
            "previous_physical_totals": {
                "all": int(old_totals[0]),
                "cis": (None if old_totals[1] is None else
                        int(old_totals[1])),
                "trans": (None if old_totals[2] is None else
                          int(old_totals[2])),
            },
        }
    with open(nmetaf, "w") as fo:
        json.dump(meta, fo)
    updateJson(ixyfs, nmetaf)


### 1D features such as peaks releated
def peaks2txt(peaks, fout):
    """
    Converting list of cLoops2.ds.Peaks objects into txt file.
    """
    with open(fout, "w") as fo:
        header = [
            "peakId", "chr", "start", "end", "summit", "length", "counts", "RPKM",
            "enrichmentScore", "poissonPvalue", "controlCounts", "controlRPKM",
            "controlScaledCounts", "enrichmentScoreVsControl",
            "poissonPvalueVsControl", "pValueHarmonicMean","significant"
        ]
        fo.write("\t".join(header) + "\n")
        for i, peak in enumerate(peaks):
            if hasattr(peak, "id"):
                pid = peak.id
            else:
                pid = "peak_%s"%i
            line = [
                pid,
                peak.chrom, 
                peak.start, 
                peak.end, 
                peak.summit, 
                peak.length, 
                peak.counts, 
                peak.density, 
                peak.enrichment_score,
                peak.poisson_p_value,
            ]
            if hasattr(peak, "control_counts"):
                line.extend([
                    peak.control_counts, peak.control_density,
                    peak.control_scaled_counts,
                    peak.enrichment_score_vs_control,
                    peak.poisson_p_value_vs_control
                ])
            else:
                line.extend(["."] * 5)
            line.append(str(peak.p_value_mean))
            line.append(str(peak.significant))
            fo.write("\t".join(list(map(str, line))) + "\n")


def peaks2bed(peaks, fout, sig=True):
    """
    Converting cLoops2.ds.Peaks objects into BED file.
    @param sig: bool, if True, only write significant peaks to file.
    """
    with open(fout, "w") as fo:
        for i, peak in enumerate(peaks):
            info = "peak_%s;%sbp;%s reads;summit:%s;p-value:%s" % (
                i, peak.length, peak.counts, peak.summit, peak.poisson_p_value)
            if hasattr(peak, "control_counts"):
                info += ";control_counts:%s;enrichment_score_vs_control:%s;control_scaled_counts:%s;p-value_vs_control:%s" % (
                    peak.control_counts, peak.enrichment_score_vs_control,
                    peak.control_scaled_counts,
                    peak.poisson_p_value_vs_control)
            line = [
                peak.chrom, peak.start, peak.end, info,
                peak.enrichment_score[0]
            ]
            if sig and peak.significant == 1:
                fo.write("\t".join(list(map(str, line))) + "\n")
            if sig == False:
                fo.write("\t".join(list(map(str, line))) + "\n")


def parseBed2Peaks(fbed):
    """
    """
    peaks = {}
    with open(fbed) as handle:
        for line in handle:
            line = line.split("\n")[0].split("\t")
            if len(line) < 3:
                continue
            key = line[0] + "-" + line[0]
            if key not in peaks:
                peaks[key] = []
            peak = Peak()
            if len(line) >=4:
                peak.id = "|".join(line[:4])
            else:
                peak.id = "|".join(line[:3])
            peak.chrom = line[0]
            peak.start = int(line[1])
            peak.end = int(line[2])
            peak.length = peak.end - peak.start
            peaks[key].append(peak)
    return peaks



### 2D features such as loops releated
def loops2txt(loops, fout):
    """
    Converting list of cLoops2.ds.Loops objects into txt file.
    """
    with open(fout, "w") as fo:
        header = [
            "loopId", "chrA", "startA", "endA", "chrB", "startB", "endB",
            "distance(bp)", "centerA", "centerB", "readsA", "readsB", "cis",
            "PETs", "density", "enrichmentScore", "P2LL", "FDR",
            "binomialPvalue", "hypergeometricPvalue", "poissonPvalue",
            "poissonPvaluePeakA", "poissonPvaluePeakB", "significant"
        ]
        fo.write("\t".join(header) + "\n")
        for i, loop in enumerate(loops):
            if hasattr(loop, "id"):
                lid = loop.id
            else:
                lid = "loop_%s-%s-%s" % (loop.chromX, loop.chromY, i)
            line = [
                lid,
                loop.chromX,
                loop.x_start,
                loop.x_end,
                loop.chromY,
                loop.y_start,
                loop.y_end,
                loop.distance,
                loop.x_center,
                loop.y_center,
                loop.ra,
                loop.rb,
                loop.cis,
                loop.rab,
                loop.density,
                loop.ES,
                loop.P2LL,
                loop.FDR,
                loop.binomial_p_value,
                loop.hypergeometric_p_value,
                loop.poisson_p_value,
                loop.x_peak_poisson_p_value,
                loop.y_peak_poisson_p_value,
                loop.significant,
            ]
            fo.write("\t".join(list(map(str, line))) + "\n")


def loops2ucscTxt(loops, fout, significant=1):
    """
    Convert loops to UCSC interact records. 
    The txt file format according to https://genome.ucsc.edu/goldenPath/help/interact.html
    @param fin: interactions in loop file
    @param significant: if set 1, only convert significant loops.
    """
    print("Converting %s loops to UCSC interact track." % (len(loops)))
    with open(fout, "w") as f:
        f.write('track type=interact name="loops" description="loops" interactDirectional=false visibility=full\n')
        line = [
            "#chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "value",
            "exp",
            "color",
            "sourceChrom",
            "sourceStart",
            "sourceEnd",
            "sourceName",
            "sourceStrand",
            "targetChrom",
            "targetStart",
            "targetENd",
            "targetName",
            "targetStrand",
        ]
        f.write("\t".join(line) + "\n")
        for i, loop in enumerate(loops):
            #only using significant results
            if significant > 0 and loop.significant < 1:
                continue
            nline = [
                loop.chromX, loop.x_start, loop.y_end,
                "loop_%s-%s-%s" % (loop.chromX, loop.chromY, i), int(loop.rab),
                loop.density,".", "#0000FF", loop.chromX, loop.x_start, loop.x_end,
                ".", ".",
                loop.chromY, loop.y_start, loop.y_end,
                ".", "."
            ]
            f.write("\t".join(list(map(str, nline))) + "\n")
    print("Converting to UCSC interact track finished for %s." % fout)


def loops2juiceTxt(loops, fout, significant=1):
    """
    Convert loops to Juicebox 2D annotation features. 
    The txt file format according to https://github.com/theaidenlab/juicebox/wiki/Loading-Annotations-(Annotations-menu)
    @param fin: interactions in loop file
    @param fout: washU  long-range interaction text file prefix
    @param significant: if set 1, only convert significant loops.
    all p-values are -log10(p) transformed to escape all shown as 0 in juicebox.
    """
    print("Converting %s loops to Juicebox 2D annotation feature." %
          (len(loops)))
    with open(fout, "w") as f:
        line = [
            "chromosome1", "x1", "x2", "chromosome2", "y1", "y2", "color",
            "observed", "loopId", "FDR", "EnrichmentScore", "P2LL", "distance",
            "counts_X", "counts_Y", "density", "-log10(binomal_p-value)",
            "-log10(poisson_p-value)", "-log10(hypergeometric_p-value)",
            "-log10(poisson_p-value_peak_x)", "-log10(poisson_p-value_peak_y)"
        ]
        f.write("\t".join(line) + "\n")
        for i, loop in enumerate(loops):
            #only using significant results
            if significant > 0 and loop.significant < 1:
                continue
            nline = [
                loop.chromX,
                loop.x_start,
                loop.x_end,
                loop.chromY,
                loop.y_start,
                loop.y_end,
                '"0,255,255"',
                loop.rab,
                "%s-%s-%s" % (loop.chromX, loop.chromY, i),
                loop.FDR,
                loop.ES,
                loop.P2LL,
                loop.distance,
                loop.ra,
                loop.rb,
                loop.density,
                -np.log10(loop.binomial_p_value),
                -np.log10(loop.poisson_p_value),
                -np.log10(loop.hypergeometric_p_value),
                -np.log10(loop.x_peak_poisson_p_value),
                -np.log10(loop.y_peak_poisson_p_value),
            ]
            f.write("\t".join(list(map(str, nline))) + "\n")
    print("Converting to Juicebox 2D annotation feature finished for %s." %
          fout)


def loops2washuTxt(loops, fout, significant=1):
    """
    Convert interaction level loop file to washU long range interactions. 
    Track format according to http://wiki.wubrowse.org/Long-range
    @param fin: interactions in loop file
    @param fout: washU long-range interaction text file prefix
    @param significant: if set 1, only convert significant loops.
    """
    print("Converting %s loops to washU long range interaction track." %
          len(loops))
    with open(fout, "w") as f:
        for i, loop in enumerate(loops):
            #only using significant results
            if significant > 0 and loop.significant < 1:
                continue
            nline = [
                "%s:%s-%s" % (loop.chromX, loop.x_start, loop.x_end),
                "%s:%s-%s" % (loop.chromY, loop.y_start, loop.y_end), "1"
            ]
            f.write("\t".join(map(str, nline)) + "\n")
    print("Converted to washU long range interaction track %s finished." %
          fout)


def loops2NewWashuTxt(loops, fout, significant=1):
    """
    Convert interaction level loop file to washU long range interactions. 
    Track format according to https://epigenomegateway.readthedocs.io/en/latest/tracks.html#long-range-chromatin-interaction
    @param fin: interactions in loop file
    @param fout: washU long-range interaction text file prefix
    @param significant: if set 1, only convert significant loops.
    """
    print("Converting %s loops to new washU long range interaction track." %
          len(loops))
    with open(fout, "w") as f:
        for i, loop in enumerate(loops):
            #only using significant results
            if significant > 0 and loop.significant < 1:
                continue
            linea = [loop.chromX, loop.x_start,loop.x_end,"%s:%s-%s,%s"%(loop.chromY,loop.y_start,loop.y_end,loop.rab)]
            f.write("\t".join(map(str, linea)) + "\n")
            lineb = [loop.chromY, loop.y_start,loop.y_end,"%s:%s-%s,%s"%(loop.chromX,loop.x_start,loop.x_end,loop.rab)]
            f.write("\t".join(map(str, lineb)) + "\n")
    print("Converted to new washU long range interaction track %s finished." %
          fout)


def parseTxt2Loops(f, cut=0, mcut=-1, mode="cis"):
    """Parse a loop text file into :class:`cLoops2.ds.Loop` objects.

    ``cut`` and ``mcut`` are genomic-distance filters and therefore only
    apply to intra-chromosomal loops.  Inter-chromosomal loops have no
    genomic distance; cLoops2 writes ``-1`` for that field.  The previous
    implementation applied ``cut`` before classifying the loop and silently
    discarded every standard trans loop when ``cut`` was left at its default
    value of zero.

    Parameters
    ----------
    f : str
        Loop text file.
    cut, mcut : int
        Minimum and maximum distance for cis loops.  A non-positive ``mcut``
        disables the upper bound.
    mode : {"cis", "trans", "all"}
        Select which loop categories to return.  The default remains cis so
        existing cis-oriented consumers do not start receiving trans records
        merely because their input file contains them.

    Notes
    -----
    Dictionary keys retain the historical ``chromX-chromY`` representation
    for API compatibility.  New metadata must store ``chromX`` and ``chromY``
    as separate fields; this parser must not be used to reverse an ambiguous
    key back into chromosome names.
    """
    if mode not in ("cis", "trans", "all"):
        raise ValueError("mode must be one of: cis, trans, all")
    loops = {}
    with open(f) as handle:
        for i, line in enumerate(handle):
            line = line.split("\n")[0].split("\t")
            if not line or not line[0]:
                continue
            if line[0].startswith("#") or line[0] == "loopId":
                continue
            if len(line) < 8:
                continue
            loop = Loop()
            loop.id = line[0]
            loop.chromX = line[1]
            loop.x_start = int(float(line[2]))
            loop.x_end = int(float(line[3]))
            loop.chromY = line[4]
            loop.y_start = int(float(line[5]))
            loop.y_end = int(float(line[6]))
            loop.cis = loop.chromX == loop.chromY
            if mode == "cis" and not loop.cis:
                continue
            if mode == "trans" and loop.cis:
                continue
            distance = line[7].strip().lower()
            if distance in ("na", "nan", "."):
                if loop.cis:
                    raise ValueError("cis loop %s has no genomic distance" %
                                     loop.id)
                loop.distance = -1
            else:
                loop.distance = int(float(line[7]))
            if loop.cis:
                if loop.distance < cut:
                    continue
                if mcut > 0 and loop.distance > mcut:
                    continue
            key = loop.chromX + "-" + loop.chromY
            loop.x_center = (loop.x_start + loop.x_end) / 2
            loop.y_center = (loop.y_start + loop.y_end) / 2
            loops.setdefault(key, []).append(loop)
    return loops



def dloops2txt(dloops, fout):
    """
    Converting list of cLoops2.ds.DiffLoops objects into txt file.
    """
    with open(fout, "w") as fo:
        header = [
            "loopId",
            "chrA", 
            "startA", 
            "endA", 
            "chrB", 
            "startB", 
            "endB",
            "distance(bp)",
            "centerA", 
            "centerB", 
            "rawTargetAnchorAReads",
            "rawTargetAnchorBReads",
            "rawControlAnchorAReads",
            "rawControlAnchorBReads",
            "scaledTargetAnchorAReads",
            "scaledTargetAnchorBReads",
            "rawTargetCounts",
            "scaledTargetCounts",
            "rawControlCounts", 
            "rawTargetNearbyMedianCounts",
            "scaledTargetNearbyMedianCounts",
            "rawControlNearbyMedianCounts", 
            "rawTargetES",
            "rawControlES",
            "targetDensity",
            "controlDensity", 
            "rawFc", 
            "scaledFc", 
            "poissonPvalue",
            "significant"
        ]
        fo.write("\t".join(header) + "\n")
        for loop in dloops:
            line = [
                loop.id,
                loop.chromX,
                loop.x_start,
                loop.x_end,
                loop.chromY,
                loop.y_start,
                loop.y_end,
                loop.distance,
                loop.x_center,
                loop.y_center,
                loop.raw_trt_ra,
                loop.raw_trt_rb,
                loop.raw_con_ra,
                loop.raw_con_rb,
                loop.scaled_trt_ra,
                loop.scaled_trt_rb,
                loop.raw_trt_rab,
                loop.scaled_trt_rab,
                loop.raw_con_rab,
                loop.raw_trt_mrab,
                loop.scaled_trt_mrab,
                loop.raw_con_mrab,
                loop.trt_es,
                loop.con_es,
                loop.trt_density,
                #loop.trt_ipk,
                #loop.scaled_trt_ipk,
                #loop.scaled_trt_nearby_ipk,
                loop.con_density,
                #loop.con_ipk,
                #loop.con_nearby_ipk,
                loop.raw_fc,
                loop.scaled_fc,
                loop.poisson_p_value,
                loop.significant,
            ]
            fo.write("\t".join(list(map(str, line))) + "\n")



def dloops2juiceTxt(loops, fout, significant=1):
    """
    Convert DiffLoop to Juicebox 2D annotation features. 
    The txt file format according to https://github.com/theaidenlab/juicebox/wiki/Loading-Annotations-(Annotations-menu)
    @param fin: interactions in loop file
    @param fout: washU  long-range interaction text file prefix
    @param significant: if set 1, only convert significant loops.
    all p-values are -log10(p) transformed to escape all shown as 0 in juicebox.
    """
    print("Converting %s loops to Juicebox 2D annotation feature." %
          (len(loops)))
    with open(fout, "w") as f:
        line = [
            "chromosome1", 
            "x1", 
            "x2", 
            "chromosome2", 
            "y1", 
            "y2", 
            "color",
            "loopId",
            "distance",
            "rawTargetCounts",
            "scaledTargetCounts",
            "rawControlCounts", 
            "rawTargetNearbyMeanCounts",
            "scaledTargetNearbyMeanCounts",
            "rawControlNearbyMeanCounts", 
            "rawTargetES",
            "rawControlES",
            "targetDensity",
            "controlDensity", 
            "rawFc", 
            "scaledFc", 
            "poissonPvalue",
        ]
        f.write("\t".join(line) + "\n")
        for i, loop in enumerate(loops):
            #only using significant results
            if significant > 0 and loop.significant < 1:
                continue
            nline = [
                loop.chromX,
                loop.x_start,
                loop.x_end,
                loop.chromY,
                loop.y_start,
                loop.y_end,
                '"0,255,255"',
                loop.id,
                loop.distance,
                loop.raw_trt_rab,
                loop.scaled_trt_rab,
                loop.raw_con_rab,
                loop.raw_trt_mrab,
                loop.scaled_trt_mrab,
                loop.raw_con_mrab,
                loop.trt_es,
                loop.con_es,
                loop.trt_density,
                loop.con_density,
                loop.raw_fc,
                loop.scaled_fc,
                loop.poisson_p_value,
            ]
            f.write("\t".join(list(map(str, nline))) + "\n")
    print("Converting to Juicebox 2D annotation feature finished for %s." %
          fout)



def dloops2NewWashuTxt(loops, fout, significant=1):
    """
    Convert interaction level loop file to washU long range interactions. 
    Track format according to https://epigenomegateway.readthedocs.io/en/latest/tracks.html#long-range-chromatin-interaction
    @param fin: interactions in loop file
    @param fout: washU long-range interaction text file prefix
    @param significant: if set 1, only convert significant loops.
    """
    print("Converting %s loops to new washU long range interaction track." %
          len(loops))
    with open(fout, "w") as f:
        for i, loop in enumerate(loops):
            #only using significant results
            if significant > 0 and loop.significant < 1:
                continue
            linea = [loop.chromX, loop.x_start,loop.x_end,"%s:%s-%s,1"%(loop.chromY,loop.y_start,loop.y_end)]
            f.write("\t".join(map(str, linea)) + "\n")
            lineb = [loop.chromY, loop.y_start,loop.y_end,"%s:%s-%s,1"%(loop.chromX,loop.x_start,loop.x_end)]
            f.write("\t".join(map(str, lineb)) + "\n")
    print("Converted to new washU long range interaction track %s finished." %
          fout)



#domains releated io
def doms2txt(doms, fout):
    """
    Converting list of cLoops2.ds.Domain objects into txt file.
    """
    with open(fout, "w") as fo:
        header = [
            "domainId", "chr", "start", "end", "length", "binSize","winSize",
            "segregationScore", "totalPETs", "withinDomainPETs",
            "enrichmentScore","density","significant"
        ]
        fo.write("\t".join(header) + "\n")
        for i, dom in enumerate(doms):
            if hasattr(dom, "id"):
                did = dom.id
            else:
                did = "domain_%s" % i
            line = [
                did,
                dom.chrom,
                dom.start,
                dom.end,
                dom.length,
                dom.bs,
                dom.ws,
                dom.ss,
                dom.totalPETs,
                dom.withinDomainPETs,
                dom.enrichmentScore,
                dom.density,
                dom.significant,
            ]
            fo.write("\t".join(list(map(str, line))) + "\n")


def doms2bed(doms, fout):
    """
    Converting cLoops2.ds.Domain objects into BED file.
    @param sig: bool, if True, only write significant peaks to file.
    """
    with open(fout, "w") as fo:
        for i, dom in enumerate(doms):
            if dom.significant:
                info = "domain_%s;%sbp;ES:%.3f;SS:%.3f;binSize:%s;winSize:%s" % (
                    i, dom.length, dom.enrichmentScore, dom.ss, dom.bs,dom.ws)
                line = [dom.chrom, dom.start, dom.end, info]
                fo.write("\t".join(list(map(str, line))) + "\n")


def parseTxt2Domains(f):
    """
    Parse _domain.txt file into cLoops2:ds:Domain objects.
    """
    domains = {}
    for i, line in enumerate(open(f)):
        if i == 0:
            continue
        line = line.split("\n")[0].split("\t")
        if len(line) < 3:
            continue
        domain = Domain()
        domain.id = line[0]
        domain.chrom = line[1]
        domain.start = int(float(line[2]))
        domain.end = int(float(line[3]))
        domain.length = domain.end - domain.start
        domain.significant = "NO-TEST"
        key = domain.chrom + "-" + domain.chrom
        domains.setdefault(key, []).append(domain)
    return domains
