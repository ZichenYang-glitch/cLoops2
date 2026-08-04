#!/usr/bin/env python3
#--coding:utf-8--
"""
extract_target_pets.py
Extract PETs related to a gene (by name) or a genomic region from a
bedpe/bedpe.gz file, and write the result as a TSV file.

"Related" means one end of a PET overlaps the target:
  - mode "body":     the whole gene span (or the given region)
  - mode "promoter": a window around the TSS (gene mode only, needs GTF)

Examples:
    # trans PETs linked to the EGFR promoter (TSS upstream 2kb / downstream 0.5kb)
    extract_target_pets.py -f sample_unique.bedpe.gz -g EGFR \
        --gtf gencode.v38.annotation.gtf --mode promoter --pet-type trans \
        -o EGFR_promoter_trans.tsv

    # all PETs overlapping a plain region
    extract_target_pets.py -f sample_unique.bedpe.gz \
        -r chr7:55019017-55211628 -o region.tsv
"""

import argparse
import gzip
import re
import sys


def help():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-f", "--file", required=True,
                   help="Input .bedpe or .bedpe.gz file.")
    p.add_argument("-g", "--gene", default=None,
                   help="Gene name (requires --gtf).")
    p.add_argument("-r", "--region", default=None,
                   help="Region as chr:start-end (commas allowed). "
                        "Body mode only.")
    p.add_argument("--gtf", default=None,
                   help="GTF annotation used to resolve the gene name.")
    p.add_argument("--mode", choices=["body", "promoter"], default="body",
                   help="body: whole gene span / given region; "
                        "promoter: TSS window (default).")
    p.add_argument("--up", type=int, default=2000,
                   help="Promoter: bp upstream of TSS (default 2000).")
    p.add_argument("--down", type=int, default=500,
                   help="Promoter: bp downstream of TSS (default 500).")
    p.add_argument("--pet-type", choices=["all", "cis", "trans"],
                   default="all", help="Keep all/cis/trans PETs (default all).")
    p.add_argument("-o", "--output", default=None,
                   help="Output TSV (default: stdout).")
    op = p.parse_args()
    if not op.gene and not op.region:
        p.error("one of -g/--gene or -r/--region is required")
    if op.gene and not op.gtf:
        p.error("--gtf is required when using -g/--gene")
    if op.region and op.mode == "promoter":
        p.error("promoter mode needs a gene (TSS from --gtf); "
                "region input only supports body mode")
    return op


def parse_region(s):
    m = re.match(r"^([^:]+):([\d,]+)-([\d,]+)$", s.strip())
    if not m:
        sys.exit("Cannot parse region '%s', expected chr:start-end" % s)
    return m.group(1), int(m.group(2).replace(",", "")), int(
        m.group(3).replace(",", ""))


def parse_gtf_attrs(text):
    attrs = {}
    for item in text.strip().strip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        m = re.match(r'(\S+)\s+"([^"]*)"', item)
        if m:
            attrs[m.group(1)] = m.group(2)
        else:
            kv = item.split(None, 1)
            if len(kv) == 2:
                attrs[kv[0]] = kv[1]
    return attrs


def find_gene(gtf_path, name):
    """
    Resolve a gene name to (chrom, start0, end, strand) using 'gene'
    features; fall back to merged 'transcript' features.
    start0 is 0-based, end is 1-based-exclusive-ish (GTF end).
    """
    genes = {}
    transcripts = {}
    names = set()
    opener = gzip.open if gtf_path.endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            feature = cols[2]
            if feature not in ("gene", "transcript"):
                continue
            attrs = parse_gtf_attrs(cols[8])
            gname = attrs.get("gene_name") or attrs.get("gene_id")
            if not gname:
                continue
            names.add(gname)
            if gname != name:
                continue
            start, end = int(cols[3]) - 1, int(cols[4])
            rec = (cols[0], start, end, cols[6])
            if feature == "gene":
                genes.setdefault(gname, []).append(rec)
            else:
                transcripts.setdefault(gname, []).append(rec)
    pool = genes.get(name) or transcripts.get(name)
    if not pool:
        similar = sorted(n for n in names if name.lower() in n.lower())[:10]
        hint = (" similar names: %s" % ",".join(similar)) if similar else ""
        sys.exit("Gene '%s' not found in %s.%s" % (name, gtf_path, hint))
    chroms = set(r[0] for r in pool)
    strands = set(r[3] for r in pool)
    if len(chroms) > 1:
        sys.exit("Gene '%s' maps to multiple chromosomes: %s" %
                 (name, ",".join(sorted(chroms))))
    start = min(r[1] for r in pool)
    end = max(r[2] for r in pool)
    strand = strands.pop() if len(strands) == 1 else "."
    return chroms.pop(), start, end, strand


def target_window(op):
    """Return (chrom, win_start, win_end, tss_or_none, strand, label)."""
    if op.region:
        chrom, start, end = parse_region(op.region)
        return chrom, start, end, None, ".", op.region
    chrom, start, end, strand = find_gene(op.gtf, op.gene)
    if op.mode == "body":
        label = "%s|body|%s:%d-%d" % (op.gene, chrom, start, end)
        return chrom, start, end, None, strand, label
    # promoter from TSS
    tss = start if strand != "-" else end
    if strand == "-":
        win_start, win_end = tss - op.down, tss + op.up
    else:
        win_start, win_end = tss - op.up, tss + op.down
    win_start = max(0, win_start)
    label = "%s|promoter(%s)|%s:%d-%d|TSS=%d" % (
        op.gene, strand, chrom, win_start, win_end, tss)
    return chrom, win_start, win_end, tss, strand, label


def overlaps(s, e, ws, we):
    return s < we and e > ws


def main():
    op = help()
    rchrom, ws, we, tss, strand, label = target_window(op)
    fi = gzip.open(op.file, "rt") if op.file.endswith(".gz") else open(op.file)
    fo = open(op.output, "w") if op.output else sys.stdout
    header = [
        "chrom1", "start1", "end1", "chrom2", "start2", "end2",
        "name", "score", "strand1", "strand2",
        "hit_end", "target", "partner_chrom", "partner_start", "partner_end",
        "partner_dist_to_tss",
    ]
    fo.write("\t".join(header) + "\n")
    kept = total = 0
    with fi:
        for line in fi:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            try:
                c1, s1, e1 = cols[0], int(cols[1]), int(cols[2])
                c2, s2, e2 = cols[3], int(cols[4]), int(cols[5])
            except ValueError:
                continue
            total += 1
            if op.pet_type == "trans" and c1 == c2:
                continue
            if op.pet_type == "cis" and c1 != c2:
                continue
            hit1 = (c1 == rchrom and overlaps(s1, e1, ws, we))
            hit2 = (c2 == rchrom and overlaps(s2, e2, ws, we))
            if not (hit1 or hit2):
                continue
            # pad short rows so name/score/strand columns line up
            row = cols + [""] * (10 - len(cols)) if len(cols) < 10 else cols
            if hit1:
                hit, pc, ps, pe = "end1", c2, s2, e2
            else:
                hit, pc, ps, pe = "end2", c1, s1, e1
            dist = ""
            if tss is not None:
                dist = str(((ps + pe) // 2) - tss)
            kept += 1
            fo.write("\t".join(map(str, row)) + "\t" +
                     "\t".join(map(str, [hit, label, pc, ps, pe, dist])) + "\n")
    if op.output:
        fo.close()
    sys.stderr.write("Target %s | scanned %d PETs, kept %d\n" %
                     (label, total, kept))


if __name__ == "__main__":
    main()
