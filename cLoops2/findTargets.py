#!/usr/bin/env python3
#--coding:utf-8 --
"""
findTargets.py 
cLoops2 loops-centric analysis module. 
Find the target genes for a set of regions, such as loops anchors or SNPs. 
- []
"""

#sys

#3rd
import pandas as pd
import networkx as nx
from tqdm import tqdm

#cLoops2
from cLoops2.ds import Peak
from cLoops2.io import parseTxt2Loops


def parseIv(item):
    chrom = item.split(":")[0]
    start = int(item.split("|")[0].split(":")[1].split("-")[0])
    end = int(item.split("|")[0].split(":")[1].split("-")[1])
    return chrom, start, end


def readNet(f):
    """
    Read the enhancer-promoter networks.
    @return nx.Graph 
    @return cov, {chrom:{i:itemId}}
    """
    cov = {}
    ns = set()
    G = nx.Graph()
    with open(f) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n").split("\t")
            if len(line) < 3 or not line[0] or line[0].startswith("#"):
                continue
            G.add_edge(line[0], line[2], type=line[1])
            #left anchor coverage
            if line[0] not in ns:
                lc, ls, le = parseIv(line[0])
                if lc not in cov:
                    cov[lc] = {}
                for i in range(ls, le + 1):
                    cov[lc][i] = line[0]
                ns.add(line[0])
            #right anchor coverage
            if line[2] not in ns:
                rc, rs, re = parseIv(line[2])
                if rc not in cov:
                    cov[rc] = {}
                for i in range(rs, re + 1):
                    cov[rc][i] = line[2]
                ns.add(line[2])
    for node in G.nodes():
        n = node.split("|")[-1]
        G.nodes[node]["type"] = n
    return G, cov


def readTargets(f):
    """
    Read the promoter target genes.
    """
    ds = {}
    with open(f) as handle:
        for i, line in enumerate(handle):
            if i == 0:
                continue
            line = line.split("\n")[0].split("\t")
            if len(line) >= 2:
                ds[line[0]] = line[1]
    return ds


def readBed(f):
    """
    Read regions
    """
    regions = []
    with open(f) as handle:
        for line in handle:
            line = line.split("\n")[0].split("\t")
            if len(line) < 3:
                continue
            if len(line) > 3 and line[3] not in ("", "."):
                k = line[3]
            else:
                k = "|".join(line[:3])
            peak = Peak()
            peak.chrom = line[0]
            peak.start = int(line[1])
            peak.end = int(line[2])
            peak.id = k
            regions.append(peak)
    return regions


def getTargets(G, cov, tgs, rs, fnOut):
    """
    Get region targets through enhancer promoter linkage network.
    """
    j = 0
    k = 0
    ds = {}
    pathes = {}
    for r in tqdm(rs):
        ts = set()
        if r.chrom not in cov:
            continue
        for i in range(r.start, r.end + 1):
            if i in cov[r.chrom]:
                ts.add(cov[r.chrom][i])
        #searching the net for targets
        if len(ts) == 0:
            continue
        for t in ts:
            node_type = G.nodes[t].get("type", t.split("|")[-1])
            if node_type == "Promoter":
                #direct targets
                dt = [t]
                #indirect targets
                idts = {}
                paths = nx.single_source_shortest_path(G, t)
                #find all releated nodes
                for n, p in paths.items():
                    if n == t:
                        continue
                    if G.nodes[n].get("type", n.split("|")[-1]) == "Promoter":
                        idts[n] = p
            elif node_type == "Enhancer":
                dt = []
                idts = {}
                paths = nx.single_source_shortest_path(G, t)
                #find all releated nodes
                for n, p in paths.items():
                    if n == t:
                        continue
                    #if n.split("|")[-1] == "Promoter" and len(p) > pathLengthCut:
                    if G.nodes[n].get("type", n.split("|")[-1]) == "Promoter":
                        if len(p) == 2:
                            dt.append(n)
                        else:
                            idts[n] = p
            else:
                # An anchor on a chromosome absent from the supplied GTF is
                # retained in the network but cannot establish gene targets.
                continue
            if len(dt) == 0 and len(idts) == 0:
                continue
            dt = [tgs[tmp] for tmp in dt if tmp in tgs]
            idt = [tgs[tmp] for tmp in idts.keys() if tmp in tgs]
            ds[j] = {
                "queryId": r.id,
                "queryChrom": r.chrom,
                "queryStart": r.start,
                "queryEnd": r.end,
                "overlappedAnchor": t,
                "directTargetGenes": ",".join(dt),
                "indirectTargetGenes": ",".join(idt),
            }
            j += 1
            #record pathes
            if len(idt) > 0:
                for g, p in idts.items():
                    if g in tgs:
                        pathes[k] = {
                            "queryId": r.id,
                            "overlappedAnchor": t,
                            "indirectTargetGenes": tgs[g],
                            "path": ",".join(p),
                        }
                        k += 1
    target_columns = [
        "queryId", "queryChrom", "queryStart", "queryEnd",
        "overlappedAnchor", "directTargetGenes", "indirectTargetGenes"
    ]
    path_columns = [
        "queryId", "overlappedAnchor", "indirectTargetGenes", "path"
    ]
    ds = pd.DataFrame.from_dict(ds, orient="index", columns=target_columns)
    pathes = pd.DataFrame.from_dict(
        pathes, orient="index", columns=path_columns)
    ds.to_csv(fnOut + "_targetGenes.txt", sep="\t", index_label="recordId")
    pathes.to_csv(fnOut + "_indirectTargetGenesPathes.txt",
                  sep="\t",
                  index_label="recordId")


def findTargets(
        netf,
        tgf,
        fnOut,
        fbed="",
):
    """
    Find targets of a set of regions.
    @param netf: str, output of cLoops2 anaLoops, _ep_net.sif file
    @param tgf: str, output of cLoops2 anaLoops, _targets.txt file
    @param bed: str, input querying bed file.
    """
    print("reading networks and anchors")
    G, cov = readNet(netf)
    tgs = readTargets(tgf)
    if fbed != "":
        print("finding target genes of %s" % fbed)
        #find regions targets
        rs = readBed(fbed)
        getTargets(G, cov, tgs, rs, fnOut)
