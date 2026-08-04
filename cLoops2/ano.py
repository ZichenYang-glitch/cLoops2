#!/usr/bin/env python3
#--coding:utf-8 --
"""
ano.py 

Previouse analyzeLoops.py, now changed to ano.py to also annotate peaks/domains. 

Include cLoops2 loops-centric analysis module. Input should be the _loops.txt file and other annotation files. Mainly contain following analysis. 
- [x] loops annotation to target genes as enhancer and promoter 
- [x] loops annotation to target genes through network method
- [x] find HUBs through HITS algorithm

2020-11-04: modified to first find overlapped TSS, if no or multiple, then find the closest one.
2021-01-21: peaks annotation going to be added.
"""

#sys
import os
from collections import defaultdict

#3rd
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy.spatial import KDTree
from joblib import Parallel, delayed

#cLoops2
from cLoops2.ds import Exon, Gene, Peak
from cLoops2.io import parseTxt2Loops


def parseGtfLine(line, tid=False):
    """
    Parse gene gtf line.
    """
    e = Exon()
    e.chrom = line[0]
    e.start = int(line[3])
    e.end = int(line[4])
    e.length = e.end - e.start
    e.strand = line[6]
    attr = line[8].replace('"', '').split(";")
    ts = {}
    for t in attr:
        t = t.split()
        if len(t) != 2:
            continue
        ts[t[0]] = t[1]
    if tid:
        try:
            e.name = ts["transcript_name"]
        except:
            e.name = ts["gene_name"]
    else:
        e.name = ts["gene_name"]
    if tid:
        e.id = ts["transcript_id"]
    else:
        e.id = ts["gene_id"]
    return e


def readGenes(gtf, tid=False):
    """
    Read gene annotion file for genes or transcripts
    """
    gs = {}
    #get all genes information
    print("reading annotaions from %s" % gtf)
    with open(gtf) as handle:
        lines = handle.read().splitlines()
    for line in tqdm(lines):
        if line.startswith("#"):
            continue
        line = line.split("\n")[0].split("\t")
        if line[2] != "exon":
            continue
        e = parseGtfLine(line, tid)
        # Gene/transcript names are not guaranteed to be globally unique.
        # Include chromosome and stable id so a same-named feature on chromY
        # cannot be merged into the chromX record during trans annotation.
        gene_key = (e.chrom, e.id, e.name)
        if gene_key not in gs:
            g = Gene()
            g.chrom = e.chrom
            g.start = e.start
            g.end = e.end
            g.strand = e.strand
            g.name = e.name
            g.id = e.id
            g.exons = {(e.start, e.end): e}
            gs[gene_key] = g
        else:
            #same position exons
            if (e.start, e.end) in gs[gene_key].exons:
                continue
            else:
                g = gs[gene_key]
                if e.start < g.start:
                    g.start = e.start
                if e.end > g.end:
                    g.end = e.end
                g.exons[(e.start, e.end)] = e
    #get all genes information
    ngs = {}  #key is chromosome
    for k, g in gs.items():
        if g.chrom not in ngs:
            ngs[g.chrom] = {}
        if g.strand == "+":
            tss = g.start
        else:
            tss = g.end
        #tss position is key, other information is value, for following search
        if tss not in ngs[g.chrom]:
            ngs[g.chrom][tss] = g
    return ngs


def findOverlapOrNearest(gs, ts, tree, start, end):
    """
    first to check direct overlap with TSS, if no or multiple items, then get the close one
    @param gs: {tss: cLoops2.ds.Gene}, tss is key and int
    @pram ts: [tss]
    @param tree: KDTree from TSSs
    @param start: query start
    @param end: query end

    return gene and distance
    """
    #step 1, find overlaps
    rs = set()
    for i in range(start, end + 1):
        if i in gs:
            rs.add(gs[i])
    if len(rs) > 0:
        rs = list(rs)
        return rs, [0] * len(rs)
    #find the nearest one
    else:
        d, i = tree.query([(start + end) / 2], k=1)
        g = gs[ts[i][0]]
        #d = ts[i][0] - (start+end)/2
        d = int(d)
        return [g], [d]


def findNearestTss(chrom, loops, gs, pdis=2000):
    """
    Find nearest TSS for loop anchors. 
    @param loops: list of cLoops2.ds.Loop
    @param gs: {tss: cLoops2.ds.Gene}, tss is key and int
    """
    ts = np.array([[tss] for tss in gs.keys()])
    cov = {}
    for tss, g in gs.items():
        cov[tss] = g
    tree = KDTree(ts)
    ds = {}
    for loop in loops:
        xgs, xds = findOverlapOrNearest(gs, ts, tree, loop.x_start, loop.x_end)
        ygs, yds = findOverlapOrNearest(gs, ts, tree, loop.y_start, loop.y_end)
        if len(xgs) > 1:
            xt = "Promoter"
            xd = 0
        else:
            xd = xds[0]
            if abs(xd) <= pdis:
                xt = "Promoter"
            else:
                xt = "Enhancer"
        if len(ygs) > 1:
            yt = "Promoter"
            yd = 0
        else:
            yd = yds[0]
            if abs(yd) <= pdis:
                yt = "Promoter"
            else:
                yt = "Enhancer"
        ds[loop.id] = {
            "typeAnchorA":
            xt,
            "typeAnchorB":
            yt,
            "nearestDistanceToGeneAnchorA":
            xd,
            "nearestDistanceToGeneAnchorB":
            yd,
            "nearestTargetGeneAnchorA":
            ",".join([
                xg.chrom + ":" + str(xg.start) + "-" + str(xg.end) + "|" +
                xg.strand + "|" + xg.name for xg in xgs
            ]),
            "nearestTargetGeneAnchorB":
            ",".join([
                yg.chrom + ":" + str(yg.start) + "-" + str(yg.end) + "|" +
                yg.strand + "|" + yg.name for yg in ygs
            ]),
        }
    return ds


def annotateLoopToGenes(loops, genes, fout, pdis=2000, cpu=1):
    """
    Annotate loops releative to genes. 
    @param loops: { "chrom-chrom":[] }, in list are cLoops2.ds.Loop
    """
    print("Annotating loops to enhancers and promoters.")
    ks = [key for key in loops.keys() if key in genes]
    ds = Parallel(n_jobs=cpu,
                  backend="multiprocessing")(delayed(findNearestTss)(
                      chrom,
                      loops[chrom],
                      genes[chrom],
                      pdis=pdis,
                  ) for chrom in tqdm(ks))
    rs = {}
    for d in ds:
        for k, v in d.items():
            rs[k] = v
    rs = pd.DataFrame(rs).T
    fo = fout + "_LoopsGtfAno.txt"
    rs.to_csv(fo, sep="\t", index_label="loopId")
    return fo


def _iter_loops(loops):
    """Yield loops in a stable order from a legacy loop mapping or sequence."""
    if isinstance(loops, dict):
        values = []
        for key in sorted(loops):
            values.extend(loops[key])
    else:
        values = list(loops)
    return sorted(values, key=lambda loop: (
        loop.chromX, int(loop.x_start), int(loop.x_end), loop.chromY,
        int(loop.y_start), int(loop.y_end), str(loop.id)))


def _annotate_anchor_axis(genes, chrom, start, end, pdis):
    """Annotate one anchor against TSSs from its own chromosome only."""
    chrom_genes = genes.get(chrom, {})
    if not chrom_genes:
        return {
            "type": "Unannotated",
            "distance": None,
            "genes": [],
            "gene_locations": "",
        }

    positions = np.asarray(sorted(chrom_genes), dtype=np.int64)
    left = int(np.searchsorted(positions, int(start), side="left"))
    right = int(np.searchsorted(positions, int(end), side="right"))
    if left < right:
        selected_positions = positions[left:right]
        distance = 0
    else:
        center = (int(start) + int(end)) / 2.0
        distances = np.abs(positions.astype(float) - center)
        minimum = float(np.min(distances))
        # Stable tie handling is useful for equidistant TSSs and independent of
        # scipy/KDTree implementation details.
        selected_positions = positions[np.flatnonzero(distances == minimum)]
        distance = int(minimum)

    selected = [chrom_genes[int(position)]
                for position in selected_positions]
    anchor_type = "Promoter" if distance <= int(pdis) else "Enhancer"
    locations = ",".join([
        gene.chrom + ":" + str(gene.start) + "-" + str(gene.end) + "|" +
        gene.strand + "|" + gene.name for gene in selected
    ])
    return {
        "type": anchor_type,
        "distance": distance,
        "genes": selected,
        "gene_locations": locations,
    }


def annotateLoopToGenesAxisAware(loops, genes, fout, pdis=2000, cpu=1):
    """Annotate cis/trans loop anchors in their respective chromosomes.

    Unlike :func:`annotateLoopToGenes`, this function never reuses one
    chromosome's TSS index for both anchors.  ``cpu`` is retained in the public
    signature for symmetry with the historical API; stable, lightweight TSS
    searches are performed in the current process.
    """
    del cpu
    print("Annotating loop anchors against chromosome-specific TSSs.")
    rows = {}
    for loop in _iter_loops(loops):
        if loop.id in rows:
            raise ValueError("duplicate loop id %r" % loop.id)
        anchor_a = _annotate_anchor_axis(
            genes, loop.chromX, loop.x_start, loop.x_end, pdis)
        anchor_b = _annotate_anchor_axis(
            genes, loop.chromY, loop.y_start, loop.y_end, pdis)
        rows[loop.id] = {
            "chromAnchorA": loop.chromX,
            "chromAnchorB": loop.chromY,
            "loopCategory": "cis" if loop.cis else "trans",
            "typeAnchorA": anchor_a["type"],
            "typeAnchorB": anchor_b["type"],
            "nearestDistanceToGeneAnchorA": anchor_a["distance"],
            "nearestDistanceToGeneAnchorB": anchor_b["distance"],
            "nearestTargetGeneAnchorA": anchor_a["gene_locations"],
            "nearestTargetGeneAnchorB": anchor_b["gene_locations"],
        }

    columns = [
        "chromAnchorA", "chromAnchorB", "loopCategory", "typeAnchorA",
        "typeAnchorB", "nearestDistanceToGeneAnchorA",
        "nearestDistanceToGeneAnchorB", "nearestTargetGeneAnchorA",
        "nearestTargetGeneAnchorB",
    ]
    result = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    output = fout + "_LoopsGtfAno.txt"
    result.to_csv(output, sep="\t", index_label="loopId")
    return output


def stichAnchors(chrom, loops, margin=1):
    """
    Stich close anchors based on postion array.
    """
    cov = set()
    for i, loop in enumerate(loops):
        cov.update(range(loop.x_start, loop.x_end + 1))
        cov.update(range(loop.y_start, loop.y_end + 1))
    cov = list(cov)
    cov.sort()
    npeaks = []
    i = 0
    while i < len(cov) - 1:
        j = i + 1
        while j < len(cov):
            if cov[j] - cov[j - 1] > margin:
                break
            else:
                j += 1
        peak = Peak()
        peak.chrom = chrom
        peak.start = cov[i]
        peak.end = cov[j - 1]
        peak.length = cov[j - 1] - cov[i] + 1
        npeaks.append(peak)
        i = j  #update search start
    return npeaks


def getNet(chrom, loops, genes, pdis=2000, gap=1):
    """
    Get the enhancer/promoter network for one chromosome.  
    """
    #step 1 get merged anchors
    anchors = stichAnchors(chrom, loops, margin=gap)
    #step 2 annotate anchors
    nanchors = {}
    ts = np.array([[tss] for tss in genes.keys()])
    tree = KDTree(ts)
    for anchor in anchors:
        gs, ds = findOverlapOrNearest(genes, ts, tree, anchor.start,
                                      anchor.end)
        if len(gs) > 1:
            t = "Promoter"
            d = 0
        else:
            d = ds[0]
            if abs(d) <= pdis:
                t = "Promoter"
            else:
                t = "Enhancer"
        n = anchor.chrom + ":" + str(anchor.start) + "-" + str(
            anchor.end) + "|" + t
        nanchors[n] = {
            "chrom":
            anchor.chrom,
            "start":
            anchor.start,
            "end":
            anchor.end,
            "type":
            n.split("|")[-1],
            "nearestDistanceToTSS":
            d,
            "nearestGene":
            ",".join([g.name for g in gs]),
            "nearestGeneLoc":
            ",".join([
                g.chrom + ":" + str(g.start) + "-" + str(g.end) + "|" +
                g.strand + "|" + g.name for g in gs
            ])
        }
    anchors = nanchors
    del nanchors
    #step 3 assign each anchor to merged annotated anchor and build the network
    anchorCov = {}
    for k, v in anchors.items():
        for i in range(v["start"], v["end"] + 1):
            anchorCov[i] = k
    ds = {}  #anchor annotations
    nets = {}  #net information
    G = nx.Graph()  #networkx graph structure
    for loop in loops:
        xt, yt = None, None
        for i in range(loop.x_start, loop.x_end + 1):
            if i in anchorCov:
                xt = anchorCov[i]
                break
        for i in range(loop.y_start, loop.y_end + 1):
            if i in anchorCov:
                yt = anchorCov[i]
                break
        ds[loop.id] = {
            "mergedAnchorA": xt,
            "mergedAnchorB": yt,
        }
        if xt == yt:
            continue
        ns = [xt, yt]
        ns.sort()  #sort for converging keys
        if ns[0] not in nets:
            nets[ns[0]] = set()
        nets[ns[0]].add(ns[1])
        #network edges
        G.add_edge(ns[0], ns[1])
    #step 4 find all enhancers linked to target gene
    targets = {}
    #step 4.1 find the direct enhancer that link to promoter
    for node in G.nodes:
        if node.split("|")[-1] == "Promoter":
            if node in targets:
                continue
            targets[node] = {
                "targetGene": anchors[node]["nearestGeneLoc"],
                "directEnhancer": set(),
                "indirectEnhancer": set(),
                "directPromoter": set(),
                "indirectPromoter": set()
            }
            ns = list(nx.descendants(G, node))
            #find all releated nodes
            for n in ns:
                p = nx.algorithms.shortest_path(G, source=node, target=n)
                if n.split("|")[-1] == "Promoter":
                    if len(p) == 2:
                        targets[node]["directPromoter"].add(n)
                    else:
                        targets[node]["indirectPromoter"].add(n)
                if n.split("|")[-1] == "Enhancer":
                    if len(p) == 2:
                        targets[node]["directEnhancer"].add(n)
                    else:
                        targets[node]["indirectEnhancer"].add(n)
            #step 4.2. find hub enhancer
            #only using non-redundant node to find hubs
            nns = []
            tmp = set()
            for n in ns:
                tn = n.split("|")[0]
                if tn not in tmp:
                    nns.append(n)
                tmp.add(tn)
            ns = list(nns)
            ns.append(node)
            subg = G.subgraph(ns)
            try:
                hubs, authorities = nx.hits(subg,
                                            max_iter=1000,
                                            normalized=True)
            except:
                print(
                    "For %s, hard to find the hub by running HITS algorithm of 1000 iteration."
                    % node)
                targets[node]["directEnhancerHub"] = ""
                targets[node]["indirectEnhancerHub"] = ""
                continue
            hubs = pd.Series(hubs)
            hubs = hubs.sort_values(inplace=False, ascending=False)
            if len(targets[node]["directEnhancer"]) >= 2:
                des = hubs[list(targets[node]["directEnhancer"])]
                des = des.sort_values(inplace=False, ascending=False)
                targets[node]["directEnhancerHub"] = des.index[0]
            else:
                targets[node]["directEnhancerHub"] = ""
            if len(targets[node]["indirectEnhancer"]) >= 2:
                indes = hubs[list(targets[node]["indirectEnhancer"])]
                indes = indes.sort_values(inplace=False, ascending=False)
                targets[node]["indirectEnhancerHub"] = indes.index[0]
            else:
                targets[node]["indirectEnhancerHub"] = ""
    return anchors, ds, nets, targets


def getNetworksFromLoops(loops, genes, fout, pdis=2000, gap=1, cpu=1):
    """
    Merge overlapped acnhors first then construct interaction network.
    """
    ks = [key for key in loops.keys() if key in genes]
    print("Merging anchors and annotating loops through networks.")
    ds = Parallel(n_jobs=cpu, backend="multiprocessing")(delayed(getNet)(
        chrom,
        loops[chrom],
        genes[chrom],
        pdis=pdis,
        gap=gap,
    ) for chrom in tqdm(ks))
    anchors, anots, nets, targets = {}, {}, {}, {}
    for d in ds:
        for k, v in d[0].items():
            anchors[k] = v
        for k, v in d[1].items():
            anots[k] = v
        for k, v in d[2].items():
            nets[k] = v
        for k, v in d[3].items():
            targets[k] = v
    #output results
    #anchors
    anchors = pd.DataFrame(anchors).T
    anchors.to_csv(fout + "_mergedAnchors.txt", sep="\t", index_label="anchor")
    with open(fout + "_mergedAnchors.bed", "w") as fo:
        for t in anchors.itertuples():
            line = [t[1], t[2], t[3], t[0]]
            fo.write("\t".join(list(map(str, line))) + "\n")
    #annotations
    anots = pd.DataFrame(anots).T
    anots.to_csv(fout + "_loop2anchors.txt", sep="\t", index_label="loopId")
    #networks
    with open(fout + "_ep_net.sif", "w") as fo:
        for s, es in nets.items():
            es = list(es)
            ta = s.split("|")[-1]
            for e in es:
                tb = e.split("|")[-1]
                t = [ta, tb]
                t.sort()
                t = "-".join(t)
                line = [s, t, e]
                fo.write("\t".join(line) + "\n")
    with open(fout + "_targets.txt", "w") as fo:
        ks = list(targets.keys())
        ks.sort()
        line = [
            "Promoter", "PromoterTarget", "directEnhancer", "indirectEnhancer",
            "directPromoter", "indirectPromoter", "directEnhancerHub",
            "indirectEnhancerHub"
        ]
        fo.write("\t".join(line) + "\n")
        for k in ks:
            line = [
                k, targets[k]["targetGene"],
                ",".join(targets[k]["directEnhancer"]),
                ",".join(targets[k]["indirectEnhancer"]),
                ",".join(targets[k]["directPromoter"]),
                ",".join(targets[k]["indirectPromoter"]),
                targets[k]["directEnhancerHub"],
                targets[k]["indirectEnhancerHub"]
            ]
            fo.write("\t".join(line) + "\n")


def _merge_axis_anchors(loops, gap):
    """Merge anchors independently within each chromosome."""
    intervals = defaultdict(list)
    for loop in _iter_loops(loops):
        intervals[loop.chromX].append((int(loop.x_start), int(loop.x_end)))
        intervals[loop.chromY].append((int(loop.y_start), int(loop.y_end)))

    merged = {}
    for chrom in sorted(intervals):
        current = []
        for start, end in sorted(intervals[chrom]):
            if start > end:
                raise ValueError("anchor start exceeds end on %s" % chrom)
            if not current or start - current[-1][1] > int(gap):
                current.append([start, end])
            else:
                current[-1][1] = max(current[-1][1], end)
        merged[chrom] = current
    return merged


def _anchor_node_for_interval(anchor_nodes, chrom, start, end):
    candidates = []
    for node, record in anchor_nodes.get(chrom, []):
        overlap = min(int(end), record["end"]) - max(int(start),
                                                       record["start"]) + 1
        if overlap > 0:
            candidates.append((-overlap, record["start"], record["end"], node))
    if not candidates:
        raise ValueError("no merged anchor covers %s:%s-%s" %
                         (chrom, start, end))
    candidates.sort()
    return candidates[0][-1]


def _stable_hub(graph, nodes):
    nodes = sorted(nodes)
    if len(nodes) < 2:
        return ""
    ranked = sorted(nodes, key=lambda node: (-graph.degree(node), node))
    return ranked[0]


def getNetworksFromLoopsAxisAware(loops, genes, fout, pdis=2000, gap=1,
                                  cpu=1):
    """Build one genome-wide cis/trans enhancer-promoter graph.

    Nodes always include their chromosome.  A trans loop therefore becomes a
    genuine edge between two chromosome-specific anchor nodes instead of being
    projected into one coordinate system.  The historical cis network API and
    files remain unchanged when ``anaLoops(mode='cis')`` is used.
    """
    del cpu
    print("Merging chromosome-specific anchors and building an axis-aware network.")
    loop_list = _iter_loops(loops)
    merged = _merge_axis_anchors(loop_list, gap)
    anchors = {}
    by_chrom = defaultdict(list)
    for chrom in sorted(merged):
        for start, end in merged[chrom]:
            annotation = _annotate_anchor_axis(
                genes, chrom, start, end, pdis)
            node = "%s:%s-%s|%s" % (
                chrom, start, end, annotation["type"])
            record = {
                "chrom": chrom,
                "start": start,
                "end": end,
                "type": annotation["type"],
                "nearestDistanceToTSS": annotation["distance"],
                "nearestGene": ",".join(
                    gene.name for gene in annotation["genes"]),
                "nearestGeneLoc": annotation["gene_locations"],
            }
            anchors[node] = record
            by_chrom[chrom].append((node, record))

    graph = nx.Graph()
    graph.add_nodes_from(sorted(anchors))
    loop_annotations = {}
    edge_loops = defaultdict(list)
    edge_categories = defaultdict(set)
    for loop in loop_list:
        source = _anchor_node_for_interval(
            by_chrom, loop.chromX, loop.x_start, loop.x_end)
        target = _anchor_node_for_interval(
            by_chrom, loop.chromY, loop.y_start, loop.y_end)
        category = "cis" if loop.cis else "trans"
        relationship = "-".join(sorted((anchors[source]["type"],
                                          anchors[target]["type"])))
        edge_type = "%s:%s" % (category, relationship)
        loop_annotations[loop.id] = {
            "mergedAnchorA": source,
            "mergedAnchorB": target,
            "edgeType": edge_type,
        }
        if source == target:
            continue
        edge = tuple(sorted((source, target)))
        edge_loops[edge].append(str(loop.id))
        edge_categories[edge].add(edge_type)
        graph.add_edge(*edge)

    targets = {}
    for promoter in sorted(graph.nodes):
        if anchors[promoter]["type"] != "Promoter":
            continue
        direct_enhancer, indirect_enhancer = set(), set()
        direct_promoter, indirect_promoter = set(), set()
        paths = nx.single_source_shortest_path(graph, promoter)
        for node, path in sorted(paths.items()):
            if node == promoter:
                continue
            direct = len(path) == 2
            if anchors[node]["type"] == "Promoter":
                (direct_promoter if direct else indirect_promoter).add(node)
            elif anchors[node]["type"] == "Enhancer":
                (direct_enhancer if direct else indirect_enhancer).add(node)
        targets[promoter] = {
            "targetGene": anchors[promoter]["nearestGeneLoc"],
            "directEnhancer": direct_enhancer,
            "indirectEnhancer": indirect_enhancer,
            "directPromoter": direct_promoter,
            "indirectPromoter": indirect_promoter,
            "directEnhancerHub": _stable_hub(graph, direct_enhancer),
            "indirectEnhancerHub": _stable_hub(graph, indirect_enhancer),
        }

    anchor_columns = [
        "chrom", "start", "end", "type", "nearestDistanceToTSS",
        "nearestGene", "nearestGeneLoc",
    ]
    anchor_frame = pd.DataFrame.from_dict(
        anchors, orient="index", columns=anchor_columns)
    anchor_frame.to_csv(fout + "_mergedAnchors.txt", sep="\t",
                        index_label="anchor")
    with open(fout + "_mergedAnchors.bed", "w") as handle:
        for node in sorted(anchors):
            record = anchors[node]
            handle.write("%s\t%s\t%s\t%s\n" % (
                record["chrom"], record["start"], record["end"], node))

    annotation_columns = ["mergedAnchorA", "mergedAnchorB", "edgeType"]
    pd.DataFrame.from_dict(
        loop_annotations, orient="index", columns=annotation_columns).to_csv(
            fout + "_loop2anchors.txt", sep="\t", index_label="loopId")

    with open(fout + "_ep_net.sif", "w") as sif, open(
            fout + "_ep_net_edges.txt", "w") as edge_output:
        edge_output.write("sourceAnchor\tedgeType\ttargetAnchor\tloopIds\n")
        for source, target in sorted(edge_loops):
            edge_type = ",".join(sorted(edge_categories[(source, target)]))
            loop_ids = ",".join(sorted(edge_loops[(source, target)]))
            sif.write("%s\t%s\t%s\n" % (source, edge_type, target))
            edge_output.write("%s\t%s\t%s\t%s\n" %
                              (source, edge_type, target, loop_ids))

    with open(fout + "_targets.txt", "w") as handle:
        columns = [
            "Promoter", "PromoterTarget", "directEnhancer",
            "indirectEnhancer", "directPromoter", "indirectPromoter",
            "directEnhancerHub", "indirectEnhancerHub",
        ]
        handle.write("\t".join(columns) + "\n")
        for promoter in sorted(targets):
            target = targets[promoter]
            handle.write("\t".join([
                promoter, target["targetGene"],
                ",".join(sorted(target["directEnhancer"])),
                ",".join(sorted(target["indirectEnhancer"])),
                ",".join(sorted(target["directPromoter"])),
                ",".join(sorted(target["indirectPromoter"])),
                target["directEnhancerHub"], target["indirectEnhancerHub"],
            ]) + "\n")
    return anchors, loop_annotations, graph, targets

### annotate loops
def anaLoops(loopf,
             fout,
             gtf=None,
             tid=False,
             pdis=2000,
             net=False,
             gap=1,
             cpu=1,
             mode="cis"):
    """
    Analyze loops.
    @param loopf: str, name of loops file,  _loops.txt or _dloops.txt file
    @param fout: str, output prefix
    @param gtf: str, GTF file name 
    @param tid: bool, if set true, use transcript id for alternative TSS
    @param pdis: <=distance nearest TSS to define as promoter
    @param net: bool, whether use network search for all linked anchors/enhancers/promoters for target gene
    @param gap: int, gap for merge anchors
    @param cpu: int, number of CPU to run analysis
    """
    if mode not in ("cis", "trans", "all"):
        raise ValueError("mode must be cis, trans, or all")
    loops = parseTxt2Loops(loopf, cut=0, mode=mode)
    if mode == "cis":
        # Preserve the historical cis-only structure and implementation.
        nloops = {}
        for key in loops.keys():
            nk = key.split("-")
            if nk[0] != nk[1]:
                continue
            nloops[nk[0]] = loops[key]
        loops = nloops
    if gtf is not None and gtf != "":
        if not os.path.isfile(gtf):
            print("Input %s not exists, continue to other analysis." % gtf)
        else:
            #gene annotions, {chrom:{tss:g}}, tss is int
            genes = readGenes(gtf, tid=tid)
            if mode == "cis":
                annotateLoopToGenes(loops, genes, fout, pdis=pdis, cpu=cpu)
            else:
                annotateLoopToGenesAxisAware(
                    loops, genes, fout, pdis=pdis, cpu=cpu)
            #get common summary of interaction type summary and distance summary
            if net:
                if mode == "cis":
                    getNetworksFromLoops(loops,
                                         genes,
                                         fout,
                                         pdis=pdis,
                                         gap=gap,
                                         cpu=cpu)
                else:
                    getNetworksFromLoopsAxisAware(
                        loops, genes, fout, pdis=pdis, gap=gap, cpu=cpu)
