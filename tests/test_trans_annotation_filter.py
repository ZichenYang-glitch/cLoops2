import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cLoops2.ano import anaLoops, readGenes
from cLoops2.filter import filterPETsByLoops, filterPETsByPeaks, samplePETs
from cLoops2.findTargets import findTargets, readNet
from cLoops2.metadata import (CAP_FORMAL_INFERENCE,
                              build_library_context,
                              inspect_actual_counts)


class TransAnnotationNetworkTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cloops2-trans-annotation-")
        self.root = Path(self.temporary.name)
        self.gtf = self.root / "genes.gtf"
        self.gtf.write_text(
            'chr1\tfixture\texon\t100\t120\t.\t+\t.\t'
            'gene_id "gene-a"; gene_name "GeneA";\n'
            'chr2\tfixture\texon\t500\t520\t.\t+\t.\t'
            'gene_id "gene-b"; gene_name "GeneB";\n')
        self.trans_loops = self.root / "trans_loops.txt"
        self.trans_loops.write_text(
            "loopId\tchromX\tx_start\tx_end\tchromY\ty_start\ty_end\tdistance\n"
            "trans-1\tchr1\t300\t310\tchr2\t495\t505\t-1\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_trans_annotation_uses_chromosome_specific_gene_indexes(self):
        prefix = str(self.root / "annotation")
        anaLoops(self.trans_loops, prefix, gtf=str(self.gtf), pdis=20,
                 mode="trans")
        table = pd.read_csv(prefix + "_LoopsGtfAno.txt", sep="\t",
                            index_col=0)
        row = table.loc["trans-1"]
        self.assertEqual(row["chromAnchorA"], "chr1")
        self.assertEqual(row["chromAnchorB"], "chr2")
        self.assertEqual(row["loopCategory"], "trans")
        self.assertIn("GeneA", row["nearestTargetGeneAnchorA"])
        self.assertNotIn("GeneB", row["nearestTargetGeneAnchorA"])
        self.assertIn("GeneB", row["nearestTargetGeneAnchorB"])
        self.assertEqual(row["typeAnchorA"], "Enhancer")
        self.assertEqual(row["typeAnchorB"], "Promoter")

    def test_same_gene_name_on_two_chromosomes_is_not_merged(self):
        duplicated = self.root / "duplicate-name.gtf"
        duplicated.write_text(
            'chr1\tfixture\texon\t10\t20\t.\t+\t.\t'
            'gene_id "one"; gene_name "Shared";\n'
            'chr2\tfixture\texon\t100\t120\t.\t-\t.\t'
            'gene_id "two"; gene_name "Shared";\n')
        genes = readGenes(str(duplicated))
        self.assertEqual(set(genes), {"chr1", "chr2"})
        self.assertEqual(genes["chr1"][10].chrom, "chr1")
        self.assertEqual(genes["chr2"][120].chrom, "chr2")

    def test_trans_edge_and_findtargets_cross_chromosome(self):
        prefix = str(self.root / "network")
        anaLoops(self.trans_loops, prefix, gtf=str(self.gtf), pdis=20,
                 net=True, mode="trans")
        graph, _ = readNet(prefix + "_ep_net.sif")
        self.assertEqual(graph.number_of_edges(), 1)
        source, target = next(iter(graph.edges()))
        self.assertEqual({source.split(":", 1)[0], target.split(":", 1)[0]},
                         {"chr1", "chr2"})
        edge_type = graph.edges[source, target]["type"]
        self.assertTrue(edge_type.startswith("trans:"), edge_type)

        # Three-column BED used to trigger an IndexError in the legacy parser.
        query = self.root / "query.bed"
        query.write_text("chr1\t300\t310\n")
        found = str(self.root / "found")
        findTargets(prefix + "_ep_net.sif", prefix + "_targets.txt",
                    found, fbed=str(query))
        targets = pd.read_csv(found + "_targetGenes.txt", sep="\t")
        self.assertEqual(targets.shape[0], 1)
        self.assertIn("GeneB", targets.loc[0, "directTargetGenes"])

    def test_default_annotation_remains_cis_only(self):
        mixed = self.root / "mixed.txt"
        mixed.write_text(
            "loopId\tchromX\tx_start\tx_end\tchromY\ty_start\ty_end\tdistance\n"
            "cis-1\tchr1\t95\t105\tchr1\t150\t160\t55\n"
            "trans-1\tchr1\t300\t310\tchr2\t495\t505\t-1\n")
        prefix = str(self.root / "legacy-cis")
        anaLoops(mixed, prefix, gtf=str(self.gtf), pdis=20)
        table = pd.read_csv(prefix + "_LoopsGtfAno.txt", sep="\t",
                            index_col=0)
        self.assertEqual(list(table.index), ["cis-1"])

        all_prefix = str(self.root / "all-loops")
        anaLoops(mixed, all_prefix, gtf=str(self.gtf), pdis=20, mode="all")
        all_table = pd.read_csv(all_prefix + "_LoopsGtfAno.txt", sep="\t",
                                index_col=0)
        self.assertEqual(set(all_table.index), {"cis-1", "trans-1"})


class AxisAwareFilterTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cloops2-trans-filter-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.matrix = np.asarray([
            [100, 1000],
            [1000, 100],
            [100, 100],
        ], dtype=np.int64)
        path = self.source / "chr1-chr2.ixy"
        joblib.dump(self.matrix, str(path))
        meta = {
            "Unique PETs": 3,
            "Unique Cis PETs": 0,
            "Unique Trans PETs": 3,
            "Total PETs": 3,
            "Total Cis PETs": 0,
            "Total Trans PETs": 3,
            "Retention": {
                "retain trans": True,
                "retained categories": ["cis", "trans"],
            },
            "data": {
                "cis": {},
                "trans": {
                    "chr1-chr2": {
                        "ixy": str(path.resolve()),
                        "chromX": "chr1",
                        "chromY": "chr2",
                        "record_id": "trans:chr1-chr2",
                    },
                },
            },
        }
        (self.source / "petMeta.json").write_text(json.dumps(meta))
        self.loop_file = self.root / "loop.txt"
        self.loop_file.write_text(
            "loopId\tchromX\tx_start\tx_end\tchromY\ty_start\ty_end\tdistance\n"
            "strict\tchr1\t90\t110\tchr2\t90\t110\t-1\n")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _meta(directory):
        return json.loads((directory / "petMeta.json").read_text())

    def _add_cis_record(self):
        cis = np.asarray([[100, 200], [500, 600]], dtype=np.int64)
        path = self.source / "chr1-chr1.ixy"
        joblib.dump(cis, str(path))
        meta = self._meta(self.source)
        meta["Unique PETs"] = 5
        meta["Unique Cis PETs"] = 2
        meta["Total PETs"] = 5
        meta["Total Cis PETs"] = 2
        meta["data"]["cis"]["chr1-chr1"] = {
            "ixy": str(path.resolve()),
            "chromX": "chr1",
            "chromY": "chr1",
            "record_id": "cis:chr1-chr1",
        }
        (self.source / "petMeta.json").write_text(json.dumps(meta))
        return cis

    def _add_unreferenced_trans_record(self):
        matrix = np.asarray([[7, 70], [8, 80]], dtype=np.int64)
        path = self.source / "chr1-chr3.ixy"
        joblib.dump(matrix, str(path))
        meta = self._meta(self.source)
        meta["Unique PETs"] += 2
        meta["Unique Trans PETs"] += 2
        meta["Total PETs"] += 2
        meta["Total Trans PETs"] += 2
        meta["data"]["trans"]["chr1-chr3"] = {
            "ixy": str(path.resolve()),
            "chromX": "chr1",
            "chromY": "chr3",
            "record_id": "trans:chr1-chr3",
        }
        (self.source / "petMeta.json").write_text(json.dumps(meta))
        return matrix

    def test_both_uses_strict_x_by_y_rectangle(self):
        output = self.root / "strict"
        output.mkdir()
        transform = filterPETsByLoops(
            str(self.source), str(self.loop_file), str(output), both=True,
            mode="trans")
        observed = joblib.load(str(output / "chr1-chr2.ixy"))
        np.testing.assert_array_equal(
            observed, np.asarray([[100, 100]], dtype=np.int64))
        self.assertEqual(transform["operation"], "filter")
        self.assertEqual(transform["parameters"]["mode"], "trans")
        self.assertFalse(transform["formal_inference_eligible"])

        meta = self._meta(output)
        context = build_library_context(
            meta, actual_counts=inspect_actual_counts(meta))
        self.assertEqual(context.validity, "valid", context.reasons)
        self.assertFalse(context.replacement_ever)
        self.assertNotIn(CAP_FORMAL_INFERENCE, context.capabilities)

        nested = self.root / "strict-sampled"
        child = samplePETs(str(output), str(nested), 1,
                           mode="trans", seed=23)
        self.assertFalse(child["replacement_ever"])
        self.assertFalse(child["formal_inference_eligible"])

    def test_any_end_is_axis_specific_and_reverse_loop_is_canonicalized(self):
        output = self.root / "any"
        output.mkdir()
        filterPETsByLoops(
            str(self.source), str(self.loop_file), str(output), both=False,
            mode="trans")
        np.testing.assert_array_equal(
            joblib.load(str(output / "chr1-chr2.ixy")), self.matrix)

        reverse = self.root / "reverse.txt"
        reverse.write_text(
            "loopId\tchromX\tx_start\tx_end\tchromY\ty_start\ty_end\tdistance\n"
            "reverse\tchr2\t90\t110\tchr1\t90\t110\t-1\n")
        reversed_output = self.root / "reverse"
        reversed_output.mkdir()
        filterPETsByLoops(
            str(self.source), str(reverse), str(reversed_output), both=True,
            mode="trans")
        np.testing.assert_array_equal(
            joblib.load(str(reversed_output / "chr1-chr2.ixy")),
            np.asarray([[100, 100]], dtype=np.int64))

    def test_filter_preserves_replacement_ancestry_without_laundering(self):
        upsampled = self.root / "upsampled"
        samplePETs(str(self.source), str(upsampled), 6,
                   mode="all", seed=17)
        output = self.root / "filtered-upsample"
        output.mkdir()
        transform = filterPETsByLoops(
            str(upsampled), str(self.loop_file), str(output), both=False,
            mode="trans")
        self.assertTrue(transform["replacement_ever"])
        meta = self._meta(output)
        context = build_library_context(
            meta, actual_counts=inspect_actual_counts(meta))
        self.assertTrue(context.replacement_ever)
        self.assertNotIn(CAP_FORMAL_INFERENCE, context.capabilities)

        nested = self.root / "filtered-sampled"
        child = samplePETs(str(output), str(nested), 1,
                           mode="trans", seed=19)
        self.assertTrue(child["replacement_ever"])
        self.assertFalse(child["formal_inference_eligible"])

    def test_invert_keeps_nonmatching_rows_and_unreferenced_pairs(self):
        unreferenced = self._add_unreferenced_trans_record()
        output = self.root / "invert"
        output.mkdir()
        filterPETsByLoops(
            str(self.source), str(self.loop_file), str(output), both=True,
            iv=True, mode="trans")
        np.testing.assert_array_equal(
            joblib.load(str(output / "chr1-chr2.ixy")), self.matrix[:2])
        np.testing.assert_array_equal(
            joblib.load(str(output / "chr1-chr3.ixy")), unreferenced)
        self.assertEqual(self._meta(output)["Unique PETs"], 4)

    def test_peak_regions_query_each_trans_axis_on_its_own_chromosome(self):
        peaks = self.root / "peaks.bed"
        peaks.write_text("chr1\t90\t110\tx-peak\n")
        output = self.root / "peak-filter"
        output.mkdir()
        transform = filterPETsByPeaks(
            str(self.source), str(peaks), str(output), mode="trans")
        np.testing.assert_array_equal(
            joblib.load(str(output / "chr1-chr2.ixy")),
            np.asarray([[100, 1000], [100, 100]], dtype=np.int64))
        self.assertEqual(transform["parameters"]["mode"], "trans")

    def test_default_mode_keeps_historical_cis_routing(self):
        self._add_cis_record()
        mixed = self.root / "mixed.txt"
        mixed.write_text(
            "loopId\tchromX\tx_start\tx_end\tchromY\ty_start\ty_end\tdistance\n"
            "cis\tchr1\t90\t110\tchr1\t190\t210\t100\n"
            "trans\tchr1\t90\t110\tchr2\t90\t110\t-1\n")
        output = self.root / "default-cis"
        output.mkdir()
        transform = filterPETsByLoops(
            str(self.source), str(mixed), str(output), both=True)
        np.testing.assert_array_equal(
            joblib.load(str(output / "chr1-chr1.ixy")),
            np.asarray([[100, 200]], dtype=np.int64))
        self.assertFalse((output / "chr1-chr2.ixy").exists())
        self.assertEqual(transform["parameters"]["mode"], "cis")

        all_output = self.root / "all-filter"
        all_output.mkdir()
        filterPETsByLoops(
            str(self.source), str(mixed), str(all_output), both=True,
            mode="all")
        self.assertTrue((all_output / "chr1-chr1.ixy").is_file())
        self.assertTrue((all_output / "chr1-chr2.ixy").is_file())


if __name__ == "__main__":
    unittest.main()
