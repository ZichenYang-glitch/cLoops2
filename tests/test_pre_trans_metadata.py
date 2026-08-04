import json
import logging
import os
import tempfile
import unittest

from cLoops2.io import parseBedpe


class PreTransMetadataTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.bedpe = os.path.join(self.tmpdir.name, "input.bedpe")
        lines = []
        for index in range(4):
            start_a = index * 100
            start_b = start_a + 1000
            lines.append(self._bedpe_line("chr1", start_a, "chr1", start_b,
                                           "cis-%s" % index))
        # Duplicate one retained cis PET: five input cis records, four unique.
        lines.append(lines[0].replace("cis-0", "cis-duplicate"))
        for index in range(6):
            lines.append(self._bedpe_line("chr1", index * 100, "chr2",
                                           index * 200, "trans-%s" % index))
        with open(self.bedpe, "w") as handle:
            handle.writelines(lines)
        self.logger = logging.getLogger("pre-trans-metadata-test")
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        self.tmpdir.cleanup()

    @staticmethod
    def _bedpe_line(chrom_a, start_a, chrom_b, start_b, name):
        return "%s\t%s\t%s\t%s\t%s\t%s\t%s\t60\t+\t-\n" % (
            chrom_a, start_a, start_a + 10, chrom_b, start_b, start_b + 10,
            name)

    def _run_pre(self, retain_trans):
        outdir = os.path.join(self.tmpdir.name,
                              "with-trans" if retain_trans else "cis-only")
        os.mkdir(outdir)
        parseBedpe([self.bedpe],
                   outdir,
                   self.logger,
                   mapq=1,
                   cis=not retain_trans,
                   cpu=1)
        with open(os.path.join(outdir, "petMeta.json")) as handle:
            return json.load(handle)

    def test_retained_trans_does_not_corrupt_cis_redundancy(self):
        meta = self._run_pre(retain_trans=True)
        self.assertEqual(meta["Total Cis PETs"], 5)
        self.assertEqual(meta["Total Trans PETs"], 6)
        self.assertEqual(meta["Unique Cis PETs"], 4)
        self.assertEqual(meta["Unique Trans PETs"], 6)
        self.assertEqual(meta["Unique PETs"], 10)
        self.assertAlmostEqual(meta["Cis PETs Redundancy"], 0.2)
        self.assertTrue(meta["Retention"]["retain trans"])
        self.assertEqual(meta["Retention"]["retained categories"],
                         ["cis", "trans"])
        self.assertTrue(meta["data"]["trans"])

    def test_default_pre_records_trans_but_does_not_retain_it(self):
        meta = self._run_pre(retain_trans=False)
        self.assertEqual(meta["Total Trans PETs"], 6)
        self.assertEqual(meta["Unique PETs"], 4)
        self.assertEqual(meta["Unique Cis PETs"], 4)
        self.assertEqual(meta["Unique Trans PETs"], 0)
        self.assertAlmostEqual(meta["Cis PETs Redundancy"], 0.2)
        self.assertFalse(meta["Retention"]["retain trans"])
        self.assertEqual(meta["Retention"]["retained categories"], ["cis"])
        self.assertFalse(meta["data"]["trans"])


if __name__ == "__main__":
    unittest.main()
