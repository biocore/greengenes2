import unittest
import skbio
import pandas as pd
from gg2.backbone_taxonomy import (graft_from_other, carryover, LEVELS,
                                   parse_lineage,
                                   remove_unmappable_polyphyletic)
import pandas.testing as pdt


class BackboneTaxonomyTests(unittest.TestCase):
    def test_remove_unmappable_polyphyletic(self):
        gtdb_data = pd.DataFrame([["G000425525", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__Ruminococcus_G; s__Ruminococcus_G gauvreauii"],
                                  ["G000526735", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__Ruminococcus_B; s__Ruminococcus_B gnavus"],
                                  ["G001917065", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__Ruminococcus_E; s__Ruminococcus_E bromii_B"],
                                  ["G003435865", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__Ruminococcus_E; s__Ruminococcus_E bromii_B"],
                                  ["G900101355", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__Ruminococcus_E; s__Ruminococcus_E bromii_A"],
                                  ["G900318235", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__Ruminococcus_E; s__Ruminococcus_E sp900317315"],
                                  ["G000518765", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Ruminococcaceae; g__Ruminococcus; s__Ruminococcus flavefaciens"],
                                  ["G000701945", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Ruminococcaceae; g__Ruminococcus; s__Ruminococcus flavefaciens_B"],
                                  ["G900119535", "d__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Ruminococcaceae; g__Ruminococcus; s__Ruminococcus flavefaciens_H"]],
                                 columns=['id', 'lineage'])
        ltp_data = pd.DataFrame([["AM915269", "Bacteria; Firmicutes_A; Clostridia; Oscillospirales; Ruminococcaceae; Ruminococcus; Ruminococcus flavefaciens"],
                                 ["X94967", "Bacteria; Firmicutes_A; Clostridia; Oscillospirales; Ruminococcaceae; Ruminococcus; Ruminococcus gnavus"],
                                 ["EF529620", "Bacteria; Firmicutes_A; Clostridia; Oscillospirales; Ruminococcaceae; Ruminococcus; Ruminococcus gauvreauii"],
                                 ["L76600", "Bacteria; Firmicutes_A; Clostridia; Oscillospirales; Ruminococcaceae; Ruminococcus; Ruminococcus bromii"],
                                 ["L76596", "Bacteria; Firmicutes_A; Clostridia; Oscillospirales; Ruminococcaceae; Ruminococcus; Ruminococcus callidus"]],
                                columns=['id', 'lineage'])
        parse_lineage(gtdb_data)
        parse_lineage(ltp_data)

        exp = ltp_data.copy()
        obs = remove_unmappable_polyphyletic(gtdb_data, ltp_data)
        obs.index = list(range(len(obs)))
        pdt.assert_frame_equal(obs, exp)

    def test_graft_from_other(self):
        other = skbio.TreeNode.read(["((a,b)c,(d,e)f);"])
        other.find('c').Rank = 5
        other.find('f').Rank = 4
        other.find('a').keepable = True
        other.find('b').keepable = True
        other.find('d').keepable = True
        other.find('e').keepable = True
        lookup = {'c': skbio.TreeNode(name='lookup-c'),
                  'f': skbio.TreeNode(name='not-used')}
        graft_from_other(other, lookup)
        self.assertEqual(['a', 'b'],
                         sorted([n.name for n in lookup['c'].children]))
        self.assertEqual(['d', 'e'], [n.name for n in lookup['f'].children])

    def test_carryover(self):
        tree = skbio.TreeNode.read(["((a,b)p__c,(d,f)c__g);"])
        tax = pd.DataFrame([['t1', 'x', 'c', 'x', 'x', 'x', 'x', 'x'],
                            ['t2', 'x', 'c', 'x', 'x', 'x', 'x', 'x'],
                            ['t3', 'x', 'c', 'g', 'x', 'x', 'x', 'x'],
                            ['t4', 'x', 'x', 'g', 'x', 'x', 'x', 'x'],
                            ['t5', 'x', 'x', 'y', 'x', 'x', 'x', 'x'],
                            ['t6', 'x', 'x', 'y', 'x', 'x', 'x', 'x']],
                           columns=['id'] + LEVELS)
        lookup = {'p__c': skbio.TreeNode(name='lookup-c'),
                  'c__g': skbio.TreeNode(name='lookup-g')}
        carryover(tax, tree, lookup)
        self.assertEqual(lookup['p__c'].children[0].name, 'c__')
        self.assertEqual(lookup['p__c'].children[0].children[0].name, 'o__')
        self.assertEqual(lookup['p__c'].children[0].children[0].children[0].name, 'f__')
        self.assertEqual(lookup['p__c'].children[0].children[0].children[0].children[0].name, 'g__')
        self.assertEqual(lookup['p__c'].children[0].children[0].children[0].children[0].children[0].name, 's__')
        s = lookup['p__c'].children[0].children[0].children[0].children[0].children[0]
        self.assertEqual(['t1', 't2'], sorted([n.name for n in s.children]))
        s = lookup['c__g'].children[0].children[0].children[0].children[0]
        self.assertEqual(['t3', 't4'], sorted([n.name for n in s.children]))


if __name__ == '__main__':
    unittest.main()
