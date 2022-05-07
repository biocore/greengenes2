import unittest
import skbio
import pandas as pd
from gg2.backbone_taxonomy import (has_polyphyletic, get_polyphyletic,
                                   graft_from_other, carryover, LEVELS)


class BackboneTaxonomyTests(unittest.TestCase):
    def test_has_polyphyletic(self):
        poly = {'foo', 'bar'}
        with_poly = skbio.TreeNode.read(["(((a)b)foo)r;"]).find('a')
        without_poly = skbio.TreeNode.read(["(((a)b)c)r;"]).find('a')
        self.assertTrue(has_polyphyletic(with_poly, poly))
        self.assertFalse(has_polyphyletic(without_poly, poly))

    def test_get_polyphyletic(self):
        df = pd.DataFrame(['p__foo', 'p__bar', 'p__bar_A', 'p__bar_B'],
                          columns=['phylum'])
        obs = get_polyphyletic(df, 'phylum')
        self.assertEqual(obs, {'p__bar', 'p__bar_A', 'p__bar_B'})

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
        polyphyletic_names = set()
        graft_from_other(other, lookup, polyphyletic_names)
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
        carryover(tax, tree, lookup, set())
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
