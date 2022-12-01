import unittest
import pandas.testing as pdt
from gg2.tree_map import (cut, chop_to_species, md_from_tree, species_cover,
                          to_species_as_tips, uniqify, genome_represented,
                          cut_uncovered_species, collapse_species_by_distance)
import pandas as pd
import numpy as np
import skbio


class Tests(unittest.TestCase):
    def test_cut(self):
        t = skbio.TreeNode.read(["((a,b)c,(d,e)f,(h:5)i)g;"])
        cut(t.find('c'))
        self.assertTrue(t.find('c').is_tip())
        self.assertEqual(sorted([n.name for n in t.tips()]),
                         ['c', 'd', 'e', 'h'])
        cut(t.find('i'))
        self.assertTrue(np.isclose(t.find('i').length, 5))

    def test_chop_to_species(self):
        t = skbio.TreeNode.read(["((((a,b)s__foo,(c,d)g__bar,(e,g)),(h,i)'g__bar; s__baz'),(j:5)'g__bar; s__cool');"],  # noqa
                                convert_underscores=False)
        chop_to_species(t)
        self.assertTrue(t.find('s__foo').is_tip())
        self.assertEqual(sorted([n.name for n in t.tips()]),
                         ['c', 'd', 'e', 'g',
                          'g__bar; s__baz',
                          'g__bar; s__cool',
                          's__foo'])
        self.assertFalse(t.find('g__bar').is_tip())
        self.assertTrue(np.isclose(t.find('g__bar; s__cool').length, 5))

        t = skbio.TreeNode.read(["(((a,b)s__foo,(c,d)g__bar,(e,g)),(h,i)'g__bar2; s__baz');"],
                                convert_underscores=False)
        chop_to_species(t, 'g')
        self.assertTrue(t.find('g__bar').is_tip())
        self.assertTrue(t.find('g__bar2').is_tip())
        self.assertEqual(sorted([n.name for n in t.tips()]),
                         ['a', 'b', 'e', 'g', 'g__bar', 'g__bar2'])

    def test_species_cover(self):
        t = skbio.TreeNode.read(["(s__foo,(c,d)g__bar,(e,g));"],
                                convert_underscores=False)
        species_cover(t)
        self.assertTrue(t.species_cover)
        self.assertFalse(t.find('c').species_cover)
        self.assertFalse(t.find('d').species_cover)
        self.assertFalse(t.find('e').species_cover)
        self.assertFalse(t.find('g').species_cover)
        self.assertFalse(t.find('g').parent.species_cover)
        self.assertFalse(t.find('g__bar').species_cover)
        self.assertTrue(t.find('s__foo').species_cover)

    def test_cut_uncovered_species(self):
        t = skbio.TreeNode.read(["(s__foo,(c,d)g__bar,(e,g));"],
                                convert_underscores=False)
        species_cover(t)
        cut_uncovered_species(t, 's__')
        self.assertTrue(t.find('s__foo').is_tip())
        self.assertEqual(len(t.children), 1)
        self.assertEqual(t.children[0].name, 's__foo')

    def test_to_species_as_tips(self):
        t = skbio.TreeNode.read(["((a,b)s__foo,(c,d)g__bar,(e,g));"],
                                convert_underscores=False)
        to_species_as_tips(t, 's')
        self.assertTrue(t.find('s__foo').is_tip())
        self.assertEqual(len(t.children), 1)
        self.assertEqual(t.children[0].name, 's__foo')

    def test_genome_represented(self):
        t = skbio.TreeNode.read(["((((G000000001, b)'c__X; o__Y'),(c,d)'c__X; o__X')'d__X; p__X')root;"],
                                convert_underscores=False)
        genome_represented(t)
        self.assertTrue(t.genome_cover)
        self.assertTrue(t.find('G000000001').genome_cover)
        self.assertTrue(t.find('c__X; o__Y').genome_cover)
        self.assertFalse(t.find('c__X; o__X').genome_cover)

    def test_md_from_tree(self):
        t = skbio.TreeNode.read(["((((s__foo,s__bar)'f__cool; g__baz'),(s__X)'f__X; g__X')'d__X; p__X; c__X; o__X')root;"],  # noqa
                                convert_underscores=False)
        exp = pd.DataFrame([['s__foo',
                             'd__X; p__X; c__X; o__X; f__cool; g__baz; s__foo', 'False',  # noqa
                             'd__X', 'p__X', 'c__X', 'o__X', 'f__cool', 'g__baz', 's__foo'],  # noqa
                            ['s__bar',
                             'd__X; p__X; c__X; o__X; f__cool; g__baz; s__bar', 'False',  # noqa
                             'd__X', 'p__X', 'c__X', 'o__X', 'f__cool', 'g__baz', 's__bar'],  # noqa
                            ['s__X',
                             'd__X; p__X; c__X; o__X; f__X; g__X; s__X', 'False',  # noqa
                             'd__X', 'p__X', 'c__X', 'o__X', 'f__X', 'g__X', 's__X']],  # noqa
                           columns=['Feature ID', 'lineage',
                                    'genome_represented',
                                    'domain', 'phylum', 'class', 'order',
                                    'family', 'genus',
                                    'species']).set_index('Feature ID')
        genome_represented(t)
        obs = md_from_tree(t)
        obs.drop(columns=[c for c in obs.columns if '_top10' in c],
                 inplace=True)
        pdt.assert_frame_equal(obs, exp, check_dtype=False)

        t = skbio.TreeNode.read(["((('o__X; f__cool'),'o__X; f__X')'d__X; p__X; c__X')root;"],  # noqa
                                convert_underscores=False)
        exp = pd.DataFrame([['o__X; f__cool',
                             'd__X; p__X; c__X; o__X; f__cool', 'False',
                             'd__X', 'p__X', 'c__X', 'o__X', 'f__cool'],
                            ['o__X; f__X',
                             'd__X; p__X; c__X; o__X; f__X', 'False',
                             'd__X', 'p__X', 'c__X', 'o__X', 'f__X']],
                           columns=['Feature ID', 'lineage', 'genome_represented',  # noqa
                                    'domain', 'phylum', 'class', 'order',
                                    'family']).set_index('Feature ID')
        genome_represented(t)
        obs = md_from_tree(t)
        obs.drop(columns=[c for c in obs.columns if '_top10' in c],
                 inplace=True)
        pdt.assert_frame_equal(obs, exp, check_dtype=False)

    def test_uniqify(self):
        t = skbio.TreeNode.read(["((('c__X; o__X'),'c__X; o__X')'d__X; p__X')root;"],  # noqa
                                convert_underscores=False)
        uniqify(t)
        self.assertTrue('c__X-0; o__X-0' in [n.name for n in t.tips()])
        self.assertTrue('c__X-1; o__X-1' in [n.name for n in t.tips()])

    def test_collapse_species_by_distance(self):
        t = skbio.TreeNode.read(["(((a:1,b:2)s__foo:3,(c:4)s__bar:5)g__baz:6)root;"],  # noqa
                                convert_underscores=False)
        collapse_species_by_distance(t, 's')
        self.assertTrue(np.isclose(t.find('s__foo').length, 4))
        self.assertTrue(np.isclose(t.find('s__bar').length, 9))


if __name__ == '__main__':
    unittest.main()
