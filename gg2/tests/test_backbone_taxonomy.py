import unittest
import skbio
import pandas as pd
from gg2.backbone_taxonomy import (graft_from_other, LEVELS,
                                   parse_lineage, get_polyphyletic,
                                   ids_of_ambiguity_from_polyphyletic,
                                   POLY_SPECIES, POLY_GENERAL)
import pandas.testing as pdt


class BackboneTaxonomyTests(unittest.TestCase):
    def test_poly_species(self):
        tests = [('s__foo bar', ('s__foo', None, 'bar', None)),
                 ('s__foo_A bar', ('s__foo', '_A', 'bar', None)),
                 ('s__foo bar_A', ('s__foo', None, 'bar', '_A')),
                 ('s__foo_A bar_B', ('s__foo', '_A', 'bar', '_B')),
                 ('s__foo-A bar_B', ('s__foo-A', None, 'bar', '_B')),
                 ('s__foo_A bar-B', ('s__foo', '_A', 'bar-B', None)),
                 ('s__foo_ABC bar_XYZ', ('s__foo', '_ABC', 'bar', '_XYZ'))]

        for in_, exp in tests:
            match = POLY_SPECIES.match(in_)
            self.assertTrue(match is not None)
            self.assertEqual(match.groups(), exp)

        tests = ['g__foo',
                 's__foo bar baz',
                 's__foo',
                 's__']
        for in_ in tests:
            match = POLY_SPECIES.match(in_)
            self.assertEqual(match, None)

    def test_poly_general(self):
        tests = [('d__foo', ('d__foo', None)),
                 ('p__foo_A', ('p__foo', '_A')),
                 ('p__foo-A', ('p__foo-A', None)),
                 ('f__foo_ABC', ('f__foo', '_ABC'))]

        for in_, exp in tests:
            match = POLY_GENERAL.match(in_)
            self.assertTrue(match is not None)
            self.assertEqual(match.groups(), exp)

        tests = ['p__foo bar',
                 'p__foo bar baz',
                 's__foo bar',
                 'p__']
        for in_ in tests:
            match = POLY_GENERAL.match(in_)
            self.assertEqual(match, None)

    def test_get_polyphyletic(self):
        df = pd.DataFrame([
            ["x1", "d__X; p__X; c__X; o__X; f__X; g__X; s__X X"],
            ["x2", "d__X; p__X_A; c__X; o__X; f__X; g__X; s__X X"],
            ["x3", "d__X; p__Y; c__X; o__X; f__X; g__X; s__X X"],
            ["x4", "d__X; p__Y; c__X; o__X; f__X; g__X; s__X X"],
            ["x5", "d__X; p__Z_B; c__X; o__X; f__X; g__X; s__X X"],
            ["x6", "d__X; p__Y; c__X; o__X; f__X; g__X_A; s__X_A X"],
            ["x7", "d__X; p__Y; c__X; o__X; f__X; g__X; s__Y X"],
            ["x8", "d__X; p__Y; c__X; o__X; f__X; g__X; s__Y X_A"],
            ["x9", "d__X; p__Y; c__X; o__X; f__X; g__X; s__Z X"]],
            columns=['id', 'lineage'])
        parse_lineage(df)
        exp = {'domain': set(),
               'phylum': {'X', 'Z'},
               'class': set(),
               'order': set(),
               'family': set(),
               'genus': {'X', },
               'species': {'X X', 'Y X'}}
        obs = get_polyphyletic(df)
        self.assertEqual(obs, exp)

    def test_ids_of_ambiguity_from_polyphyletic(self):
        gtdb = pd.DataFrame([
            ["x1", "d__Xa; p__Xb; c__Xc; o__Xd; f__Xe; g__Xf; s__Xf Xg"],
            ["x2", "d__Xa; p__Xb; c__Xc; o__Xd; f__Xe; g__Xf; s__Xf Xh"],
            ["x3", "d__Xa; p__Xb; c__Xc; o__Xd; f__Xe; g__Xi; s__Xi Xj"],
            ["x4", "d__Xa; p__Xb; c__Xc; o__Xd; f__Xe; g__Xi_A; s__Xi_A Xj"],
            ["x5", "d__Xa; p__Xb; c__Xc; o__Xd; f__Xe; g__Xk; s__Xk Xl_A"],
            ["x6", "d__Xa; p__Xo_A; c__Xp; o__Xq; f__Xr; g__Xm; s__Xm Xn"],
            ["x7", "d__Xa; p__Xo_A; c__Xp; o__Xq; f__Xr; g__Xm; s__Xm Xz_B"]],
            columns=['id', 'lineage'])
        parse_lineage(gtdb)
        poly = get_polyphyletic(gtdb)
        poly_exp = {'domain': set(),
                    'phylum': {'Xo', },
                    'class': set(),
                    'order': set(),
                    'family': set(),
                    'genus': {'Xi', },
                    'species': {'Xi Xj', 'Xk Xl', 'Xm Xz'}}
        self.assertEqual(poly, poly_exp)

        ltp = pd.DataFrame([
            ["y1", "Xa;Xb;Xc;Xd;Xe;Xf;Xf Xg", False],
            ["y2", "Xa;Xb;Xc;Xd;Xe;Xf;Xf Xh", False],
            ["y3", "Xa;Xb;Xc;Xd;Xe;Xf;Xi Xj", False],
            ["y4", "Xa;Xb;Xc;Xd;Xe;Xf;Xk Xl", False],
            ["y5", "Xa;Xo;Xp;Xq;Xr;Xm;Xm Xn", False],
            ["y6", "Xa;Xo;Xp;Xq;Xr;Xm;Xm Xz", False],
            ["y7", "Xb;X1;X2;X3;X4;X5;X6 X7", False],
            ["y8", "Xb;X1;X2;X3;X4;X5;X6 X8", False]],
            columns=['id', 'lineage', 'explicitly_set'])
        parse_lineage(ltp)
        obs = ids_of_ambiguity_from_polyphyletic(ltp, poly)
        exp = {'y3', 'y4', 'y5', 'y6'}
        self.assertEqual(obs, exp)

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


if __name__ == '__main__':
    unittest.main()
