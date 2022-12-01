import unittest
import skbio
import pandas as pd
import numpy as np
import pandas.testing as pdt
import io
from gg2.species_report import (extract_species, parse_full_length, seq_id,
                                check_alignment, extract_label)

class Tests(unittest.TestCase):
    def test_extract_species(self):
        tests = [('s__foo bar', 's__foo bar'),
                 ('s__foo_A_1234 bar', 's__foo bar'),
                 ('s__foo bar_A_123', 's__foo bar'),
                 ('s__', 's__')]
        for test, exp in tests:
            obs = extract_species(test)
            self.assertEqual(obs, exp)

    def test_extract_label(self):
        tests = [('s__foo bar', 's__foo bar'),
                 ('s__foo_A_1234 bar', 's__foo bar'),
                 ('s__foo bar_A_123', 's__foo bar'),
                 ('s__', 's__'),
                 ('g__', 'g__'),
                 ('g__foo_A', 'g__foo'),
                 ('g__foo_B_123', 'g__foo')]
        for test, exp in tests:
            obs = extract_label(test)
            self.assertEqual(obs, exp)

    def test_parse_full_length(self):
        exp = {'a': skbio.DNA('ATGC'),
               'G012345678': skbio.DNA('TTGGCC'),
               'G011111111': skbio.DNA('TT')}
        obs = parse_full_length(DATA, {'a', 'G012345678', 'G011111111'})
        self.assertEqual([(str(a), str(b)) for a, b in obs.items()],
                         [(str(a), str(b)) for a, b in exp.items()])

    def test_seq_id(self):
        aln = skbio.TabularMSA([skbio.DNA('AATTGGCC'),
                                skbio.DNA('AAAAGGCC')])
        exp_id, exp_len = 75.0, 8
        obs_id, obs_len = seq_id(aln)
        self.assertTrue(np.isclose(obs_id, exp_id))
        self.assertEqual(obs_len, exp_len)

    def test_check_alignment(self):
        seqs = {'a': skbio.DNA('TTAATTGG'),
                'b': skbio.DNA('TTATAAGG'),
                'c': skbio.DNA('TTAATTCG'),
                'MJ012-foobar': skbio.DNA('AATTGG')}
        exp = pd.DataFrame([['a', 'b', 75.0, 8],
                            ['a', 'c', 87.5, 8]],
                           columns=['Subject', 'Query', 'Sequence ID',
                                    'Length'])
        obs = check_alignment(seqs, {'a', }, {'b', 'c'})
        obs.drop(columns=['Score'], inplace=True)
        obs.sort_values('Sequence ID', inplace=True)
        obs.index = list(range(len(obs)))
        exp.index = list(range(len(exp)))
        pdt.assert_frame_equal(obs, exp, check_index_type=False)




DATA = io.StringIO(""">a
ATGC
>b
CCGG
>G012345678_1
TTGGCC
>G012345678_2
TTGGCCA
>G012345678_3
TTGGCCT
>G011111111_1
TT
""")

if __name__ == '__main__':
    unittest.main()
