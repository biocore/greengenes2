import click
import bp
import pandas as pd
from t2t.nlevel import make_consensus_tree, pull_consensus_strings
from functools import partial
import skbio


def get_polyphyletic(df, level):
    """gtdb uses foo, foo_A, foo_B etc for polyphyletic groups"""
    no_suffix = df[level].apply(lambda x: x.split('__', 1)[1])
    unique = pd.Series(no_suffix.unique())
    suffix_removed = unique.apply(lambda x: x.rsplit('_', 1)[0])
    dedup = pd.DataFrame([unique, suffix_removed], index=[level, 'suffix_removed']).T
    polyphyletic_names = set()
    for name, grp in dedup.groupby('suffix_removed'):
        if len(grp) > 1:
            polyphyletic_names.update(set(grp[level]))
    idx = no_suffix[no_suffix.isin(polyphyletic_names)].index
    return set(df.loc[idx, level])


def has_polyphyletic(node, polyphyletic_names):
    check = [node.name, ]
    check += [a.name for a in node.ancestors()]
    if set(check) & polyphyletic_names:
        return True
    else:
        return False


def clean_tree(tree):
    for node in tree.traverse(include_self=True):
        if hasattr(node, 'ChildLookup'):
            delattr(node, 'ChildLookup')


LEVELS = ['domain', 'phylum', 'class', 'order', 'family',
          'genus', 'species']
def parse_lineage(df):
    def splitter(idx, x):
        # one of the records in ltp has dual ;;
        x = x.replace(';;', ';')

        if x.startswith(' Bacteria'):
            # a subset of records in LTP_12_2021 have unusual lineages
            # where the lineage appears duplicated. All the ones found
            # also seem to start with a single space. see for example
            # KF863150
            x = x.strip().split(' Bacteria;')[0]
        parts = x.split(';')
        if idx >= len(parts):
            return ""
        else:
            return parts[idx].strip()

    for idx, level in enumerate(LEVELS):
        df[level] = df['lineage'].apply(partial(splitter, idx))


def strip_ranks(df):
    for level in LEVELS:
        df[level] = df[level].apply(lambda x: x.split('__', 1)[1])


def correct_ltp_issues(ltp_tax):
    # correct mislabelings and issues in LTP
    for _, row in ltp_tax.iterrows():
        if 'Actinomycetotaa' in row['lineage']:
            row['lineage'] = row['lineage'].replace('Actinomycetotaa',
                                                    'Actinomycetota')

        if row['id'] == 'MK559992':
            # MK559992 has a genus that appears inconsistent with the species
            row['lineage'] = row['lineage'].replace('Aureimonas', 'Aureliella')

        if row['id'] == 'HM038000':
            # the wrong order is specified
            row['lineage'] = row['lineage'].replace('Oligoflexia',
                                                    'Bdellovibrionia')

        if ';Campylobacterales;' in row['lineage']:
            if ";Campylobacteria;" not in row['lineage']:
                row['lineage'] = row['lineage'].replace('Campylobacterales;',
                                                        'Campylobacteria;Campylobacterales;')

        if 'incertae_sedis' in row['lineage']:
            parts = row['lineage'].split(';')
            keep = []
            for p in parts:
                if 'incertae_sedis' not in p:
                    keep.append(p)
                else:
                    break
            row['lineage'] = ';'.join(keep)

        if 'ActinomycetotaAcidimicrobiia' in row['lineage']:
            row['lineage'] = row['lineage'].replace('ActinomycetotaAcidimicrobiia',
                                                    'Actinomycetota;Acidimicrobiia')

        if 'ThermodesulfobacteriotaThermodesulfobacteria' in row['lineage']:
            row['lineage'] = row['lineage'].replace('ThermodesulfobacteriotaThermodesulfobacteria',
                                                    'Thermodesulfobacteriota;Thermodesulfobacteria')

def check_overlap(gtdb_tax, ltp_tax):
    # test if any names on overlap
    for i in LEVELS:
        for j in LEVELS:
            if i == j:
                continue
            a = set(gtdb_tax[i]) - set([""])
            b = set(gtdb_tax[j]) - set([""])
            if len(a & b):
                print("gtdb conflict %s %s" % (i, j))
                print(a & b)
                raise ValueError()
            a = set(ltp_tax[i]) - set([""])
            b = set(ltp_tax[j]) - set([""])
            if len(a & b):
                print("ltp conflict %s %s" % (i, j))
                print(a & b)
                raise ValueError()

def check_species_labels(ltp_tax):
    # make sure we do not have species labels where we lack higher level names
    for _, row in ltp_tax.iterrows():
        if row['species'] and not row['genus']:
            row['species'] = ""

def check_consistent_parents(gtdb_tax, ltp_tax):
    # make sure the label at a given level always has the same parent name
    for idx, i in enumerate(LEVELS[1:], 1):  # not domain
        for name, grp in gtdb_tax.groupby(i):
            if len(grp[LEVELS[idx-1]].unique()) != 1:
                print(i, name, list(grp[LEVELS[idx-1]].unique()))
                raise ValueError()
        for name, grp in ltp_tax.groupby(i):
            if name == '':
                continue
            if len(grp[LEVELS[idx-1]].unique()) != 1:
                print(i, name, list(grp[LEVELS[idx-1]].unique()))

def format_name(level, name):
    return "%s__%s" % (level[0], name)

def prep_trees(gtdb_tree, ltp_tree):
    # decorate various flags on to the ltp tree
    # and specifically remove "ChildLookup", which is not used
    # and which appears to break TreeNode.copy
    for node in ltp_tree.traverse(include_self=False):
        node.keepable = False
        if node.is_tip():
            node.keepable = True
        else:
            node.name = format_name(LEVELS[node.Rank], node.name)

    # all tips of gtdb are keepable
    for tip in gtdb_tree.tips():
        tip.keepable = True

    clean_tree(gtdb_tree)
    clean_tree(ltp_tree)

def graft_from_other(other, lookup, polyphyletic_names):
    # in postorder, if we have a name in ltp that is either
    # genus or species, attempt to find the corresponding node
    # in gtdb. ignore nodes which are polyphyletic in gtdb.
    # graft the ungrafted portion of the subtree onto gtdb.
    # remove the subtree from ltp
    for node in list(other.postorder(include_self=False)):
        if node.is_tip():
            continue

        if node.Rank not in (5, 6):  # genus species
            continue

        if node.name in lookup:
            gtdb_node = lookup[node.name]

            if has_polyphyletic(gtdb_node, polyphyletic_names):
                continue

            node.parent.remove(node)
            node = node.copy()

            for desc in node.postorder(include_self=True):
                if not desc.is_tip():
                    desc.keepable = any([c.keepable for c in desc.children])
                    desc.keepable &= not has_polyphyletic(desc, polyphyletic_names)

            for desc in list(node.postorder(include_self=False)):
                if not desc.keepable:
                    desc.parent.remove(desc)

            if node.children:
                gtdb_node.extend(node.children)

def carryover(tax, tree, tree_lookup, polyphyletic_names):
    # for any group which has an exact match, and which is not polyphyletic,
    # adopt all tip labels at the respective taxonomic level
    unused = set(tax['id']) - {n.name for n in tree.tips()}
    for idx, level in enumerate(LEVELS[::-1]):
        for name, grp in tax.groupby(level):
            if name is None or name == '':
                continue

            to_set = set(grp['id']) & unused
            if len(to_set) == 0:
                continue

            query = format_name(level, name)
            if query in tree_lookup:
                current = tree_lookup[query]

                if current.name in polyphyletic_names:
                    continue

                if has_polyphyletic(current, polyphyletic_names):
                    continue

                # 0 species
                # 1 genus
                # etc

                for to_add in range(len(LEVELS) - idx, len(LEVELS)):
                    level_to_add = LEVELS[to_add]
                    to_add_name = level_to_add[0] + '__'
                    blank = skbio.TreeNode(name=to_add_name)

                    blank.Rank = to_add + 1
                    current.append(blank)
                    current = blank

                for tip in sorted(to_set):
                    current.append(skbio.TreeNode(name=tip))
                unused = unused - set(grp['id'])

@click.command()
@click.option('--tree', type=click.Path(exists=True),
              help='The backbone tree')
@click.option('--gtdb', type=click.Path(exists=True),
              help='The GTDB taxonomy with a subset of IDs mapping into the '
                   'backbone')
@click.option('--ltp', type=click.Path(exists=True),
              help='The LTP taxonomy with a subset of IDs mapping into the '
                   'backbone')
@click.option('--output', type=click.Path(exists=False))
def harmonize(tree, gtdb, ltp, output):
    tree = bp.to_skbio_treenode(bp.parse_newick(open(tree).read()))
    gtdb_tax = pd.read_csv(gtdb, sep='\t', names=['id', 'lineage'])
    ltp_tax = pd.read_csv(ltp, sep='\t', names=['id', 'original_species',
                                                'lineage', 'u0', 'type',
                                                'u1', 'u2'])
    tree_tips = {n.name for n in tree.tips()}

    correct_ltp_issues(ltp_tax)

    gtdb_tax = gtdb_tax[gtdb_tax['id'].isin(tree_tips)]
    ltp_tax = ltp_tax[ltp_tax['id'].isin(tree_tips)]

    parse_lineage(gtdb_tax)
    parse_lineage(ltp_tax)

    # the ltp taxonomy doesn't have species in the lineages, so add it in
    ltp_tax['species'] = ltp_tax['original_species']

    check_species_labels(ltp_tax)
    check_overlap(gtdb_tax, ltp_tax)
    check_consistent_parents(gtdb_tax, ltp_tax)

    # gather names that appear polyphyletic in gtdb
    polyphyletic_names = set()
    for level in LEVELS:
        polyphyletic_names.update(get_polyphyletic(gtdb_tax, level))

    # construct taxonomy trees
    for_consensus = [list(v) for _, v in gtdb_tax[LEVELS].iterrows()]
    gtdb_tree, lookup = make_consensus_tree(for_consensus,
                                            check_for_rank=True,
                                            tips=list(gtdb_tax['id']))
    for_consensus = [list(v) for _, v in ltp_tax[LEVELS].iterrows()]
    ltp_tree, ltp_lookup = make_consensus_tree(for_consensus,
                                               check_for_rank=False,
                                               tips=list(ltp_tax['id']))

    prep_trees(gtdb_tree, ltp_tree)

    graft_from_other(ltp_tree, lookup, polyphyletic_names)
    carryover(ltp_tax, gtdb_tree, lookup, polyphyletic_names)

    result = pull_consensus_strings(gtdb_tree, append_prefix=False)
    f = open(output, 'w')
    f.write('\n'.join(result))
    f.write('\n')
    f.close()


if __name__ == '__main__':
    harmonize()
