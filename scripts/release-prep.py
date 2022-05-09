import pandas as pd
import bp
import click
import os
import skbio
import hashlib
import numpy as np
import qiime2
import subprocess
import gzip


def reindex(df, mapping):
    df = df.copy()
    name = df.index.name
    df.index = [mapping.get(i, i) for i in df.index]
    df.index.name = name
    return df

def reindex_tree(tree, mapping):
    names = np.zeros(len(tree.B), dtype=object)
    for i, v in enumerate(tree.B):
        if v:
            name = tree.name(i)
            names[i] = mapping.get(name, name)
        else:
            names[i] = None
    tree.set_names(names)

def loadtax(f):
    df = pd.read_csv(f, sep='\t', dtype=str, names=['Feature ID', 'Taxon'])
    return df.set_index('Feature ID')

def savetax(df, f):
    df.to_csv(f, sep='\t', index=True, header=True, compression='gzip')
    f = f.replace('.gz', '.qza')
    qiime2.Artifact.import_data('FeatureData[Taxonomy]', df).save(f)

def savetree(tree, f):
    if isinstance(tree, bp.BP):
        tree = bp.to_skbio_treenode(tree)
    tree.write(f)
    qiime2.Artifact.import_data('Phylogeny[Rooted]', tree).save(f + '.qza')

def taxtotree(df):
    records = [(i.Index, i.Taxon.split('; ')) for i in df.itertuples()]
    return skbio.TreeNode.from_taxonomy(records)

@click.command()
@click.option('--coarse-level', type=str, required=True)
@click.option('--coarse-threshold', type=float, required=True)
@click.option('--other-level', required=True, multiple=True)
@click.option('--other-level-threshold', required=True, multiple=True)
@click.option('--basename', type=str, required=True)
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--release-name', type=str, required=True)
@click.option('--coarse-level-seqs', type=click.Path(exists=True), required=True)
def release(coarse_level, other_level, other_level_threshold, coarse_threshold,
            basename, output, release_name, coarse_level_seqs):
    def treepath(f):
        return f + f'/{basename}.nwk'

    def taxpath(f):
        return f + f'/{basename}.tsv'

    def releasetaxname(i):
        return f'{output}/{release_name}.taxonomy.{i}.tsv.gz'

    def releasetaxtreename(i):
        return f'{output}/{release_name}.taxonomy.{i}.nwk'

    def releasetreename(i):
        return f'{output}/{release_name}.phylogeny.{i}.nwk'

    def releaseseqsname():
        return f'{output}/{release_name}.seqs.fna.qz'

    for path in [coarse_level] + list(other_level):
        assert os.path.exists(treepath(path))
        assert os.path.exists(taxpath(path))

    THRESHOLD = 'Confidence'

    coarse_tree = bp.parse_newick(open(treepath(coarse_level)).read())
    coarse_tax = loadtax(taxpath(coarse_level))
    coarse_tax[THRESHOLD] = coarse_threshold

    for t, path in zip(other_level_threshold, other_level):
        level_tax = loadtax(taxpath(path))
        coarse_tax.loc[level_tax.index, THRESHOLD] = t
    coarse_tax_tree = taxtotree(coarse_tax)
    savetax(coarse_tax, releasetaxname('id'))
    savetree(coarse_tax_tree, releasetaxtreename('id'))

    # construct mappings for ASVs
    id_to_asv = {}
    id_to_md5 = {}
    for rec in skbio.read(coarse_level_seqs, format='fasta'):
        id = rec.metadata['id']
        if len(rec) < 600:
            md5 = hashlib.md5(rec._string).hexdigest()
            id_to_asv[id] = str(rec)
            id_to_md5[id] = md5

    # various taxonomy perspectives
    coarse_tax_asv = reindex(coarse_tax, id_to_asv)
    coarse_tax_md5 = reindex(coarse_tax, id_to_md5)
    savetax(coarse_tax_asv, releasetaxname('asv'))
    savetax(coarse_tax_md5, releasetaxname('md5'))

    # various taxonomy tree perspectives
    coarse_tax_asv_tree = taxtotree(coarse_tax_asv)
    coarse_tax_md5_tree = taxtotree(coarse_tax_md5)
    savetree(coarse_tax_asv_tree, releasetaxtreename('asv'))
    savetree(coarse_tax_md5_tree, releasetaxtreename('md5'))

    # various phylogeny perspectives
    coarse_tree = bp.parse_newick(open(treepath(coarse_level)).read())
    savetree(coarse_tree, releasetreename('id'))
    coarse_tree = bp.parse_newick(open(treepath(coarse_level)).read())
    reindex_tree(coarse_tree, id_to_asv)
    savetree(coarse_tree, releasetreename('asv'))
    coarse_tree = bp.parse_newick(open(treepath(coarse_level)).read())
    reindex_tree(coarse_tree, id_to_md5)
    savetree(coarse_tree, releasetreename('md5'))

    with open(coarse_level_seqs, 'rb') as in_:
        with gzip.open(releaseseqsname(), 'wb') as out:
            out.writelines(in_)

if __name__ == '__main__':
    release()
