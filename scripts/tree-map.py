from qiime2.plugins import emperor, umap, empress
import click
import bp
import skbio
import pandas as pd
import re
import qiime2
import numpy as np
import biom
from gg2.tree_map import (cut, chop_to_species, md_from_tree, species_cover,
                          to_species_as_tips, uniqify, genome_represented,
                          cut_uncovered_species, collapse_species_by_distance)


@click.command()
@click.option('--tree', type=click.Path(exists=True), required=True)
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--rank', type=str, required=True)
@click.option('--use-umap', is_flag=True, default=False)
@click.option('--shear-to-full', is_flag=True, default=False)
@click.option('--half-neighbors', is_flag=True, default=False)
@click.option('--collapse-species', is_flag=True, default=False)
def tree_map(tree, output, rank, use_umap, shear_to_full, half_neighbors,
             collapse_species):
    assert rank in 'pcofgs'
    print('loading...', flush=True)

    tree = bp.parse_newick(open(tree).read())
    if shear_to_full:
        asv = re.compile(r"^[0-9]{8}$")
        operon = re.compile(r"^MJ[0-9]{3}-")

        keep = []
        for i in range(len(tree.B) - 1):
            if tree.B[i] and not tree.B[i + 1]:
                name = tree.name(i)
                if asv.match(name) or operon.match(name):
                    continue
                else:
                    keep.append(name)
        tree = tree.shear(set(keep))

    tree = bp.to_skbio_treenode(tree)
    print('setting genomes...', flush=True)
    genome_represented(tree)
    if collapse_species:
        collapse_species_by_distance(tree, rank)

    print('cutting...', flush=True)
    n_before = len(list(tree.tips()))
    to_species_as_tips(tree, rank)
    n_after = len(list(tree.tips()))
    print(f"Tips: {n_before} -> {n_after}", flush=True)

    uniqify(tree)

    md = md_from_tree(tree)
    md.to_csv(output + '.species-metadata.tsv', sep='\t', index=True, header=True)
    md = qiime2.Metadata(md)

    dm = tree.tip_tip_distances()
    dm_ar = qiime2.Artifact.import_data('DistanceMatrix', dm)
    dm_ar.save(output + '.dm.qza')
    mins = row_mins(dm)
    mins.to_csv(output + '.min-distances.tsv', sep='\t', index=True, header=True)

    if use_umap:
        if half_neighbors:
            n_after = n_after // 2
        pc_ar, = umap.actions.embed(dm_ar, n_neighbors=n_after, number_of_dimensions=3)
    else:
        pc = skbio.stats.ordination.pcoa(dm, number_of_dimensions=5, method='fsvd')
        pc_ar = qiime2.Artifact.import_data('PCoAResults', pc)
    viz, = emperor.actions.plot(pc_ar, md)
    viz.save(output)
    pc_ar.save(output + '.pc.qza')

    for n in tree.traverse(include_self=False):
        if n.length is None:
            n.length = 0
        if n.length < 0:
            print("what")
            print(n.length)
            ntips = list(n.tips())
            print(ntips)
            print(n.name)
            if len(ntips) > 0:
                print(ntips[0].name)
            n.length = 0

    featmd = pd.DataFrame([[n.name, str(n.genome_cover)] for n in tree.tips()],
                          columns=['Feature ID', 'genome_represented']).set_index('Feature ID')
    featmd.to_csv(output + '.feature-metadata.tsv', sep='\t', index=True, header=True)
    featmd = qiime2.Metadata(featmd)

    mat = np.identity(len(dm.ids))
    feattab = biom.Table(mat, list(dm.ids), list(dm.ids))
    feattab_ar = qiime2.Artifact.import_data('FeatureTable[Frequency]', feattab)
    feattab_ar.save(output + '.feature-table.qza')
    phy_ar = qiime2.Artifact.import_data('Phylogeny[Rooted]', tree)
    phy_ar.save(output + '.phylogeny.qza')
    viz, = empress.actions.community_plot(tree=phy_ar, feature_table=feattab_ar, pcoa=pc_ar,
                                          sample_metadata=md, feature_metadata=featmd)

    viz.save(output + '.empress')

if __name__ == '__main__':
    tree_map()
