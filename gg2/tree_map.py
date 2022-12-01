import re
from collections import defaultdict
import pandas as pd

def cut(node):
    children = node.children[:]

    if len(children) == 1 and node.children[0].is_tip():
        # if single descendent, have child adopt present name
        name = node.name
        node.name = None
        children[0].name = name
        node.invalidate_caches()
        return children[0]
    else:
        for c in children:
            node.remove(c)
            c.parent = None  # out of caution
        return node

def chop_to_species(tree, rank='s'):
    rank_specifier = f"{rank}__"
    traversal_order = list(tree.postorder())
    for node in traversal_order:
        if node.name is not None and rank_specifier in node.name:
            node = cut(node)
            if node.name.count(';') > 0:
                parts = node.name.split('; ')
                for idx, name in enumerate(parts):
                    if name.startswith(rank_specifier):
                        break
                node.name = '; '.join(parts[:idx+1])

def species_cover(tree, rank='s'):
    rank_specifier = f"{rank}__"
    count = 0
    for node in tree.postorder(include_self=True):
        if node.is_tip():
            if rank_specifier in node.name:
                node.species_cover = True
                count += 1
            else:
                node.species_cover = False
        else:
            node.species_cover = any([c.species_cover for c in node.children])

def cut_uncovered_species(tree, rank_specifier):
    traversal_order = list(tree.preorder(include_self=False))
    removed = set()
    for node in traversal_order:
        if node in removed:
            continue

        if not node.species_cover:
            node.parent.remove(node)
            node.parent = None
            removed.update({d for d in node.traverse(include_self=False)})


def to_species_as_tips(t, rank='s'):
    chop_to_species(t, rank)
    species_cover(t, rank)
    cut_uncovered_species(t, rank)


def genome_represented(t):
    genome = re.compile(r"^G[0|9][0-9]{8}$")
    genome_counts = 0
    for n in t.postorder(include_self=True):
        if n.is_tip():
            if genome.match(n.name):
                n.genome_cover = True

                genome_counts += 1
            else:
                n.genome_cover = False
        else:
            n.genome_cover = any([c.genome_cover for c in n.children])

def md_from_tree(t):
    ranks = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    detail = []
    t.lineage = []
    for n in t.preorder(include_self=False):
        n.lineage = n.parent.lineage[:]
        if n.name is not None:
            n.lineage.extend(n.name.split('; '))

    for n in t.tips():
        detail.append([n.name, '; '.join(n.lineage), str(n.genome_cover)] + n.lineage)

    rank_max = len(n.lineage)
    md = pd.DataFrame(detail,
                      columns=['Feature ID', 'lineage', 'genome_represented'] + ranks[:rank_max])

    for rank in ranks[1:rank_max]:
        top_10 = {k: k for k in md[rank].value_counts().head(10).index}
        md[rank + '_top10'] = md[rank].apply(lambda x: top_10.get(x, 'Other'))

    return md.set_index('Feature ID')


def uniqify(t):
    name_node = defaultdict(list)
    for n in t.tips():
        if n.name is not None:
            for name in n.name.split('; '):
                name_node[name].append(n)
    for name, nodes in name_node.items():
        if len(nodes) > 1:
            for idx, n in enumerate(nodes):
                n.name = n.name.replace(name, f"{name}-{idx}")

def row_mins(dm):
    mat = dm.data.copy()
    ids = np.array(dm.ids)
    np.fill_diagonal(mat, 9999)
    ordered = mat.argsort(axis=1)
    min10_idx = ordered[:, :10]
    min10_vals = np.vstack([mat[i, r] for i, r in enumerate(min10_idx)]).T
    min10_ids = np.vstack([ids[r] for r in min10_idx]).T
    df = pd.DataFrame([ids, ], index=['Feature ID', ]).T
    for i in range(10):
        df['nearest_%d_id' % i] = min10_ids[i]
        df['nearest_%d_dist' % i] = min10_vals[i]
    return df


def collapse_species_by_distance(tree, rank):
    # for species with multiple isolates represented, collect the minimum
    # distance to tip, and add it to our species distance to parent. the reason
    # we do this is to account for within species divergance, and to represent
    # that in the species-species divergence. species represented by single
    # isolates are unaffected as their
    rank_specifier = f"{rank}__"
    for node in tree.traverse():
        if node.name is not None and rank_specifier in node.name:
            dists = [tip.distance(node) for tip in node.tips()]
            min_ = min(dists)
            node.length += min_

            # and perform the cut here
            for c in node.children:
                node.remove(c)
                c.parent = None

