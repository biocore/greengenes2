import click
import pandas as pd
import qiime2
import skbio
import numpy as np
import re
from collections import defaultdict
import gzip
from gg2.species_report import (extract_species, parse_full_length, seq_id,
                                check_alignment, ASV)

@click.group()
def cli():
    pass


@cli.command()
@click.option('--fragments', type=click.Path(exists=True), required=True)
def fragment_report(fragments):
    fragments = qiime2.Artifact.load(fragments).view(pd.Series)

    inv_fragments = defaultdict(list)
    for k, v in fragments.items():
        inv_fragments[str(v)].append(k)

    dist = pd.Series([len(v) for v in inv_fragments.values()])
    print(dist.describe(percentiles=list(np.arange(0, 1, 0.05))))


@cli.command()
@click.option('--species', type=str, required=True)
@click.option('--fragments', type=click.Path(exists=True), required=True)
@click.option('--full-length', type=click.Path(exists=True), required=True)
@click.option('--taxonomy', type=click.Path(exists=True), required=True)
def species_report(species, fragments, full_length, taxonomy):
    taxonomy = qiime2.Artifact.load(taxonomy).view(pd.DataFrame)
    fragments = qiime2.Artifact.load(fragments).view(pd.Series)

    non_asvs = {i for i in taxonomy.index if ASV.match(i) is None}
    taxonomy = taxonomy.loc[non_asvs]

    fragments = {k: str(v) for k, v in fragments.items() if k in non_asvs}
    inv_fragments = defaultdict(list)
    for k, v in fragments.items():
        inv_fragments[v].append(k)

    full_length_in = gzip.open(full_length, 'rt')
    full_length = parse_full_length(full_length_in, non_asvs)

    taxonomy['species'] = taxonomy['Taxon'].apply(lambda x: x.split('; ')[-1])
    taxonomy['binomial'] = taxonomy['species'].apply(extract_species)
    tax_to_species = {r.Index: r.Taxon for r in taxonomy.itertuples()}

    subset = taxonomy[taxonomy['binomial'] == species]
    if len(subset) == 0:
        print("%s not found" % species)
        cli.exit(1)

    subset_fragments = {k: fragments[k] for k in subset.index if k in fragments}
    unique_fragments = defaultdict(list)
    for k, v in subset_fragments.items():
        unique_fragments[v].append(k)

    print("Species: %s" % species)
    print("Number of full length assigned to species: %d" % len(subset))
    print("Number of unique fragment regions: %d" % len(unique_fragments))

    missing = {k for k in subset.index if k not in fragments}
    if missing:
        print("Isolates in which a fragment could not be extracted:")
        for m in sorted(missing):
            print("\t* %s" % m)

    species_ids = set(subset.index)
    bag_of_species = set()
    fragment_hits = []
    alignment_report = {}
    for k, in_species_containing in unique_fragments.items():
        # get the full set of records containing this fragment
        other_records = set(inv_fragments[k])

        # remove any of our within species IDs
        diff = other_records - species_ids

        # if we have other records, record their taxonomies
        # and also calculate alignment reports
        if len(diff) > 0:
            fragment_hits.append((k, diff))
            alignment_report[k] = check_alignment(full_length, set(in_species_containing), diff)
            for d in sorted(diff):
                bag_of_species.add(tax_to_species[d])

    if bag_of_species:
        print("\nSummary of other observed species with fragments:")
        for k in sorted(bag_of_species):
            print("* %s" % k)

    if fragment_hits:
        for k, diff in fragment_hits:
            report = alignment_report.get(k)
            if report is None:
                continue
            report['lineage'] = report['Query'].apply(lambda x: tax_to_species[x])
            report = report.sort_values('Sequence ID', ascending=False)
            print("\nAlignment report from seed: %s" % k)
            for lin, grp in report.groupby('lineage'):
                print("\t%s" % lin)
                for _, row in grp.iterrows():
                    max_id = row['Sequence ID']
                    subj = row['Subject']
                    query = row['Query']
                    length = row['Length']
                    print("\t\tsubject: %s, query %s, sequence ID: %0.2f%%, length: %d" % (subj, query, max_id, length))

        for k, diff in fragment_hits:
            print("\nFragment: %s" % k)
            print("Fragment also found in...")
            for d in sorted(diff):
                print("\t* %s : %s" % (d, tax_to_species[d]))

    if not fragment_hits:
        print("NOTE: appears unique")


if __name__ == '__main__':
    cli()
