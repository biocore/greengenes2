import click
import bp
import pandas as pd
from t2t.nlevel import make_consensus_tree, pull_consensus_strings
from functools import partial
import skbio
from fuzzywuzzy import fuzz
import re
from collections import defaultdict


POLY_GENERAL = re.compile(r"^([dpcofg]__[a-zA-Z0-9-]+)(_[A-Z]+)?$")
POLY_SPECIES = re.compile(r"^(s__[a-zA-Z0-9-]+)(_[A-Z]+)? ([a-zA-Z0-9-]+)(_[A-Z]+)?$")  # noqa
def get_polyphyletic(df):
    """Compute what labels have a polyphyletic representation"""
    result = {}
    for level in LEVELS[:-1]:
        result[level] = set()

        for name in df[level].unique():
            if len(name) == 3 and name.endswith('__'):
                continue

            match = POLY_GENERAL.match(name)
            if match is None:
                raise ValueError(f"could not match: {name}")

            label, label_poly = match.groups()

            if label_poly is not None:
                # ltp doesn't have rank explicit, and we already have the
                # association so let's strip it
                label = label.split('__', 1)[1]
                result[level].add(label)  # add the label w/o poly suffix

    # poly can be on either part of the binomial so species is special
    level = LEVELS[-1]
    result[level] = set()
    for name in df[level].unique():
        if len(name) == 3 and name.endswith('__'):
            continue

        match = POLY_SPECIES.match(name)
        if match is None:
            raise ValueError(f"could not match species: {name}")

        genus, genus_poly, species, species_poly = match.groups()

        if genus_poly is not None or species_poly is not None:
            # ltp doesn't have rank explicit, and we already have the
            # association so let's strip it
            genus = genus.split('__', 1)[1]
            non_poly_binomial = f'{genus} {species}'
            result[level].add(non_poly_binomial)

    return result

def ids_of_ambiguity_from_polyphyletic(df, poly):
    not_explicitly_set = df[df['explicitly_set'] == False]
    obs = set()
    for level in LEVELS:
        matches = poly[level]
        hits = not_explicitly_set[level].isin(matches)
        ids = not_explicitly_set[hits]['id']
        obs.update(set(ids))
    return obs


LEVELS = ['domain', 'phylum', 'class', 'order', 'family',
          'genus', 'species']
def parse_lineage(df):
    def splitter(idx, x):
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

LTP_RENAMES = [
    # see https://gtdb.ecogenomic.org/genome?gid=GCF_001423565.1
    {'old': 'Actinobacteria;Micrococcales', 'new': 'Actinobacteria;Actinomycetales'},
    {'old': 'Actinobacteria', 'new': 'Actinomycetia'},

    # see https://gtdb.ecogenomic.org/genome?gid=GCF_001423565.1
    {'old': 'Actinomycetota', 'new': 'Actinobacteriota'},

    # see https://gtdb.ecogenomic.org/genome?gid=GCA_001598955.1
    {'old': 'Intrasporangiaceae', 'new': 'Dermatophilaceae'},

    # typos
    {'old': 'ThermodesulfobacteriotaThermodesulfobacteria',
     'new': 'Thermodesulfobacteriota;Thermodesulfobacteria'},
    {'old': 'Nannocystale;Nannocystaceae',
     'new': 'Nannocystales;Nannocystaceae'},
    {'old': 'Bdellovibrionota;Bdellovibrionota', 'new': 'Bdellovibrionota'},
    {'old': 'ActinomycetotaAcidimicrobiia', 'new': 'Actinomycetota;Acidimicrobiia'},
    {'old': 'Actinomycetotaa', 'new': 'Actinomycetota'},
    {'old': 'ThermodesulfobacteriotaThermodesulfobacteria',
     'new': 'Thermodesulfobacteriota;Thermodesulfobacteria'},

    # general remappings
    {'old': 'Bacillota', 'new': 'Firmicutes'},
    {'old': 'Pseudomonadota', 'new': 'Proteobacteria'},
    {'old': 'Armatimonadetes', 'new': 'Armatimonadota'},
    {'old': 'Verrucomicrobiotales',
     'new': 'Verrucomicrobiales'},

    # see https://gtdb.ecogenomic.org/genome?gid=GCA_902779565.1
    {'old': 'Kiritimatiellota', 'new': 'Verrucomicrobiota'},

    # see https://gtdb.ecogenomic.org/genome?gid=GCF_000170755.1
    {'old': 'Lentisphaerota', 'new': 'Verrucomicrobiota'},

    # see https://gtdb.ecogenomic.org/genome?gid=GCA_001802655.1
    {'old': 'Hydrogenophilalia', 'new': 'Gammaproteobacteria'},

    # see https://gtdb.ecogenomic.org/genome?gid=GCA_001508095.1
    {'old': 'Thermodesulfobacteriota', 'new': 'Desulfobacterota'},

    # https://forum.gtdb.ecogenomic.org/t/what-happend-with-betaproteobacteria/131/2
    {'old': 'Betaproteobacteria', 'new': 'Gammaproteobacteria'},

    # https://gtdb.ecogenomic.org/genome?gid=GCA_000690595.2
    {'old': 'Natrialbales', 'new': 'Halobacteriales'},

    # https://gtdb.ecogenomic.org/genome?gid=GCA_000145985.1
    {'old': 'Desulfurococcales', 'new': 'Sulfolobales'},

    # note: https://www.nature.com/articles/s41564-021-00918-8.pdf?proof=t
    # table 1, checked against what seems to also be in LTP
    {'old': 'Thermoprotei', 'new': 'Thermoproteia'},
    {'old': 'Crenarchaeota', 'new': 'Thermoproteota'},
    {'old': 'Euryarchaeota;Halobacteria', 'new': 'Halobacteriota;Halobacteria'},
    {'old': 'Euryarchaeota;Methanomicrobia;Methanosarcinales',
     'new': 'Halobacteriota;Methanosarcinia;Methanosarcinales'},
    {'old': 'Euryarchaeota;Methanomicrobia',  # map the remaining
     'new': 'Halobacteriota;Methanomicrobia'},
    {'old': 'Euryarchaeota;Thermoplasmata',
     'new': 'Thermoplasmatota;Thermoplasmata'},
    {'old': 'Euryarchaeota;Archaeoglobi',
     'new': 'Halobacteriota;Archaeoglobi'},

    # https://gtdb.ecogenomic.org/genome?gid=GCA_000613065.1
    {'old': 'Nostocales', 'new': 'Cyanobacteriales'},

    # see https://gtdb.ecogenomic.org/searches?s=al&q=g__schaalia
    {'old': 'Schaalia;Schaalia', 'new': 'Pauljensenia;Pauljensenia'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Dermacoccaceae
    {'old': 'Dermacoccaceae', 'new': 'Dermatophilaceae'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Cellulosimicrobium
    {'old': 'Promicromonosporaceae;Cellulosimicrobium',
     'new': 'Cellulomonadaceae;Cellulosimicrobium'},

    # Alteromonadaceae falls under Enterobacterales now
    # https://gtdb.ecogenomic.org/searches?s=al&q=Alteromonadaceae%3B
    {'old': 'Alteromonadales;Alteromonadaceae',
     'new': 'Enterobacterales;Alteromonadaceae'},

    # Chlorobia now falls under Bacteroidota
    # https://gtdb.ecogenomic.org/searches?s=al&q=Chlorobia
    {'old': 'Chlorobiota;Chlorobia', 'new': 'Bacteroidota;Chlorobia'},

    # Chloroflexia now falls under Chloroflexota
    # https://gtdb.ecogenomic.org/searches?s=al&q=Chloroflexota
    {'old': 'Chloroflexiota;Chloroflexia',
     'new': 'Chloroflexota;Chloroflexia'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Ignavibacteria
    # falls under Bacteroidota now
    {'old': 'Ignavibacteriota;Ignavibacteria',
     'new': 'Bacteroidota;Ignavibacteria'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Nautiliales
    {'old': 'Nautiliia;Nautiliales', 'new': 'Campylobacteria;Nautiliales'},

    # https://gtdb.ecogenomic.org/genome?gid=GCF_002149925.1
    {'old': 'Gammaproteobacteria;Thiotrichales',
     'new': 'Gammaproteobacteria;Beggiatoales'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Ectothiorhodospiraceae
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Alkalilimnicola',
     'new': 'Nitrococcales;Halorhodospiraceae;Alkalilimnicola'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Halorhodospira',
     'new': 'Nitrococcales;Halorhodospiraceae;Halorhodospira'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Thiorhodospira',
     'new': 'Ectothiorhodospirales;Ectothiorhodospiraceae;Thiorhodospira'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Spiribacter',
     'new': 'Nitrococcales;Nitrococcaceae;Spiribacter'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Thioalkalivibrio',
     'new': 'Ectothiorhodospirales;Thioalkalivibrionaceae;Thioalkalivibrio'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Arhodomonas',
     'new': 'Nitrococcales;Nitrococcaceae;Arhodomonas'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Ectothiorhodospira',
     'new': 'Ectothiorhodospirales;Ectothiorhodospiraceae;Ectothiorhodospira'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Acidihalobacter',
     'new': 'Ectothiorhodospirales;Acidihalobacteraceae;Acidihalobacter'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Acidihalobacter',
     'new': 'Ectothiorhodospirales;Acidihalobacteraceae;Acidihalobacter'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Halofilum',
     'new': 'XJ16;Halofilaceae;Halofilum'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Oceanococcus',
     'new': 'Nevskiales;Oceanococcaceae;Oceanococcus'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Thioalbus',
     'new': 'DSM-26407;DSM-26407;Thioalbus'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;g__Inmirania',
     'new': 'DSM-100275;DSM-100275;Inmirania'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Thiogranum',
     'new': 'DSM-19610;DSM-19610;Thiogranum'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Thiohalomonas',
     'new': 'Thiohalomonadales;Thiohalomonadaceae;Thiohalomonas'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Aquisalimonas',
     'new': 'Nitrococcales;Aquisalimonadaceae;Aquisalimonas'},
    {'old': 'Chromatiales;Ectothiorhodospiraceae;Thiohalospira',
     'new': 'Thiohalospirales;Thiohalospiraceae;Thiohalospira'},
    # genus changed so make sure to reflect it in the binomial
    {'old': 'Bacteria;Pseudomonadota;Gammaproteobacteria;Chromatiales;Ectothiorhodospiraceae;Alkalispirillum;Alkalispirillum mobile',
     'new': 'Bacteria;Proteobacteria;Gammaproteobacteria;Nitrococcales;Halorhodospiraceae;Alkalilimnicola;Alkalilimnicola mobilis'},
    # these aren't in gtdb so we will move these records to ignore
    # Methylohalomonas
    # Natronospira
    # Natronocella
    # Methylonatrum

    # https://gtdb.ecogenomic.org/searches?s=al&q=Hyphomicrobiaceae
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Hyphomicrobium',
     'new': 'Rhizobiales;Hyphomicrobiaceae;Hyphomicrobium'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Rhodomicrobium',
     'new': 'Rhizobiales;Rhodomicrobiaceae;Rhodomicrobium'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Methyloceanibacter',
     'new': 'Rhizobiales;Methyloligellaceae;Methyloceanibacter'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Prosthecomicrobium',
     'new': 'Rhizobiales;Ancalomicrobiaceae;Prosthecomicrobium'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Methyloligella',
     'new': 'Rhizobiales;Methyloligellaceae;Methyloceanibacter'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Methyloceanibacter',
     'new': 'Rhizobiales;Methyloligellaceae;Methyloceanibacter'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Methyloceanibacter',
     'new': 'Rhizobiales;Methyloligellaceae;Methyloceanibacter'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Rhodoplanes',
     'new': 'Rhizobiales;Xanthobacteraceae;Rhodoplanes'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Dichotomicrobium',
     'new': 'Rhizobiales;Rhodomicrobiaceae;Dichotomicrobium'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Caenibius',
     'new': 'Sphingomonadales;Sphingomonadaceae;Caenibius'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Aquabacter',
     'new': 'Rhizobiales;Xanthobacteraceae;Aquabacter'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Angulomicrobium',
     'new': 'Rhizobiales;Xanthobacteraceae;Angulomicrobium'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Prosthecomicrobium',
     'new': 'Rhizobiales;Kaistiaceae;Prosthecomicrobium'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Limoniibacter',
     'new': 'Rhizobiales;Rhizobiaceae;Limoniibacter'},
    {'old': 'Hyphomicrobiales;Hyphomicrobiaceae;Filomicrobium',
     'new': 'Rhizobiales;Hyphomicrobiaceae;Filomicrobium'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Kiloniellaceae
    {'old': 'Rhodospirillales;Kiloniellaceae;Kiloniella',
     'new': 'Kiloniellales;Kiloniellaceae;Kiloniella'},
    {'old': 'Rhodospirillales;Kiloniellaceae;Aestuariispira',
     'new': 'UBA8366;GCA-2696645;Aestuariispira'},
    {'old': 'Rhodospirillales;Rhodospirillaceae;Denitrobaculum',
     'new': 'Kiloniellales;Kiloniellaceae;Denitrobaculum'},
    {'old': 'Rhodospirillales;Kiloniellaceae;Curvivirga',
     'new': 'UBA8366;GCA-2696645;Curvivirga'},
    {'old': 'Hyphomicrobiales;Rhodovibrionaceae;Pelagibius',
     'new': 'Kiloniellales;Kiloniellaceae;Pelagibius'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Parvibaculaceae
    {'old': 'Hyphomicrobiales;Parvibaculaceae;Parvibaculum',
     'new': 'Parvibaculales;Parvibaculaceae;Parvibaculum'},
    {'old': 'Hyphomicrobiales;Parvibaculaceae;Tepidicaulis',
     'new': 'Parvibaculales;Parvibaculaceae;Tepidicaulis'},
    {'old': 'Hyphomicrobiales;Parvibaculaceae;Rhodoligotrophos',
     'new': 'Rhizobiales;Im1;Rhodoligotrophos'},
    {'old': 'Hyphomicrobiales;Parvibaculaceae;Pyruvatibacter',
     'new': 'Parvibaculales;CGMCC-115125;Pyruvatibacter'},
    {'old': 'Hyphomicrobiales;Parvibaculaceae;Kaustia',
     'new': 'Rhizobiales;Im1;Kaustia'},
    {'old': 'Hyphomicrobiales;Parvibaculaceae;Anderseniella',
     'new': 'Rhizobiales;Aestuariivirgaceae;Anderseniella'},
    {'old': 'Hyphomicrobiales;Parvibaculaceae;Tepidicaulis',
     'new': 'Parvibaculales;Parvibaculaceae;Tepidicaulis'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Piscirickettsiaceae
    {'old': 'Thiotrichales;Piscirickettsiaceae;Cycloclasticus',
     'new': 'Methylococcales;Cycloclasticaceae;Cycloclasticus'},
    {'old': 'Thiotrichales;Piscirickettsiaceae;Hydrogenovibrio',
     'new': 'Thiomicrospirales;Thiomicrospiraceae;Hydrogenovibrio'},
    {'old': 'Thiotrichales;Piscirickettsiaceae;Methylophaga',
     'new': 'Nitrosococcales;Methylophagaceae;Methylophaga'},
    {'old': 'Thiotrichales;Piscirickettsiaceae;Piscirickettsia',
     'new': 'Piscirickettsiales;Piscirickettsiaceae;Piscirickettsia'},
    {'old': 'Thiotrichales;Piscirickettsiaceae;Sulfurivirga',
     'new': 'Thiomicrospirales;Thiomicrospiraceae;Sulfurivirga'},
    {'old': 'Thiotrichales;Piscirickettsiaceae;Thiomicrorhabdus',
     'new': 'Thiomicrospirales;Thiomicrospiraceae;Thiomicrorhabdus'},
    {'old': 'Thiotrichales;Piscirickettsiaceae;Thiomicrospira',
     'new': 'Thiomicrospirales;Thiomicrospiraceae;Thiomicrospira'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Sporomusaceae
    {'old': 'Selenomonadales;Sporomusaceae;Acetonema',
     'new': 'Sporomusales_A;Acetonemaceae;Acetonema'},
    {'old': 'Selenomonadales;Sporomusaceae;Anaeromusa',
     'new': 'Anaeromusales;Anaeromusaceae;Anaeromusa'},
    {'old': 'Selenomonadales;Sporomusaceae;Anaerosporomusa',
     'new': 'Sporomusales_A;Acetonemaceae;Anaerosporomusa'},
    {'old': 'Selenomonadales;Sporomusaceae;Dendrosporobacter',
     'new': 'DSM-1736;Dendrosporobacteraceae;Dendrosporobacter'},
    {'old': 'Selenomonadales;Sporomusaceae;Methylomusa',
     'new': 'Sporomusales;Sporomusaceae;Methylomusa'},
    {'old': 'Selenomonadales;Sporomusaceae;Pelosinu',
     'new': 'Propionisporales;Propionisporaceae;Pelosinus'},
    {'old': 'Selenomonadales;Sporomusaceae;Propionispora',
     'new': 'Propionisporales;Propionisporaceae;Propionispora'},
    {'old': 'Selenomonadales;Sporomusaceae;Sporolituus',
     'new': 'Sporomusales;Thermosinaceae;Sporolituus'},
    {'old': 'Selenomonadales;Sporomusaceae;Sporomusa',
     'new': 'Sporomusales;Sporomusaceae;Sporomusa'},
    {'old': 'Selenomonadales;Sporomusaceae;Thermosinus',
     'new': 'Sporomusales;Thermosinaceae;Thermosinus'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Guyparkeria
    # this species isn't present but the genus appears consistent
    {'old': 'Chromatiales;Thioalkalibacteraceae;Guyparkeria;Guyparkeria hydrothermalis',
     'new': 'Halothiobacillales;Halothiobacillaceae;Guyparkeria;Guyparkeria hydrothermalis'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Haloarchaeobius
    {'old': 'Halobacteriales;Halobacteriaceae;Haloarchaeobius',
     'new': 'Halobacteriales;Natrialbaceae;Haloarchaeobius'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Luteolibacter
    {'old': 'Verrucomicrobiotaceae;Luteolibacter',
     'new': 'Akkermansiaceae;Luteolibacter'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Magnetospirillum
    {'old': 'Rhodospirillaceae;Magnetospirillum',
     'new': 'Magnetospirillaceae;Magnetospirillum'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Oleispira
    {'old': 'Oceanospirillaceae;Oleispira',
     'new': 'DSM-6294;Oleispira'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Phaeospirillum
    {'old': 'Rhodospirillaceae;Phaeospirillum',
     'new': 'Magnetospirillaceae;Phaeospirillum'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Phyllobacterium
    {'old': 'Hyphomicrobiales;Phyllobacteriaceae;Phyllobacterium',
     'new': 'Rhizobiales_A;Rhizobiaceae_A;Phyllobacterium'},

    # https://gtdb.ecogenomic.org/searches?s=al&q=Pseudaminobacter
    {'old': 'Hyphomicrobiales;Phyllobacteriaceae',
     'new': 'Rhizobiales;Rhizobiaceae'},

    # Brucella is not getting picked up completely in automatic
    # rewrite. the rewrite seems pretty consistent
    # https://gtdb.ecogenomic.org/searches?s=al&q=g__Brucella
    {'old': 'Hyphomicrobiales;Brucellaceae;Brucella',
     'new': 'Rhizobiales_A;Rhizobiaceae_A;Brucella'},
]

LTP_TO_DROP = {
    # Genus is not in GTDB
    # NCBI taxonomy reports eubacteriales unclassified family XII
    # LPSN is consistent (https://lpsn.dsmz.de/family/eubacteriales-no-family)
    # We can at best guess based on taxonomy what this might be, as such
    # we should remove it and allow tax2tree to resolve
    'AY272039',

    # The Mogibacterium genus is not polyphyletic, however many
    # Mogibacterium have been reclassified into different genera
    # in GTDB. These species were not found in GTDB.
    # Based on this, we should omit and allow tax2tree to resolve
    # https://gtdb.ecogenomic.org/searches?s=al&q=%22Mogibacterium%22
    'AB037875',
    'AB021702',

    # This species is not in GTDB, its genus is polyphyletic and there
    # appears to be change in classification.
    # https://gtdb.ecogenomic.org/searches?s=al&q=%22Fusibacter%22
    'KJ420408',
    'LM999901',

    # https://gtdb.ecogenomic.org/genome?gid=GCA_005048345.1
    # species failed QC in GTDB so we do not have its GTDB lineage
    # we do have a fully specified NCBI lineage, however it is polyphyletic
    # in a few ranks in GTDB so we are omitting.
    'GQ461828',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Sedimentibacter
    # species is not present in GTDB. There is some evidence of
    # reclassification, so removing.
    'AF433166',

    # Species and genus not present in GTDB, and the family is
    # incertae sedis in NCBI taxonomy so omitting
    'AB218661',

    # Species is not present in GTDB and there appears to be some
    # reclassification
    # https://gtdb.ecogenomic.org/searches?s=al&q=%22Thermaerobacter%22
    'AB061441',
    'AY936496',
    'AB454087',

    # Species, genus and family is not present
    # LPSN notes family is incertae sedis
    # https://lpsn.dsmz.de/family/bacillota-no-family
    'HQ452857',

    # The class doesn't exist in GTDB so let's drop
    # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=1852309
    # https://gtdb.ecogenomic.org/searches?s=al&q=Calorithrix
    'KU306340',

    # No family per lpsn
    # https://lpsn.dsmz.de/family/gammaproteobacteria-no-family
    'KY643661',

    # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=332340
    # https://gtdb.ecogenomic.org/searches?s=al&q=halomonadaceae
    # family is present but evidence it was split so dropping
    'DQ019934',
    'DQ129689',

    # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=1229656
    # incertae sedis in ncbi so dropping
    'JX412366',

    # the family Ectothiorhodospiraceae was heavily rewritten in gtdb
    # and the genera for these records is undescribed by gtdb representatives
    'DQ834966',
    'KU720568',
    'EF103128',
    'DQ789390',

    # the family Hyphomicrobiaceae was heavily rewritten in gtdb
    # and the genera for these records is undescribed by gtdb representatives
    'HM037996',
    'GU269549',
    'GU269548',
    'X97693',

    # the family Kiloniellaceae was heavily rewritten in gtdb
    # and the genera for these records is undescribed by gtdb representatives
    'KP162059',

    # the family Piscirickettsiaceae was heavily rewritten in gtdb
    # and the genera for these reocrds is undescribed by gtdb representatives
    'JQ080912',

    # the family Sporomusaceae was heavily rewritten in gtdb
    # and the genera for these reocrds is undescribed by gtdb representatives
    'AJ010960',
    'HG317005',
    'AJ010961',

    # the family for Brucella was rewritten, and this species
    # isn't described by GTDB so unclear where to place it
    'HG932316',

    # Rheinheimera is split in GTDB so unclear how to map these
    # records
    'GQ168584',
    'KJ816861',
    'AY701891',
    'LT627667',
    'JQ922423',
    'MH087228',
    'MK512363',
    'KM588222',
    'EU183319',
    'JQ922424',
    'KM588221',
    'EF575565',
    'HQ111524',
    'KT900247',
    'KM025195',
    'DQ298025',
    'LC004727',
}

# LTP has incertae sedis mappings which are not meaningful for decoration.
# Many of the species exist in GTDB, and GTDB has resolved classification
# for the clades. The mappings here are manually obtained going from the
# LTP record to the GTDB lineage. We prioritize GTDB reference and NCBI
# type materials. When the species is not present, if the genus appears
# monophyletic, we add the species. If the genus appears polyphyletic,
# then we cannot be sure what label it falls under, so we place under the
# nominal name and let tax2tree sort it out with its polyphyletic
# capabilities. If there appears to be reclassifications of genus, then we
# omit the record as we cannot be sure where it falls.
INCERTAE_SEDIS_MAPPINGS = {
    # https://gtdb.ecogenomic.org/genome?gid=GCA_003447605.1
    'X60418': 'Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Plesiomonas;Plesiomonas shigelloides',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_011207455.1
    'KY471008': 'Bacteria;Desulfobacterota;Thermodesulfobacteria;Thermodesulfobacteriales;ST65;Thermosulfuriphilus;Thermosulfuriphilus ammonigenes',

    # same genus, not the same species, but placing within the genus
    # per gtdb. note that the genus is polyphyletic so we are relying
    # on tax2tree to resolve it
    # https://gtdb.ecogenomic.org/genome?gid=GCA_000438555.1
    'AM072763': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium;Exiguobacterium artemiae',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000019905.1
    'CP001022': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A sibiricum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000620805.1
    'DQ019165': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A undae',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000702625.1
    'AB105164': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A oxidotolerans',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000299435.1
    'DQ019164': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A antarcticum',

    # note, appears renamed in GTDB from soli -> antarcticum
    # https://gtdb.ecogenomic.org/genome?gid=GCF_017874675.1
    'AY864633': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A antarcticum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000702605.1
    'DQ019167': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A acetylicum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_001939065.1
    'AJ846291': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A indicum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_001456895.1
    'JF893462': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A indicum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000714435.1
    'EU379016': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium;Exiguobacterium alkaliphilum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_001766415.1
    'AB680657': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium;Exiguobacterium aurantiacum_A',

    # species isn't in GTDB, but genus is, placing under the genus
    # and lets let tax2tree sort out polyphyletic
    'JF775503': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium;Exiguobacterium aquaticum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_005960665.1
    'AM072764': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium;Exiguobacterium mexicanum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000620845.1
    'AY594266': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium;Exiguobacterium marinum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_001909285.1
    'AY818050': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium;Exiguobacterium profundum_A',

    # species isnt' in GTDB, but genus is, placing under the genus
    'AY594264': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium;Exiguobacterium aestuarii',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_003344535.1
    'MH375463': 'Bacteria;Firmicutes;Bacilli;Exiguobacterales;Exiguobacteraceae;Exiguobacterium_A;Exiguobacterium_A flavidum',

    # gemella does not appear polyphyletic. species isnt in gtdb, placing under the genus
    # https://gtdb.ecogenomic.org/searches?s=al&q=gemella
    'HM103931': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella parahaemolysans',

    # gemella does not appear polyphyletic. species isnt in gtdb, placing under the genus
    # https://gtdb.ecogenomic.org/searches?s=al&q=gemella
    'HM103934': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella taiwanensis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_901873445.1
    'L14326': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella haemolysans_C',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_900476045.1
    'L14327': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella morbillorum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000701685.1
    'Y13364': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella sanguinis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000469465.1
    'Y13365': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella bergeri',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000425665.1
    'AJ251987': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella cuniculi',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_001553005.1
    'EU427463': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella asaccharolytica',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_013415235.1
    'Y17280': 'Bacteria;Firmicutes;Bacilli;Staphylococcales;Gemellaceae;Gemella;Gemella palaticanis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000510645.1
    'AJ242495': 'Bacteria;Firmicutes;Bacilli;Thermicanales;Thermicanaceae;Thermicanus;Thermicanus aegyptius',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_004345705.1
    'KX822012': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Thermotaleaceae;Marinisporobacter;Marinisporobacter balticus',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_002998925.1
    'AB037874': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Anaerovoracaceae;Mogibacterium;Mogibacterium diversum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_002243385.1
    'CP016199': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Anaerovoracaceae;Mogibacterium;Mogibacterium pumilum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000525775.1
    'Z36296': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Anaerovoracaceae;Mogibacterium;Mogibacterium timidum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_010669305.1
    'AB298771': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Anaerovoracaceae;Aminipila;Aminipila butyrica',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000426305.1
    'AJ251215': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Anaerovoracaceae;Anaerovorax;Anaerovorax odorimutans',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_018390735.1
    'AF050099': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Acidaminobacteraceae;Fusibacter;Fusibacter paucivorans',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_016908355.1
    'FR851323': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Acidaminobacteraceae;Fusibacter_C;Fusibacter_C tunisiensis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_900103005.1
    'AF016691': 'Bacteria;Firmicutes_A;Clostridia;Peptostreptococcales;Acidaminobacteraceae;Acidaminobacter;Acidaminobacter hydrogenoformans',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000312505.2
    'HM587321': 'Bacteria;Firmicutes_A;Clostridia;Tissierellales;Peptoniphilaceae;Fenollaria;Fenollaria massiliensis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000311985.1
    'JN837487': 'Bacteria;Firmicutes_A;Clostridia;Tissierellales;Peptoniphilaceae;Kallipyga;Kallipyga massiliensis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_900130025.1
    'AF358114': 'Bacteria;Firmicutes_A;Clostridia;Tissierellales;Sporanaerobacteraceae;Sporanaerobacter;Sporanaerobacter acetigenes',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_013403245.1
    'L11305': 'Bacteria;Firmicutes_A;Clostridia;Tissierellales;Sedimentibacteraceae;Sedimentibacter;Sedimentibacter hydroxybenzoicus',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_007830175.1
    'AJ404680': 'Bacteria;Firmicutes_A;Clostridia;Tissierellales;Sedimentibacteraceae;Sedimentibacter;Sedimentibacter saalensis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_017873955.1
    'AB598276': 'Bacteria;Firmicutes_A;Clostridia;Tissierellales;Sedimentibacteraceae;Sedimentibacter;Sedimentibacter acidaminivorans',

    # https://gtdb.ecogenomic.org/searches?s=al&q=%22Anaerobranca%22
    # species is not present but genus appears consistent
    'U21809': 'Bacteria;Firmicutes_D;Proteinivoracia;Proteinivoracales;Proteinivoraceae;Anaerobranca;Anaerobranca horikoshii',

    # https://gtdb.ecogenomic.org/searches?s=al&q=%22Anaerobranca%22
    # species is not present but genus appears consistent
    'EF190921': 'Bacteria;Firmicutes_D;Proteinivoracia;Proteinivoracales;Proteinivoraceae;Anaerobranca;Anaerobranca zavarzinii',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_900142275.1
    'AY064218': 'Bacteria;Firmicutes_D;Proteinivoracia;Proteinivoracales;Proteinivoraceae;Anaerobranca;Anaerobranca californiensis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_900111575.1
    'AF203703': 'Bacteria;Firmicutes_D;Proteinivoracia;Proteinivoracales;Proteinivoraceae;Anaerobranca;Anaerobranca gottschalkii',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_900167165.1
    # note that GTDB reclassified the type strain
    'AY673988': 'Bacteria;Firmicutes_B;GCA-003054495;Carboxydocellales;Carboxydocellaceae;Carboxydocella;Carboxydocella thermautotrophica',

    # https://gtdb.ecogenomic.org/genome?gid=GCA_003054495.1
    'AY061974': 'Bacteria;Firmicutes_B;GCA-003054495;Carboxydocellales;Carboxydocellaceae;Carboxydocella;Carboxydocella thermautotrophica',

    # https://gtdb.ecogenomic.org/searches?s=al&q=%22Carboxydocella%22
    # species is not present but genus appears consistent
    'GU584133': 'Bacteria;Firmicutes_B;GCA-003054495;Carboxydocellales;Carboxydocellaceae;Carboxydocella;Carboxydocella manganica',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000183545.2
    'AF343566': 'Bacteria;Firmicutes_E;Thermaerobacteria;Thermaerobacterales;Thermaerobacteraceae;Thermaerobacter;Thermaerobacter subterraneus',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000184705.1
    'CP002344': 'Bacteria;Firmicutes_E;Thermaerobacteria;Thermaerobacterales;Thermaerobacteraceae;Thermaerobacter;Thermaerobacter marianensis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000213255.1
    'CP002360': 'Bacteria;Firmicutes_A;Clostridia;Mahellales;Mahellaceae;Mahella;Mahella australiensis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_003991135.1
    'KC794015': 'Bacteria;Firmicutes_F;Halanaerobiia;DY22613;DY22613;Anoxybacter;Anoxybacter fermentans',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000946815.1
    'EU386162': 'Bacteria;Firmicutes_B;Moorellia;Thermacetogeniales;Thermacetogeniaceae;Syntrophaceticus;Syntrophaceticus schinkii',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_013664685.1
    'MN602556': 'Bacteria;Firmicutes_G;UBA4882;UBA10575;UBA10575;Capillibacterium;Capillibacterium thermochitinicola',

    # https://gtdb.ecogenomic.org/searches?s=al&q=%22Exilispira%22
    # species is not present but genus appears consistent
    'AB364473': 'Bacteria;Spirochaetota;JAAYUW01;JAAYUW01;JAAYUW01;Exilispira;Exilispira thermophila',

    # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=434126
    # using GTDB phylum
    # https://gtdb.ecogenomic.org/searches?s=al&q=Synergistaceae
    'EF468685': 'Bacteria;Synergistota;Synergistia;Synergistales;Synergistaceae;Rarimicrobium;Rarimicrobium hominis',

    # https://gtdb.ecogenomic.org/searches?s=al&q=flintibacter
    # flintibacter looks to be Lawsonibacter in GTDB
    'KF447772': 'Bacteria;Firmicutes_A;Clostridia;Oscillospirales;Oscillospiraceae;Lawsonibacter;Lawsonibacter butyricus',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Alkalimonas
    # species isn't present but genus is
    'AB270706': 'Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Alteromonadaceae;Alkalimonas;Alkalimonas collagenimarina',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Thorsellia
    # genus is present but not species
    'KM269290': 'Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Thorsellia;Thorsellia kenyensis',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Thorsellia
    # genus is present but not species
    'KM269289': 'Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Thorsellia;Thorsellia kandunguensis',

    # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=1812101
    # https://gtdb.ecogenomic.org/searches?s=al&q=Thorselliaceae
    # family is present but now placed under enterobacteraceae
    'KU748636': 'Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Coetzeea;Coetzeea brasiliensis',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Alkalimonas
    # genus is present
    'X92130': 'Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Alteromonadaceae;Alkalimonas;Alkalimonas delamerensis',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Cellvibrionaceae
    # family is present and seems well represented
    'AB900126': 'Bacteria;Proteobacteria;Gammaproteobacteria;Pseudomonadales;Cellvibrionaceae;Marinibactrum;Marinibactrum halimedae',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Litorivivens
    # genus is present
    'LC167346': 'Bacteria;Proteobacteria;Gammaproteobacteria;Pseudomonadales;Spongiibacteraceae;Litorivivens;Litorivivens aequoris',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Thiolapillus
    # genus is present
    'AP012273': 'Bacteria;Proteobacteria;Gammaproteobacteria;Chromatiales;Sedimenticolaceae;Thiolapillus;Thiolapillus brandeum',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Sedimenticola
    # genus is present
    'ATZE01000023': 'Bacteria;Proteobacteria;Gammaproteobacteria;Chromatiales;Sedimenticolaceae;Sedimenticola;Sedimenticola selenatireducens',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Sedimenticola
    # genus is present
    'JN882289': 'Bacteria;Proteobacteria;Gammaproteobacteria;Chromatiales;Sedimenticolaceae;Sedimenticola;Sedimenticola thiotaurini',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Thiohalobacter
    # genus is present
    'FJ482231': 'Bacteria;Proteobacteria;Gammaproteobacteria;Thiohalobacterales;Thiohalobacteraceae;Thiohalobacter;Thiohalobacter thiocyanaticus',

    # https://gtdb.ecogenomic.org/searches?s=al&q=Anaerolineaceae%3B
    # family is present and well represented
    'AB669272': 'Bacteria;Chloroflexota;Anaerolineae;Anaerolineales;Anaerolineaceae;Thermomarinilinea;Thermomarinilinea lacunifontana',
}

MANUAL_REMAPPINGS = {
    # https://gtdb.ecogenomic.org/genome?gid=GCF_005490985.1
    # this record appears mislabeled in LTP as the Genbank detail is
    # substantially different
    # https://www.ncbi.nlm.nih.gov/nuccore/GU269547
    'GU269547': 'Bacteria;Actinobacteriota;Actinomycetia;Actinomycetales;Microbacteriaceae;Curtobacterium;Curtobacterium oceanosedimentum',
}

def adjust_ltp(ltp_tax, gtdb_unique, ncbi_to_gtdb):
    mappings = MANUAL_REMAPPINGS.copy()

    # remap to gtdb if a unique exact match for a NCBI species name exists
    for _, row in ltp_tax.iterrows():
        if row['original_species'] in gtdb_unique:
           mappings[row['id']] = gtdb_unique[row['original_species']]

    # carry over our incertae sedis mappings
    mappings.update(INCERTAE_SEDIS_MAPPINGS.copy())

    # remove extraneous quotes in the ltp data
    ltp_tax['lineage'] = ltp_tax['lineage'].apply(lambda x: x.replace('"', ''))
    ltp_tax['original_species'] = ltp_tax['original_species'].apply(
        lambda x: x.replace('"', ''))

    # the ltp taxonomy doesn't have species in the lineages, so add it in
    def append_species(row):
        return row['lineage'] + ';' + row['original_species']
    ltp_tax['lineage'] = ltp_tax.apply(append_species, axis=1)

    # tedious manual modifications that don't follow a general pattern
    for _, row in ltp_tax.iterrows():
        if ';;' in row['lineage']:
            # one of the records in ltp has dual ;;
            row['lineage'] = row['lineage'].replace(';;', ';')

        if row['lineage'].startswith(' Bacteria'):
            # note the prefixed whitespace
            # a subset of records in LTP_12_2021 have unusual lineages
            # where the lineage appears duplicated. All the ones found
            # also seem to start with a single space. see for example
            # KF863150
            row['lineage'] = row['lineage'].strip().split(' Bacteria;')[0]

        if row['id'] == 'MK559992':
            # MK559992 has a genus that appears inconsistent with the species
            row['lineage'] = row['lineage'].replace('Aureimonas', 'Aureliella')

        if row['id'] == 'HM038000':
            # the wrong order is specified
            row['lineage'] = row['lineage'].replace('Oligoflexia',
                                                    'Bdellovibrionia')

        if ';Campylobacterales;' in row['lineage']:
            # typo
            if ";Campylobacteria;" not in row['lineage']:
                row['lineage'] = row['lineage'].replace('Campylobacterales;',
                                                        'Campylobacteria;Campylobacterales;')

        if ' subsp. ' in row['lineage']:
            # we can't handle subspecies so let's remove
            parts = row['lineage'].split(';')
            species = parts[-1]
            species = species.split(' subsp. ')[0]
            parts[-1] = species
            row['lineage'] = ';'.join(parts)

        if row['id'] == 'AB104858':
            # species name typo
            row['lineage'] = row['lineage'].replace('Methanothermobacter wolfei',
                                                    'Methanothermobacter wolfeii')
            row['original_species'] = 'Methanothermobacter wolfeii'

    # replace lineages from incertae sedis and manual
    ltp_tax['explicitly_set'] = False
    ltp_tax.set_index('id', inplace=True)
    for k, v in mappings.items():
        ltp_tax.loc[k, 'lineage'] = v
        ltp_tax.loc[k, 'explicitly_set'] = True

    # for each lineage, from species -> domain, test if we have an exact match
    # at the same level within the NCBI taxonomy as indexed by GTDB. If a match
    # exists, test if it is identical to the GTDB lineage up to that point.
    # If it is not, then store the lineage for rewrite
    auto_rename = set()
    for r in ltp_tax.itertuples():
        parts = r.lineage.split(';')

        # from species -> domain
        for lvl, name in list(enumerate(parts))[::-1]:
            key = (lvl, name)

            if key in ncbi_to_gtdb:
                current = ';'.join(parts[:lvl+1])
                gtdb = ncbi_to_gtdb[key]

                # anchor the name just to be safe
                if lvl < 6:
                    current += ';'
                    gtdb += ';'

                if gtdb != current:
                    # gross gross sanity check
                    if 'Bacteria' in current and 'Archaea' in gtdb:
                        raise ValueError()
                    auto_rename.add((current, gtdb))

    # map these into the structore of the existing rewrite rules
    auto_rename = [{'old': current,
                    'new': gtdb,
                    'length': current.count(';')}
                    for current, gtdb in auto_rename]

    # we want to apply from longest to shortest so the rewrites do not
    # step on each others toes
    auto_rename = sorted(auto_rename, key=lambda x: x['length'], reverse=True)

    # write out for manual review if needed
    f = open('rewrite_rules', 'w')
    for x in auto_rename:
        f.write(repr(x))
        f.write('\n')
    f.close()

    # bulk rename entries
    for entry in auto_rename + LTP_RENAMES:
        def renamer(lin):
            if entry['old'] in lin:
                lin = lin.replace(entry['old'], entry['new'])
            return lin

        ltp_tax['lineage'] = ltp_tax['lineage'].apply(renamer)

    # finally drop any records we indicate we cannot handle
    ltp_tax = ltp_tax.loc[set(ltp_tax.index) - LTP_TO_DROP]

    ltp_tax.reset_index(inplace=True)
    return ltp_tax


def check_overlap(gtdb_tax, ltp_tax):
    # GTDB uses these names at multiple levels and differentiates by
    # rank designation
    permissible = {'UBA8346', 'AKS1', 'JAAYUW01', 'DSM-19610', 'UBA10575',
                   'DSM-17781', 'SK-Y3', 'DSM-22653', 'JC228', 'DSM-100275',
                   'DSM-16500', 'HP12', 'DSM-26407', 'DY22613', 'UBA6429'}
    # test if any names on overlap
    for i in LEVELS:
        for j in LEVELS:
            if i == j:
                continue

            a = set(gtdb_tax[i]) - set([""])
            b = set(gtdb_tax[j]) - set([""])
            if (a & b).issubset(permissible):
                continue

            if len(a & b):
                print("gtdb conflict %s %s" % (i, j))
                print(a & b)
                raise ValueError()

            a = set(ltp_tax[i]) - set([""])
            b = set(ltp_tax[j]) - set([""])
            if (a & b).issubset(permissible):
                continue

            if len(a & b):
                print("ltp conflict %s %s" % (i, j))
                print(a & b)
                print(ltp_tax[ltp_tax[i].isin(a & b)][['id', 'phylum','class','order']])
                print(ltp_tax[ltp_tax[j].isin(a & b)][['id', 'phylum','class','order']])
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
            if len(name) == 3 and name.endswith('__'):
                continue

            if len(grp[LEVELS[idx-1]].unique()) != 1:
                print('gtdb', i, name, list(grp[LEVELS[idx-1]].unique()))
                raise ValueError()
        for name, grp in ltp_tax.groupby(i):
            if name == '':
                continue
            if len(grp[LEVELS[idx-1]].unique()) != 1:
                print('ltp', i, name, list(grp[LEVELS[idx-1]].unique()))
                raise ValueError()


def format_name(level, name):
    return "%s__%s" % (level[0], name)


def prep_trees(gtdb_tree, ltp_tree):
    # decorate various flags on to the ltp tree
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

def clean_tree(tree):
    for node in tree.traverse(include_self=True):
        if hasattr(node, 'ChildLookup'):
            delattr(node, 'ChildLookup')


def graft_from_other(other, lookup):
    # in postorder, attempt to find the corresponding node
    # in gtdb.
    # graft the ungrafted portion of the subtree onto gtdb.
    # remove the subtree from ltp
    for node in list(other.postorder(include_self=False)):
        if node.is_tip():
            continue

        if node.name in lookup:
            gtdb_node = lookup[node.name]

            node.parent.remove(node)
            node = node.copy()
            for desc in node.postorder(include_self=True):
                if not desc.is_tip():
                    desc.keepable = any([c.keepable for c in desc.children])

            for desc in list(node.postorder(include_self=False)):
                if not desc.keepable:
                    desc.parent.remove(desc)

            if node.children:
                gtdb_node.extend(node.children)


def gtdb_unique_species_mappings(ar_fp, bt_fp):
    ar = pd.read_csv(ar_fp, sep='\t', dtype=str)
    ba = pd.read_csv(bt_fp, sep='\t', dtype=str)
    gtdb_metadata = pd.concat([ar, ba])
    gtdb_metadata['ncbi_species'] = [r.ncbi_taxonomy.split(';')[-1].strip().split('__', 1)[1]
                                     for r in gtdb_metadata.itertuples()]
    gtdb_mappings = defaultdict(set)
    for r in gtdb_metadata.itertuples():
        gtdb_taxonomy = r.gtdb_taxonomy.split(';')
        ltp_style = []
        for n in gtdb_taxonomy:
            ltp_style.append(n.split('__', 1)[1].strip())  # remove rank specification
        gtdb_mappings[r.ncbi_species].add(';'.join(ltp_style))

    # only keep mappings if they are unambiguous
    return {k: list(v)[0] for k, v in gtdb_mappings.items() if len(v) == 1}


def ncbi_to_gtdb_mappings(ar_fp, bt_fp):
    ar = pd.read_csv(ar_fp, sep='\t', dtype=str)
    ba = pd.read_csv(bt_fp, sep='\t', dtype=str)
    gtdb_metadata = pd.concat([ar, ba])
    mappings = {}

    # let's be conservative as we're assessing all over the gtdb taxonomy
    # and not examining specific records. as such let's limit to more trusted
    # and presumably stable records
    # TODO: should be use "gtdb_type_designation" != "not type material" ?
    gtdb_metadata = gtdb_metadata[(gtdb_metadata['gtdb_representative'] == 't') &
                                  (gtdb_metadata['ncbi_type_material_designation'] != 'none')]

    # some names are unexpectedly renamed, even when restricting to representative
    # and type material. An example is "Acinetobacter lwoffii" ->
    # "Acinetobacter fasciculus". To protect from this, we will limit automated
    # remapping at the species level to situations where the species name
    # remains consistent
    drop = set()
    for r in gtdb_metadata.itertuples():
        ncbi = [n.split('__', 1)[1] for n in r.ncbi_taxonomy.split(';')]
        gtdb = [n.split('__', 1)[1] for n in r.gtdb_taxonomy.split(';')]
        for lvl, name in enumerate(ncbi):
            if len(name) == 0:
                continue

            if lvl == 6:
                # verify the species names remain consistent
                if ncbi[-1] != gtdb[-1]:
                    continue

            # some names like eubacteriales are split into multiple orders,
            # so a one-to-one mapping does not exist. if we detect a conflict,
            # then we remove that as a viable rewrite mapping.
            # if there is variation in what we remap, the ncbi name has
            # been split and we cannot construct a 1-1 mapping.
            remap = ';'.join(gtdb[:lvl+1])
            if mappings.get((lvl, name), remap) != remap:
                drop.add((lvl, name))
            else:
                mappings[(lvl, name)] = remap

    for k in drop:
        del mappings[k]

    return mappings


@click.command()
@click.option('--tree', type=click.Path(exists=True), required=True,
              help='The backbone tree')
@click.option('--gtdb', type=click.Path(exists=True), required=True,
              help='The GTDB taxonomy with a subset of IDs mapping into the '
                   'backbone')
@click.option('--ltp', type=click.Path(exists=True), required=True,
              help='The LTP taxonomy with a subset of IDs mapping into the '
                   'backbone')
@click.option('--gtdb-archaea', type=click.Path(exists=True), required=True,
              help='The GTDB archaea metadata')
@click.option('--gtdb-bacteria', type=click.Path(exists=True), required=True,
              help='The GTDB bacteria metadata')
@click.option('--output', type=click.Path(exists=False))
def harmonize(tree, gtdb, ltp, output, gtdb_archaea, gtdb_bacteria):
    tree = bp.to_skbio_treenode(bp.parse_newick(open(tree).read()))
    gtdb_tax = pd.read_csv(gtdb, sep='\t', names=['id', 'lineage'])
    ltp_tax = pd.read_csv(ltp, sep='\t', names=['id', 'original_species',
                                                'lineage', 'u0', 'type',
                                                'u1', 'u2'])
    gtdb_unique = gtdb_unique_species_mappings(gtdb_archaea, gtdb_bacteria)
    ncbi_to_gtdb = ncbi_to_gtdb_mappings(gtdb_archaea, gtdb_bacteria)

    tree_tips = {n.name for n in tree.tips()}

    ltp_tax = adjust_ltp(ltp_tax, gtdb_unique, ncbi_to_gtdb)

    ltp_tax = ltp_tax[ltp_tax['id'].isin(tree_tips)]

    parse_lineage(gtdb_tax)
    parse_lineage(ltp_tax)

    check_species_labels(ltp_tax)
    check_overlap(gtdb_tax, ltp_tax)

    poly = get_polyphyletic(gtdb_tax)
    poly_ids_in_ltp = ids_of_ambiguity_from_polyphyletic(ltp_tax, poly)

    # deal with writing out LTP we cannot map in a structured format
    ltp_tax_to_write = ltp_tax.copy()
    ltp_tax_to_write = ltp_tax_to_write[ltp_tax_to_write['id'].isin(poly_ids_in_ltp)]
    ltp_tax_to_write_tree = skbio.TreeNode.from_taxonomy([(r['id'],
                                                           [r[n] for n in LEVELS])
                                                          for i, r in ltp_tax_to_write.iterrows()])
    ltp_tax_to_write_tree.Rank = -1  # bridge node
    for n in ltp_tax_to_write_tree.preorder(include_self=False):
        if n.is_tip():
            continue
        n.Rank = n.parent.Rank + 1
    prep_trees(skbio.TreeNode(), ltp_tax_to_write_tree)
    result = pull_consensus_strings(ltp_tax_to_write_tree, append_prefix=False)

    # these are the LTP records which were not integrated with GTDB
    f = open(output + '.removed_ltp_taxa', 'w')
    f.write('\n'.join(result))
    f.write('\n')
    f.close()

    # these are the LTP records which were integrated
    ltp_tax = ltp_tax[~ltp_tax['id'].isin(ltp_tax_to_write['id'])]

    # we check for consistent parents AFTER removal of unmappable
    # records from the polyphyly. as we explicitly map some names
    # from ltp, we expect labels like Firmicutes_A to now exist in
    # ltp, and we want to keep that. However, we will only have
    # a valid taxonomy once the records which remain ambiguous
    # are removed
    check_consistent_parents(gtdb_tax, ltp_tax)

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

    graft_from_other(ltp_tree, lookup)
    missing = [n.name for n in ltp_tree.tips()][:5]
    print(missing)

    result = pull_consensus_strings(gtdb_tree, append_prefix=False)
    f = open(output, 'w')
    f.write('\n'.join(result))
    f.write('\n')
    f.close()
    result_df = pd.read_csv(output, sep='\t', names=['id', 'Taxon'])
    result_df = result_df[result_df['id'].isin(tree_tips)]
    result_df.to_csv(output + '.treeoverlap', sep='\t', index=False, header=False)


if __name__ == '__main__':
    harmonize()
