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
    obs = set()
    for level in LEVELS:
        matches = poly[level]
        hits = df[level].isin(matches)
        ids = df[hits]['id']
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


def adjust_ltp(ltp_tax):
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
    incertae_sedis_mappings = {
        # https://gtdb.ecogenomic.org/genome?gid=GCA_003447605.1
        'X60418': 'd__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__Plesiomonas; s__Plesiomonas shigelloides',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_011207455.1
        'KY471008': 'd__Bacteria; p__Desulfobacterota; c__Thermodesulfobacteria; o__Thermodesulfobacteriales; f__ST65; g__Thermosulfuriphilus; s__Thermosulfuriphilus ammonigenes',

        # same genus, not the same species, but placing within the genus
        # per gtdb. note that the genus is polyphyletic so we are relying
        # on tax2tree to resolve it
        # https://gtdb.ecogenomic.org/genome?gid=GCA_000438555.1
        'AM072763': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium; s__Exiguobacterium artemiae',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000019905.1
        'CP001022': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A sibiricum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000620805.1
        'DQ019165': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A undae',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000702625.1
        'AB105164': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A oxidotolerans',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000299435.1
        'DQ019164': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A antarcticum',

        # note, appears renamed in GTDB from soli -> antarcticum
        # https://gtdb.ecogenomic.org/genome?gid=GCF_017874675.1
        'AY864633': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A antarcticum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000702605.1
        'DQ019167': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A acetylicum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_001939065.1
        'AJ846291': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A indicum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_001456895.1
        'JF893462': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A indicum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000714435.1
        'EU379016': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium; s__Exiguobacterium alkaliphilum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_001766415.1
        'AB680657': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium; s__Exiguobacterium aurantiacum_A',

        # species isn't in GTDB, but genus is, placing under the genus
        # and lets let tax2tree sort out polyphyletic
        'JF775503': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium; s__Exiguobacterium aquaticum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_005960665.1
        'AM072764': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium; s__Exiguobacterium mexicanum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000620845.1
        'AY594266': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium; s__Exiguobacterium marinum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_001909285.1
        'AY818050': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium; s__Exiguobacterium profundum_A',

        # species isnt' in GTDB, but genus is, placing under the genus
        'AY594264': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium; s__Exiguobacterium aestuarii',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_003344535.1
        'MH375463': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Exiguobacterales; f__Exiguobacteraceae; g__Exiguobacterium_A; s__Exiguobacterium_A flavidum',

        # gemella does not appear polyphyletic. species isnt in gtdb, placing under the genus
        # https://gtdb.ecogenomic.org/searches?s=al&q=gemella
        'HM103931': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella parahaemolysans',

        # gemella does not appear polyphyletic. species isnt in gtdb, placing under the genus
        # https://gtdb.ecogenomic.org/searches?s=al&q=gemella
        'HM103934': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella taiwanensis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_901873445.1
        'L14326': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella haemolysans_C',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_900476045.1
        'L14327': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella morbillorum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000701685.1
        'Y13364': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella sanguinis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000469465.1
        'Y13365': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella bergeri',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000425665.1
        'AJ251987': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella cuniculi',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_001553005.1
        'EU427463': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella asaccharolytica',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_013415235.1
        'Y17280': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Staphylococcales; f__Gemellaceae; g__Gemella; s__Gemella palaticanis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000510645.1
        'AJ242495': 'd__Bacteria; p__Firmicutes; c__Bacilli; o__Thermicanales; f__Thermicanaceae; g__Thermicanus; s__Thermicanus aegyptius',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_004345705.1
        'KX822012': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Thermotaleaceae; g__Marinisporobacter; s__Marinisporobacter balticus',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_002998925.1
        'AB037874': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Anaerovoracaceae; g__Mogibacterium; s__Mogibacterium diversum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_002243385.1
        'CP016199': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Anaerovoracaceae; g__Mogibacterium; s__Mogibacterium pumilum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000525775.1
        'Z36296': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Anaerovoracaceae; g__Mogibacterium; s__Mogibacterium timidum',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_010669305.1
        'AB298771': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Anaerovoracaceae; g__Aminipila; s__Aminipila butyrica',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000426305.1
        'AJ251215': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Anaerovoracaceae; g__Anaerovorax; s__Anaerovorax odorimutans',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_018390735.1
        'AF050099': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Acidaminobacteraceae; g__Fusibacter; s__Fusibacter paucivorans',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_016908355.1
        'FR851323': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Acidaminobacteraceae; g__Fusibacter_C; s__Fusibacter_C tunisiensis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_900103005.1
        'AF016691': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Peptostreptococcales; f__Acidaminobacteraceae; g__Acidaminobacter; s__Acidaminobacter hydrogenoformans',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000312505.2
        'HM587321': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Tissierellales; f__Peptoniphilaceae; g__Fenollaria; s__Fenollaria massiliensis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000311985.1
        'JN837487': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Tissierellales; f__Peptoniphilaceae; g__Kallipyga; s__Kallipyga massiliensis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_900130025.1
        'AF358114': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Tissierellales; f__Sporanaerobacteraceae; g__Sporanaerobacter; s__Sporanaerobacter acetigenes',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_013403245.1
        'L11305': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Tissierellales; f__Sedimentibacteraceae; g__Sedimentibacter; s__Sedimentibacter hydroxybenzoicus',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_007830175.1
        'AJ404680': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Tissierellales; f__Sedimentibacteraceae; g__Sedimentibacter; s__Sedimentibacter saalensis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_017873955.1
        'AB598276': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Tissierellales; f__Sedimentibacteraceae; g__Sedimentibacter; s__Sedimentibacter acidaminivorans',

        # https://gtdb.ecogenomic.org/searches?s=al&q=%22Anaerobranca%22
        # species is not present but genus appears consistent
        'U21809': 'd__Bacteria; p__Firmicutes_D; c__Proteinivoracia; o__Proteinivoracales; f__Proteinivoraceae; g__Anaerobranca; s__Anaerobranca horikoshii',

        # https://gtdb.ecogenomic.org/searches?s=al&q=%22Anaerobranca%22
        # species is not present but genus appears consistent
        'EF190921': 'd__Bacteria; p__Firmicutes_D; c__Proteinivoracia; o__Proteinivoracales; f__Proteinivoraceae; g__Anaerobranca; s__Anaerobranca zavarzinii',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_900142275.1
        'AY064218': 'd__Bacteria; p__Firmicutes_D; c__Proteinivoracia; o__Proteinivoracales; f__Proteinivoraceae; g__Anaerobranca; s__Anaerobranca californiensis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_900111575.1
        'AF203703': 'd__Bacteria; p__Firmicutes_D; c__Proteinivoracia; o__Proteinivoracales; f__Proteinivoraceae; g__Anaerobranca; s__Anaerobranca gottschalkii',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_900167165.1
        # note that GTDB reclassified the type strain
        'AY673988': 'd__Bacteria; p__Firmicutes_B; c__GCA-003054495; o__Carboxydocellales; f__Carboxydocellaceae; g__Carboxydocella; s__Carboxydocella thermautotrophica',

        # https://gtdb.ecogenomic.org/genome?gid=GCA_003054495.1
        'AY061974': 'd__Bacteria; p__Firmicutes_B; c__GCA-003054495; o__Carboxydocellales; f__Carboxydocellaceae; g__Carboxydocella; s__Carboxydocella thermautotrophica',

        # https://gtdb.ecogenomic.org/searches?s=al&q=%22Carboxydocella%22
        # species is not present but genus appears consistent
        'GU584133': 'd__Bacteria; p__Firmicutes_B; c__GCA-003054495; o__Carboxydocellales; f__Carboxydocellaceae; g__Carboxydocella; s__Carboxydocella manganica',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000183545.2
        'AF343566': 'd__Bacteria; p__Firmicutes_E; c__Thermaerobacteria; o__Thermaerobacterales; f__Thermaerobacteraceae; g__Thermaerobacter; s__Thermaerobacter subterraneus',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000184705.1
        'CP002344': 'd__Bacteria; p__Firmicutes_E; c__Thermaerobacteria; o__Thermaerobacterales; f__Thermaerobacteraceae; g__Thermaerobacter; s__Thermaerobacter marianensis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000213255.1
        'CP002360': 'd__Bacteria; p__Firmicutes_A; c__Clostridia; o__Mahellales; f__Mahellaceae; g__Mahella; s__Mahella australiensis',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_003991135.1
        'KC794015': 'd__Bacteria; p__Firmicutes_F; c__Halanaerobiia; o__DY22613; f__DY22613; g__Anoxybacter; s__Anoxybacter fermentans',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_000946815.1
        'EU386162': 'd__Bacteria; p__Firmicutes_B; c__Moorellia; o__Thermacetogeniales; f__Thermacetogeniaceae; g__Syntrophaceticus; s__Syntrophaceticus schinkii',

        # https://gtdb.ecogenomic.org/genome?gid=GCF_013664685.1
        'MN602556': 'd__Bacteria; p__Firmicutes_G; c__UBA4882; o__UBA10575; f__UBA10575; g__Capillibacterium; s__Capillibacterium thermochitinicola',

        # https://gtdb.ecogenomic.org/searches?s=al&q=%22Exilispira%22
        # species is not present but genus appears consistent
        'AB364473': 'd__Bacteria; p__Spirochaetota; c__JAAYUW01; o__JAAYUW01; f__JAAYUW01; g__Exilispira; s__Exilispira thermophila',
	}

    ltp_to_drop = {
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
    }

    for k, v in list(incertae_sedis_mappings.items()):
        v = v.strip()
        newv = []
        for name in v.split('; '):
            # remove prefix to resemble LTP structure
            newv.append(name.split('__', 1)[0])
        incertae_sedis_mappings[k] = ';'.join(newv)

    # remove extraneous quotes
    ltp_tax['lineage'] = ltp_tax['lineage'].apply(lambda x: x.replace('"', ''))
    ltp_tax['original_species'] = ltp_tax['original_species'].apply(lambda x: x.replace('"', ''))

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

        if 'Armatimonadetes' in row['lineage']:
            row['lineage'] = row['lineage'].replace('Armatimonadetes',
                                                    'Armatimonadota')

        if 'Pseudomonadota' in row['lineage']:
            row['lineage'] = row['lineage'].replace('Pseudomonadota',
                                                    'Proteobacteria')

        if 'ThermodesulfobacteriotaThermodesulfobacteria' in row['lineage']:
            # typo
            row['lineage'] = row['lineage'].replace('ThermodesulfobacteriotaThermodesulfobacteria',
                                                    'Thermodesulfobacteriota;Thermodesulfobacteria')
        if 'Thermodesulfobacteriota' in row['lineage']:
            # see https://gtdb.ecogenomic.org/genome?gid=GCA_001508095.1
            row['lineage'] = row['lineage'].replace('Thermodesulfobacteriota',
                                                    'Desulfobacterota')

        if 'Hydrogenophilalia' in row['lineage']:
            # see https://gtdb.ecogenomic.org/genome?gid=GCA_001802655.1
            row['lineage'] = row['lineage'].replace('Hydrogenophilalia',
                                                    'Gammaproteobacteria')

        if 'Lentisphaerota' in row['lineage']:
            # see https://gtdb.ecogenomic.org/genome?gid=GCF_000170755.1
            row['lineage'] = row['lineage'].replace('Lentisphaerota',
                                                    'Verrucomicrobiota')
        if 'Kiritimatiellota' in row['lineage']:
            # see https://gtdb.ecogenomic.org/genome?gid=GCA_902779565.1
            row['lineage'] = row['lineage'].replace('Kiritimatiellota',
                                                    'Verrucomicrobiota')
        if 'Bacillota' in row['lineage']:
            row['lineage'] = row['lineage'].replace('Bacillota', 'Firmicutes')

        if 'Actinomycetotaa' in row['lineage']:
            # typo
            row['lineage'] = row['lineage'].replace('Actinomycetotaa',
                                                    'Actinomycetota')

        if 'ActinomycetotaAcidimicrobiia' in row['lineage']:
            # typo
            row['lineage'] = row['lineage'].replace('ActinomycetotaAcidimicrobiia',
                                                    'Actinomycetota;Acidimicrobiia')

        if 'Bdellovibrionota;Bdellovibrionota' in row['lineage']:
            row['lineage'] = row['lineage'].replace('Bdellovibrionota;Bdellovibrionota',
                                                    'Bdellovibrionota')

        if 'Nannocystale;Nannocystaceae' in row['lineage']:
            row['lineage'] = row['lineage'].replace('Nannocystale;Nannocystaceae',
                                                    'Nannocystales;Nannocystaceae')

        if 'Actinobacteria;Micrococcales' in row['lineage']:
            # see https://gtdb.ecogenomic.org/genome?gid=GCF_001423565.1
            row['lineage'] = row['lineage'].replace('Actinobacteria;Micrococcales',
                                                    'Actinobacteria;Actinomycetales')

        if 'Actinomycetota' in row['lineage']:
            # see https://gtdb.ecogenomic.org/genome?gid=GCF_001423565.1
            row['lineage'] = row['lineage'].replace('Actinomycetota',
                                                    'Actinobacteriota')

        if 'Intrasporangiaceae' in row['lineage']:
            # see https://gtdb.ecogenomic.org/genome?gid=GCA_001598955.1
            row['lineage'] = row['lineage'].replace('Intrasporangiaceae',
                                                    'Dermatophilaceae')

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

        if 'incertae_sedis' in row['lineage']:
            parts = row['lineage'].split(';')
            keep = []
            for p in parts:
                if 'incertae_sedis' not in p:
                    keep.append(p)
                else:
                    break

            keep.extend(['' for i in range((len(LEVELS) - len(keep)) - 1)])
            row['lineage'] = ';'.join(keep)

        if 'ThermodesulfobacteriotaThermodesulfobacteria' in row['lineage']:
            row['lineage'] = row['lineage'].replace('ThermodesulfobacteriotaThermodesulfobacteria',
                                                    'Thermodesulfobacteriota;Thermodesulfobacteria')

        if 'Betaproteobacteria' in row['lineage']:
            # https://forum.gtdb.ecogenomic.org/t/what-happend-with-betaproteobacteria/131/2
            row['lineage'] = row['lineage'].replace('Betaproteobacteria', 'Gammaproteobacteria')

        if ' subsp. ' in row['original_species']:
            # we can't handle subspecies so let's remove
            row['original_species'] = row['original_species'].split(' subsp. ')[0]

        if 'Schaalia' in row['lineage']:
            # see https://gtdb.ecogenomic.org/searches?s=al&q=g__schaalia
            row['lineage'] = row['lineage'].replace('Schaalia', 'Pauljensenia')
            row['original_species'] = row['original_species'].replace('Schaalia', 'Pauljensenia')

    # the ltp taxonomy doesn't have species in the lineages, so add it in
    def append_species(row):
        return row['lineage'] + ';' + row['original_species']
    ltp_tax['lineage'] = ltp_tax.apply(append_species, axis=1)

    ltp_tax.set_index('id', inplace=True)

    # replace lineages for incertae sedis
    for k, v in incertae_sedis_mappings.items():
        ltp_tax.loc[k, 'lineage'] = v

    # drop incertae sedis we cannot handle
    ltp_tax = ltp_tax.loc[set(ltp_tax.index) - ltp_to_drop]

    ltp_tax.reset_index(inplace=True)
    return ltp_tax


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
    for node in ltp_tree.traverse(include_self=False):
        node.keepable = False
        if node.is_tip():
            node.keepable = True
        else:
            print(node.name, node.Rank, node.parent.name)
            print(repr(node.children))
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


@click.command()
@click.option('--tree', type=click.Path(exists=True), required=True,
              help='The backbone tree')
@click.option('--gtdb', type=click.Path(exists=True), required=True,
              help='The GTDB taxonomy with a subset of IDs mapping into the '
                   'backbone')
@click.option('--ltp', type=click.Path(exists=True), required=True,
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

    ltp_tax = adjust_ltp(ltp_tax)

    #gtdb_tax = gtdb_tax[gtdb_tax['id'].isin(tree_tips)]
    ltp_tax = ltp_tax[ltp_tax['id'].isin(tree_tips)]

    parse_lineage(gtdb_tax)
    parse_lineage(ltp_tax)

    check_species_labels(ltp_tax)
    check_overlap(gtdb_tax, ltp_tax)
    check_consistent_parents(gtdb_tax, ltp_tax)

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
