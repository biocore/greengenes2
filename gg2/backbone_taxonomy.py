#!/usr/bin/env python
"""Integrate GTDB and LTP taxonomies

This script is difficult. It's primary responsiblity is to integrate two
different taxonomic systems. It's seconadary goal is to emit reasonable
NCBI tax IDs for the entries. The complexity stems from the following:

    * There are numerous error modes in input dataset, many of which
        may derive from human error as they are largely not systematic.
        As a result, there is a substantial amount of code dedicated
        to special casing error handling.
    * LTP and GTDB taxonomies are fundamentally different. We attempt to
        bridge the taxonomies in multiple ways. First, we use NCBI mappings
        provided by GTDB, for its representatives and reported NCBI type
        material, to infer lineages relative to GTDB for many species. This
        allows for many species names to be directly mapped from LTP into
        GTDB. Second, for species, genera, etc which are in LTP but not in
        not in GTDB (e.g., doesn't have a sequenced genome), we construct
        rewrite rules from species up, based on the NCBI / GTDB bridge
        we construct from the GTDB metadata. Following rewrites, we graft
        the LTP taxonomic entries to the GTDB hierarchy, allowing us to
        express novel lineages within GTDB.
    * GTDB does not always use the NCBI name, in some cases it falls on
        a basionym or homotypic synonym, this resulted in manually assessing
        many records.
    * WOL uses GTDB, but not all entries in WOL are in GTDB, and a handful
        of cases, the WOL entries are remarked as historical by Genbank/RefSeq

Because of the special case nature of many of the error modes, much of the code
is exceedingly difficult to test. To mitigate the introduction of errors, the
code attempts to be defensive where feasible, and testing edge cases at runtime.

This script can absolutely be refactored. It is spaghetti due to the ad hoc
discovery of many of the error modes, and the repeated revisions to error
resolution.
"""

import click
import bp
import pandas as pd
from t2t.nlevel import make_consensus_tree, pull_consensus_strings
from functools import partial
import skbio
import re
from collections import defaultdict
import ete3


# We either could not directly obtain an NCBI tax ID from GTDB for these records
# or the name was not completely resolvable
# {id: (old name, new name, tax_id)}
MANUAL_GTDB_TAXID_ASSESSMENT = {
    'G000420225': ('Mycoplasma moatsii', 'Mesomycoplasma moatsii', 171287),
    'G001007625': ('Synechococcus spongiarum', 'Synechococcus spongiarum', 431041),
    'G008634845': ('Ordinivivax streblomastigis', 'Ordinivivax streblomastigis', 2540710),
    'G003995005': ('Halichondribacter symbioticus', 'Halichondribacter symbioticus', 2494554),
    'G000315095': ('Kuenenia stuttgartiensis', 'Kuenenia stuttgartiensis', 174633),
    'G900660435': ('Mycoplasma orale', 'Metamycoplasma orale', 2121),
    'G001684175': ('Glomeribacter gigasporarum', 'Glomeribacter gigasporarum', 132144),
    'G001180145': ('Pelagibacter ubique', 'Pelagibacter ubique', 198252),
    'G000375745': ('Nitromaritima sp. AB-629-B06', 'Nitromaritima sp. AB-629-B06', 1131270),
    'G000375765': ('Nitromaritima sp. AB-629-B18', 'Nitromaritima sp. AB-629-B18', 1131269),
    'G001584765': ('Mycobacterium simiae', 'Mycobacterium simiae', 1784),
	'G900660435': ('Mycoplasma orale', 'Metamycoplasma orale', 2121),
	'G000315095': ('Kuenenia stuttgartiensis', 'Kuenenia stuttgartiensis', 174633),
}

# the gtdb failures file doesn't provide as much information as the webpage
# TBH this could, and should, be wrapped into the MANUAL_GTDB_TAXID_ASSESSMENT
# object...
MANUAL_GTDB_FAILURES = {
    'G000252705': ('Arthromitus sp. SFB-3', 1054944),
    'G000341755': ('Flavobacterium sp. MS220-5C', 926447),
    'G000409605': ('Microbacterium sp. AO20a11', 1201038),
    'G000463735': ('Chitinophaga sp. JGI 0001002-D04', 1235985),
    'G000483005': ('Variovorax sp. JGI 0001016-M12', 1286832),
    'G000800935': ('Methylobacterium sp. ZNC0032', 1339244),
    'G000955975': ('Saccharothrix sp. ST-888', 1427391),
    'G000980745': ('Candidatus Caldipriscus sp. T1.1', 1643674),
    'G001044495': ('Candidatus Nitromaritima sp. SCGC AAA799-C22', 1628279),
    'G001044505': ('Candidatus Nitromaritima sp. SCGC AAA799-A02', 1628278),
    'G001292585': ('Candidatus Magnetomorum sp. HK-1', 1509431),
    'G001313045': ('Nocardioides sp. JCM 18999', 1301083),
    'G001799425': ('Desulfobacula sp. RIFOXYA12_FULL_46_16', 1797911),
    'G001829755': ('Sulfurimonas sp. RIFOXYB2_FULL_37_5', 1802257),
    'G001829775': ('Sulfurimonas sp. RIFOXYC2_FULL_36_7', 1802258),
    'G002010445': ('Dehalococcoides sp. JdFR-57', 1935050),
    'G003030645': ('Marinobacter sp. Z-F4-2', 2137199),
    'G004127505': ('Salmonella sp. 3DZ2-4SM', 2175006),
    'G004961165': ('Mesorhizobium sp.', 1871066),
    'G004961685': ('Mesorhizobium sp.', 1871066),
    'G004961775': ('Mesorhizobium sp.', 1871066),
    'G004962305': ('Mesorhizobium sp.', 1871066),
    'G004962345': ('Mesorhizobium sp.', 1871066),
    'G004962385': ('Mesorhizobium sp.', 1871066),
    'G004963055': ('Mesorhizobium sp.', 1871066),
    'G004963865': ('Mesorhizobium sp.', 1871066),
    'G004964325': ('Mesorhizobium sp.', 1871066),
    'G004964405': ('Mesorhizobium sp.', 1871066),
    'G004964655': ('Mesorhizobium sp.', 1871066),
    'G004964895': ('Mesorhizobium sp.', 1871066),
    'G004964905': ('Mesorhizobium sp.', 1871066),
    'G004965145': ('Mesorhizobium sp.', 1871066),
    'G004965155': ('Mesorhizobium sp.', 1871066),
    'G004965175': ('Mesorhizobium sp.', 1871066),
    'G004965225': ('Mesorhizobium sp.', 1871066),
    'G004965685': ('Mesorhizobium sp.', 1871066),
    'G005063255': ('Mesorhizobium sp.', 1871066),
    'G005063315': ('Mesorhizobium sp.', 1871066),
    'G005117065': ('Mesorhizobium sp.', 1871066),
    'G007581405': ('Salmonella enterica subsp. diarizonae', 59204),
    'G008273775': ('Escherichia coli', 562),
    'G008349105': ('Campylobacter jejuni', 197),
    'G012161415': ('Escherichia albertii', 208962),
    'G012357415': ('Escherichia coli', 562),
    'G901564725': ('Streptococcus pyogenes', 1314),
    'G902536535': ('uncultured Polaribacter sp.', 174711),
    'G902590875': ('uncultured Candidatus Actinomarina sp.', 1985113),
}


# We either could not directly obtain an NCBI tax ID for these records
# or the name was not completely resolvable
# {id: (old name, new name, tax_id)}
MANUAL_LTP_TAXID_ASSESSMENT = {
    'KJ606916': ('Yersinia rochesterensi', 'Yersinia rochesterensis', 1604335),
    'JN175340': ('Haemophilus piscium', 'Haemophilus piscium', 80746),  # Haemophilus piscium corrig. Snieszko et al. 1950
    'MK007076': ('Alteromonas fortis', 'Alteromonas fortis', 226),  # No tax ID, using genus tax id, https://lpsn.dsmz.de/species/alteromonas-fortis
    'MT180568': ('Halomonas bachuensi', 'Halomonas bachuensis', 2717286),
    'LR797591': ('Pseudomonas neuropathic', 'Pseudomonas neuropathica', 2730425),
    'CP002881': ('Pseudomonas stutzeri', 'Pseudomonas stutzeri', 316),   # Stutzerimonas stutzeri per NCBI (taxid 316), however GTDB and LPSN still use Pseudomonas stutzeri
    'CP000267': ('Albidiferax ferrireducens', 'Rhodoferax ferrireducens', 192843),
    'AB021399': ('Pseudomonas cissicola', 'Xanthomonas cissicola', 86186),
    'AB021404': ('Stenotrophomonas geniculata', 'Stenotrophomonas geniculata', 86188),
    'HE860713': ('Ruegeria litorea', 'Ruegeria litorea', 1280831),   # Renamed, but GTDB uses Ruegeria litorea
    'LN810645': ('Cereibacter alkalitolerans', 'Luteovulum alkalitolerans', 1608950),   # Not found anywhere. Genbank uses "[Luteovulum] alkalitolerans" and is supported by LPSN
    'KY310591': ('Altericroceibacterium endophyticus', 'Altericroceibacterium endophyticum', 1808508),
    'KX129901': ('Allorhizobium oryziradicis', 'Allorhizobium oryziradicis', 1867956),
    'EF440185': ('Rhizobium selenitireducens', 'Allorhizobium selenitireducens', 448181),   # basionym with Ciceribacter selenitireducens, however GTDB appears to use Allorhizobium for genus
    'MH355952': ('Komagataeibacter pomaceti', 'Komagataeibacter cocois', 1747507),   # basionym with Novacetimonas, GTDB appears to have renamed type material, https://gtdb.ecogenomic.org/genome?gid=GCF_003207955.1
    'X75620': ('Komagataeibacter hansenii', 'Komagataeibacter hansenii', 436),   # basionym with Acetobacter hansenii, GTDB uses Komagataeibacter hansenii, https://gtdb.ecogenomic.org/genome?gid=GCF_003207935.1
    'L14624': ('Arcobacter cryaerophilus', 'Aliarcobacter cryaerophilus', 28198),
    'AF418179': ('Nitratidesulfovibrio vulgaris', 'Nitratidesulfovibrio vulgaris', 881),
    'DQ296030': ('Humidesulfovibrio arcti', 'Humidesulfovibrio arcticus', 360296),
    'CP001087': ('Desulfobacterium autotrophicum', 'Desulfobacterium autotrophicum', 2296),
    'U96917': ('Geomonas bremense', 'Citrifermentans bremense', 60035),   # basionym is Geobacter bremensis (https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=60035&lvl=3&lin=f&keep=1&srchmode=1&unlock)
    'DQ991964': ('Seleniibacterium seleniigenes', 'Seleniibacterium seleniigenes', 407188),   # NCBI uses Pelobacter seleniigenes, but GTDB uses Seleniibacterium seleniigenes
    'ADMV01000001': ('Streptococcus oralis oralis subsp. oralis', 'Streptococcus oralis subsp. oralis', 1891914),
    'KY019172': ('Staphylococcus ursi', 'Staphylococcus sp010365305', 1912064),   # Ursi in LPSN https://lpsn.dsmz.de/species/staphylococcus-ursi, GTDB maps this to Staphylococcus sp010365305 (https://gtdb.ecogenomic.org/genome?gid=GCF_010365305.1)
    'AB009936': ('Staphylococcus urealyticus', 'Staphylococcus ureilyticus', 94138),
    'AM747813': ('Brevibacterium frigoritolerans', 'Peribacillus frigoritolerans', 450367),   # basionym is Brevibacterium frigoritolerans
    'KX865139': ('Niallia endozanthoxylicus', 'Niallia endozanthoxylica', 2036016),
    'AB021189': ('Lederbergia lentus', 'Bacillus lentus', 1467),   # GTDB uses basionym Bacillus lentus (https://gtdb.ecogenomic.org/genome?gid=GCF_001591545.1)
    'HQ433466': ('Litchfieldia salsus', 'Litchfieldia salsa', 930152),
    'KF265350': ('Rossellomorea enclensis', 'Rossellomorea enclensis', 1402860),
    'MH454613': ('Bacillus salinus', 'HMF5848 sp003944835', 1409),   # GTDB uses the sp name (https://gtdb.ecogenomic.org/genome?gid=GCF_900094975.1")
    'KU201962': ('Paenibacillus cucumis', 'Paenibacillus cucumis', 1776858),   # Note this is a duplicated binomial (https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=1776858)
    'DQ504377': ('Salibacterium aidingensis', 'Salibacterium aidingense', 384933),

	# note genus change, consistent with other shouchella, see https://www.ncbi.nlm.nih.gov/nuccore/HM054473 for an example
    'HM054473': ('Shouchella hunanensis', 'Alkalihalobacillus hunanensis', 766894),
    'HM054474': ('Shouchella xiaoxiensis', 'Alkalihalobacillus xiaoxiensis', 766895),
    'HQ620634': ('Shouchella shacheensis', 'Alkalihalobacillus shacheensis', 1649580),
    'AY258614': ('Shouchella patagoniensis', 'Alkalihalobacillus patagoniensis', 228576),
    'CP019985': ('Shouchella clausii', 'Alkalihalobacillus clausii', 79880),
    'X76446': ('Shouchella gibsonii', 'Alkalihalobacillus gibsonii', 79881),

	# note genus change, consistent with other halalkalibacter, see https://www.ncbi.nlm.nih.gov/nuccore/AB086897 for an example
    'AB086897': ('Halalkalibacter krulwichiae', 'Alkalihalobacillus krulwichiae', 199441),
    'AB043851': ('Halalkalibacter wakoensis', 'Alkalihalobacillus wakoensis', 127891),
    'AB043858': ('Halalkalibacter akibai', 'Alkalihalobacillus akibai', 1411),
    'AJ606037': ('Halalkalibacter alkalisediminis', 'Alkalihalobacillus alkalisediminis', 935616),
    'LN610501': ('Halalkalibacter kiskunsagensis', 'Alkalihalobacillus kiskunsagensis', 1548599),
    'KR611043': ('Halalkalibacter oceani', 'Alkalihalobacillus oceani', 1653776),
    'KY612314': ('Halalkalibacter urbisdiaboli', 'Alkalihalobacillus urbisdiaboli', 1960589),
    'GQ292773': ('Halalkalibacter nanhaiisediminis', 'Alkalihalobacillus nanhaiisediminis', 688079),

	# note genus change, Basionym genus is Mycoplasma
    'FJ226577': ('Mycoplasmopsis gallopavoni', 'Mycoplasmopsis gallopavonis', 76629),
    'AF412976': ('Mycoplasmopsis cricetul', 'Mycoplasmopsis cricetuli', 171283),
    'M23940': ('Mycoplasmoides pirum', 'Mycoplasmoides pirum', 2122),
    'FJ655918': ('Mycoplasmoides alvi', 'Mycoplasmoides alvi', 78580),
    'AB680686': ('Mycoplasmoides gallisepticum', 'Mycoplasmoides gallisepticum', 2096),
    'AAGX01000019': ('Mycoplasmoides genitalium', 'Mycoplasmoides genitalium', 2097),
    'AF125878': ('Mycoplasmoides fastidiosum', 'Mycoplasmoides fastidiosum', 92758),

    'M23933': ('Acholeplasma modicum', 'Acholeplasma modicum', 2150),
    'AUAL01000024': ('Acholeplasma axanthum', 'Acholeplasma axanthum', 29552),

	# GTDB uses the homotypic synonym
    'ABOU02000049': ('Ruminococcus lactaris', 'Mediterraneibacter lactaris', 46228),
    'AAVP02000040': ('Ruminococcus torques', 'Mediterraneibacter torques', 33039),
    'X94967': ('Ruminococcus gnavus', 'Mediterraneibacter gnavus', 33038),
    'MG780325': ('Mycobacterium chelonae', 'Mycobacterium chelonae', 1774),
    'AY457067': ('Mycobacterium houstonense', 'Mycobacterium houstonense', 146021),
    'AY457084': ('Mycobacterium farcinogenes', 'Mycobacterium farcinogenes', 1802),

    'MK085088': ('Nocardioides lijunqinia', 'Nocardioides lijunqiniae', 2760832),
    'U30254': ('Clavibacter tessellarius', 'Clavibacter tessellarius', 31965),
    'U09761': ('Clavibacter insidiosus', 'Clavibacter insidiosus', 33014),
    'HE614873': ('Clavibacter nebraskensis', 'Clavibacter nebraskensis', 31963),
    'CP012573': ('Clavibacter capsici', 'Clavibacter capsici', 1874630),
    'AM849034': ('Clavibacter sepedonicus', 'Clavibacter sepedonicus', 31964),
    'AB012590': ('Zimmermannella alba', 'Pseudoclavibacter alba', 272241),   # https://gtdb.ecogenomic.org/genome?gid=GCF_001570905.1
    'MG200147': ('Geodermatophilus marinus', 'Geodermatophilus marinus', 1663241),
    'FN178463': ('Olsenella umbonata', 'Parafannyhessea umbonata', 604330),   # https://gtdb.ecogenomic.org/genome?gid=GCF_900105025.1
    'MT556637': ('Fortiea necridiiformans', 'Fortiea necridiiformans', 2741322),
    'MK300544': ('Nostoc neudorfense', 'Nostoc neudorfense', 1245825),
    'LC329340': ('Spongiibacterium fuscum', 'Coraliitalea coralii', 499064),   # https://www.ncbi.nlm.nih.gov/nuccore/LC329340
    'MN955413': ('Chryseobacterium caseinilyticu', 'Chryseobacterium caseinilyticum', 2771428),
    'GQ259742': ('Chryseobacterium yonginense', 'Kaistella yonginensis', 658267),   # basionym is Chryseobacterium yonginense, GTDB seems to use revised genus
    'MF983698': ('Kaistella flava', 'Kaistella flava', 2038776),
    'AB682225': ('Chryseobacterium palustre', 'Kaistella palustris', 493376),   # https://gtdb.ecogenomic.org/genome?gid=GCF_000422265.1
    'EU516352': ('Chryseobacterium solincola', 'Kaistella solincola', 510955),   # https://gtdb.ecogenomic.org/genome?gid=GCF_000812875.1
    'FJ713810': ('Chryseobacterium buanense', 'Soonwooa buanensis', 619805),   # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=619805&lvl=3&lin=f&keep=1&srchmode=1&unlock
    'MG768961': ('Pedobacter pollutisol', 'Pedobacter pollutisoli', 2707017),
    'MN381950': ('Pedobacter riviphilus', 'Pedobacter riviphilus', 2606215),
    'GQ339899': ('Pseudoflavitalea ginsenosidimutans', 'Pseudobacter ginsenosidimutans', 661488),   # https://gtdb.ecogenomic.org/genome?gid=GCF_007970185.1
    'AB278570': ('Chitinophaga terrae', 'Chitinophaga terrae', 408074),

	# note genus change, others consistent, see as an example https://gtdb.ecogenomic.org/genome?gid=GCF_014201705.1
    'HE582779': ('Borrelia spielmanii', 'Borreliella spielmanii', 88916),
    'D67023': ('Borrelia tanukii', 'Borreliella tanukii', 56146),
    'D67022': ('Borrelia turdi', 'Borreliella turdi', 57863),

    'EU580141': ('Treponema caldarium', 'Treponema caldarium', 215591),   # GTDB uses the homotypic synonym, https://gtdb.ecogenomic.org/genome?gid=GCF_000219725.1
    'FR733664': ('Treponema stenostreptum', 'Spirochaeta stenostrepta', 152),   # GTDB has the genus represented, https://gtdb.ecogenomic.org/searches?s=al&q=g__Spirochaeta; using binomial from genbank record https://www.ncbi.nlm.nih.gov/nuccore/FR733664
    'GU434243': ('Streptomyces albidus', 'Streptomyces albidus', 722709),
    'MN620389': ('Ramlibacter terrae', 'Ramlibacter terrae', 174951),   # https://lpsn.dsmz.de/species/ramlibacter-terrae; using genus tax id
    'MN620391': ('Ramlibacter montanisoli', 'Ramlibacter montanisoli', 174951),   # https://lpsn.dsmz.de/species/ramlibacter-montanisoli; using genus tax id
    'MN911321': ('Hymenobacter taeanensis', 'Hymenobacter taeanensis', 89966),   # https://lpsn.dsmz.de/species/hymenobacter-taeanensis; using genus tax id
    'MW811459': ('Jiella mangrovi', 'Jiella mangrovi', 1775688),   # https://lpsn.dsmz.de/species/jiella-mangrovi; using genus tax id
    'MZ041112': ('Phycicoccus avicenniae', 'Phycicoccus avicenniae', 367298),  # https://lpsn.dsmz.de/species/phycicoccus-avicenniae; using genus tax id
    'MW540506': ('Bifidobacterium choladohabitan', 'Bifidobacterium choladohabitans', 2750947),
    'CP077075': ('Pseudomonas xanthosomatis', 'Pseudomonas xanthosomae', 2842356),
    'CP077078': ('Pseudomonas azerbaijanorientalis', 'Pseudomonas azerbaijanoriens', 2842350),
    'JAHSTU000000000': ('Pseudomonas azerbaijanoccidentalis', 'Pseudomonas azerbaijanoccidens', 2842347),
    'MH043116': ('Hydrogeniiclostridium mannosilyticum', 'Hydrogeniiclostridium mannosilyticum', 2764322),
    'KR698371': ('Lysobacter humi', 'Lysobacter humi', 1720254),
    'EU560726': ('Micromonospora endophytica', 'Micromonospora endophytica', 515350),
    'KT962840': ('Nocardioides flavus', 'Nocardioides flavus', 2058780),
    'JN399173': ('Novosphingobium aquaticum', 'Novosphingobium aquaticum', 1117703),
    'AB244764': ('Sphingobacterium composti', 'Sphingobacterium composti', 363260),
    'AB245347': ('Sphingomonas ginsengisoli', 'Sphingomonas ginsengisoli', 363835),
	'KX987135': ('Brenneria populi subsp. brevivirga', 'Brenneria populi subsp. brevivirga', 2036006),
    'KJ632518': ('Brenneria populi subsp. populi', 'Brenneria populi subsp. populi', 1505588),
}

    # ETE3 missed these due to extra whitespace in the LTP input file
    #'Pseudomonas nicosulfuronedens': ('nan', nan),  # ETE3 missed this?
    #'Qipengyuania flava': ('nan', nan),  # ETE3 missed this?
    #'Cupidesulfovibrio oxamicus': ('nan', nan),  # ETE3 missed this?
    #'Streptacidiphilus fuscans': ('nan', nan),  # ETE3 missed?
    #'Dictyobacter formicarum': ('nan', nan),  # ETE3 missed?
    #'Croceivirga litoralis': ('nan', nan),  # ETE3 missed?
    #'Tenacibaculum finnmarkense': ('nan', nan),  # ETE3 missed?
    #'Corynebacterium zhongnanshanii': ('nan', nan),  # ETE3 missed?


def names_to_taxid(names):
    ncbi = ete3.NCBITaxa()

    name_to_id = ncbi.get_name_translator(names)

    # make sure we have a unique ID, and not a reused name
    for k, v in list(name_to_id.items()):
        if not k:
            print(v)
            raise ValueError()
        if len(v) > 1:
            print("multiple IDs: %s, %s" % (k, v))
            raise ValueError()
        else:
            name_to_id[k] = v[0]

    missing = set(names) - set(name_to_id)
    missing -= set(['', ])

    # for records missing a name, see if we can recover it by stripping off
    # a subspecies designation
    no_subsp = defaultdict(list)
    missing_names = set()
    for m in missing:
        if 'subsp.' in m:
            no_subsp[m.split(' subsp.')[0]].append(m)
        else:
            missing_names.add(m)

    # note that we cannot reliably just try the genus part of the binomoal
    # as in some cases the organism has changed genera -- I don't know
    # where to obtain those mappings so I think that needs to be done
    # manually

    no_subsp_to_ids = ncbi.get_name_translator(list(no_subsp))
    for k, v in list(no_subsp_to_ids.items()):
        if not k:
            print(v)
            raise ValueError()
        if len(v) > 1:
            print("multiple IDs: %s, %s" % (k, v))
        else:
            no_subsp_to_ids[k] = v[0]

    missing_no_subsp = set(no_subsp) - set(no_subsp_to_ids)

    for k in missing_no_subsp:
        missing_names.update(set(no_subsp[k]))

    for k, v in no_subsp_to_ids.items():
        name_to_id[k] = v

    # placeholder
    for m in missing_names:
        name_to_id[m] = None

    # construct a lookup for names modified (e.g., subspecies stripped)
    lookup = {}
    for k, original in no_subsp.items():
        for v in original:
            lookup[v] = k

    # verify we have names for everything
    for k, v in name_to_id.items():
        if v is None:
            print("Missing for: |%s|" % k)
    return name_to_id, lookup


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

    # looks like a bulk rename
    # see https://gtdb.ecogenomic.org/genome?gid=GCF_900108485.1
    {'old': 'Chromatiales;Chromatiaceae;Rheinheimera',
     'new': 'Enterobacterales;Alteromonadaceae;Rheinheimera'},

    # typo in HQ223108
    {'old': 'Actinomycetota;;Thermoleophilia',
     'new': 'Actinomycetota;Thermoleophilia'},

    # rewritten in gtdb
    {'old': 'Acidimicrobiales_incertae_sedis;Aciditerrimonas',
     'new': 'Acidimicrobiaceae;Aciditerrimonas'},

    # one of the records has a period after the genus
    {'old': 'Prolixibacteraceae;Aquipluma.',
     'new': 'Prolixibacteraceae;Aquipluma'},

    # family is incertae sedis in ltp but genus is in gtdb
    {'old': 'Pseudomonadales_incertae_sedis;Spartinivicinus',
     'new': 'Pseudomonadales;Zooshikellaceae'},

    # family is incertae sedis in ltp but genus is in gtdb
    {'old': 'Burkholderiales_incertae_sedis;Inhella',
     'new': 'Burkholderiaceae;Inhella'},


]

LTP_TO_DROP = {
    # family incertae sedis and genus is not in GTDB
    'KF981441',
    'JN699062',
    'MG709347',
    'AP023361',
    'AM117931',
    'MK503700',
    'AB180657',

    # family is incertae sedis, and genus is shown to split in GTDB
    'LN794224',
    'KF551137',
    'AB297965',
    'AB245356',
    'KM670026',

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

    # genus doesn't seem represented in GTDB, and its family is
    # incertae sedis in ncbi
    'KF981441',
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
    # https://gtdb.ecogenomic.org/genome?gid=GCF_003991585.1
    'KY039174': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Saezia;Saezia sanguinis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_004217215.1
    'AM774413': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Rivibacter;Rivibacter subsaxonicus',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_002116905.1
    'CP024645': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Rhizobacter;Rhizobacter gummiphilus',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_006386545.1
    'MF687442': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Pseudorivibacter;Pseudorivibacter rhizosphaerae',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_004011805.1
    'KU667249': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Piscinibacter;Piscinibacter defluvii',

    # https://gtdb.ecogenomic.org/genome?gid=GCA_003265685.1
    'KX390668': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Piscinibacter;Piscinibacter caeni',

    # https://gtdb.ecogenomic.org/genome?gid=GCA_007994965.1
    'AB681749': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Piscinibacter;Piscinibacter aquaticus',

    # https://gtdb.ecogenomic.org/genome?gid=GCA_017306535.1
    'AF176594': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Methylibium;Methylibium petroleiphilum',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_000969605.1
    'DQ656489': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Aquincola;Aquincola tertiaricarbonis',

    # https://gtdb.ecogenomic.org/genome?gid=GCF_004016505.1
    'LT594462': 'Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Burkholderiaceae;Rubrivivax;Rubrivivax rivuli',

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
    for idx, row in ltp_tax.iterrows():
        if row['id'] == 'KF303137':
            print(row['lineage'])
            assert row['lineage'].count(';') == 5
            # this record is missing its genus, need to do it here as we've already tacked on species
            ltp_tax.loc[idx, 'lineage'] = row['lineage'].replace('Flavobacteriaceae;Aestuariibaculum scopimerae',
                                                                 'Flavobacteriaceae;Aestuariibaculum;Aestuariibaculum scopimerae')

        if ';;' in row['lineage']:
            # one of the records in ltp has dual ;;
            ltp_tax.loc[idx, 'lineage'] = row['lineage'].replace(';;', ';')

        if row['lineage'].startswith(' Bacteria'):
            # note the prefixed whitespace
            # a subset of records in LTP_12_2021 have unusual lineages
            # where the lineage appears duplicated. All the ones found
            # also seem to start with a single space. see for example
            # KF863150
            ltp_tax.loc[idx, 'lineage'] = row['lineage'].strip().split(' Bacteria;')[0]

        if row['id'] == 'MK559992':
            # MK559992 has a genus that appears inconsistent with the species
            ltp_tax.loc[idx, 'lineage'] = row['lineage'].replace('Aureimonas', 'Aureliella')

        if row['id'] == 'HM038000':
            # the wrong order is specified
            ltp_tax.loc[idx, 'lineage'] = row['lineage'].replace('Oligoflexia',
                                                                 'Bdellovibrionia')

        if ';Campylobacterales;' in row['lineage']:
            # typo
            if ";Campylobacteria;" not in row['lineage']:
                ltp_tax.loc[idx, 'lineage'] = row['lineage'].replace('Campylobacterales;',
                                                                     'Campylobacteria;Campylobacterales;')

        if ' subsp. ' in row['lineage']:
            # we can't handle subspecies so let's remove
            parts = row['lineage'].split(';')
            species = parts[-1]
            species = species.split(' subsp. ')[0]
            parts[-1] = species
            ltp_tax.loc[idx, 'lineage'] = ';'.join(parts)

        if row['id'] == 'AB104858':
            # species name typo
            ltp_tax.loc[idx, 'lineage'] = row['lineage'].replace('Methanothermobacter wolfei',
                                                                 'Methanothermobacter wolfeii')
            ltp_tax.loc[idx, 'original_species'] = 'Methanothermobacter wolfeii'

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


def gtdb_taxids_from_metadata(ar_fp, bt_fp):
    ar = pd.read_csv(ar_fp, sep='\t', dtype=str)
    ba = pd.read_csv(bt_fp, sep='\t', dtype=str)
    gtdb_metadata = pd.concat([ar, ba])
    acc_to_wolid = re.compile(r'_([0-9]{9})\.')

    def fetch(x):
        return 'G' + acc_to_wolid.scanner(x).search().groups()[0]

    gtdb_metadata['wolid'] = gtdb_metadata['accession'].apply(fetch)

    return {r.wolid: r.ncbi_taxid for r in gtdb_metadata.itertuples()}


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


def non_wol_taxa_in_gtdb(tree_tips, ar_fp, bt_fp):
    ar = pd.read_csv(ar_fp, sep='\t', dtype=str)
    ba = pd.read_csv(bt_fp, sep='\t', dtype=str)
    gtdb_metadata = pd.concat([ar, ba]).set_index('accession')

    match = re.compile(r'(GB_GCA)|(RS_GCF)_([0-9]{9})\.[0-9]')
    matches = [n for n in tree_tips if match.match(n)]

    def make_wollike(n):
        extract = re.compile(r'_([0-9]{9})\.[0-9]')
        return 'G' + extract.search(n).groups()[0]

    extract_gca = re.compile(r'(GB_GCA_[0-9]{9}\.[0-9])')
    extract_gcf = re.compile(r'(RS_GCF_[0-9]{9}\.[0-9])')

    acc = defaultdict(list)
    for n in matches:
        a = extract_gca.match(n)
        b = extract_gcf.match(n)
        if a:
            key = a.groups()[0]
        elif b:
            key = b.groups()[0]
        else:
            raise ValueError()
        acc[key].append(n)

    tax_data = gtdb_metadata.loc[set(acc)]['gtdb_taxonomy'].to_dict()
    expanded_data = [[v, tax_data[k]] for k, all_v in acc.items()
                     for v in all_v]
    expanded_data = pd.DataFrame(expanded_data, columns=['id', 'lineage'])
    expanded_data['wol_like'] = expanded_data['id'].apply(make_wollike)

    return expanded_data


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


def preprocess_ltp(ltp_tax, gtdb_unique, ncbi_to_gtdb):
    # use the synonym if it exists
    for _, r in ltp_tax.iterrows():
        if not pd.isnull(r['synonym']):
            r['original_species'] = r['synonym']

    # remove extraneous whitespace
    ltp_tax['original_species'] = ltp_tax['original_species'].apply(lambda x: x.strip())

    # set manually obtained tax ids
    ltp_tax['ncbi_tax_id'] = None
    for idx, row in ltp_tax.iterrows():
        if row['id'] in MANUAL_LTP_TAXID_ASSESSMENT:
            oldname, newname, taxid = MANUAL_LTP_TAXID_ASSESSMENT[row['id']]
            ltp_tax.loc[idx, 'ncbi_tax_id'] = taxid
            ltp_tax.loc[idx, 'original_species'] = newname

    # fetch other tax ids
    to_fetch = list(ltp_tax[ltp_tax['ncbi_tax_id'].isnull()]['original_species'])
    ltp_ncbi_taxids, ltp_taxid_name_lookup = names_to_taxid(to_fetch)

    # sanity check to make sure we have an ID for everything
    missing = {k for k, v in ltp_ncbi_taxids.items() if v is None}
    missing_records = ltp_tax[ltp_tax['original_species'].isin(missing)]
    for r in missing_records.itertuples():
        print(r.original_species)
        raise ValueError()

    # use the already set tax ID if exists otherwise use the one pulled via ETE3
    def ltp_tax_id(row):
        if not pd.isnull(row['ncbi_tax_id']):
            return row['ncbi_tax_id']
        else:
            name = row['original_species']
            if name in ltp_ncbi_taxids:
                return ltp_ncbi_taxids[name]
            else:
                if name in ltp_taxid_name_lookup:
                    reduced_name = ltp_taxid_name_lookup[name]
                    return ltp_ncbi_taxids[reduced_name]
                else:
                    raise ValueError("Cannot resolve %s" % name)
    ltp_tax['ncbi_tax_id'] = ltp_tax.apply(ltp_tax_id, axis=1)

    ltp_tax = adjust_ltp(ltp_tax, gtdb_unique, ncbi_to_gtdb)

    return ltp_tax


def gtdb_failed_names(failed_fp):
    # these are from the qc_failed file. GTDB has different QC processes
    # than WOL
    df = pd.read_csv(failed_fp, sep='\t', dtype=str).set_index('Accession')

    acc_to_wolid = re.compile(r'_([0-9]{9})\.')

    def fetch(x):
        return 'G' + acc_to_wolid.scanner(x).search().groups()[0]

    df.index = [fetch(x) for x in df.index]
    df['NCBI species'] = df['NCBI species'].apply(lambda x: x.split('__', 1)[1])
    df = df[df['NCBI species'] != '']

    return df['NCBI species'].to_dict()


def preprocess_gtdb(gtdb_bacteria, gtdb_archaea, gtdb_failed, ids_of_focus):
    # TODO: consider using the refseq/genbank assembly metadata directly
    # which may be easier?

    # get what tax IDs we can get from the GTDB metadata
    gtdb_taxids = gtdb_taxids_from_metadata(gtdb_archaea, gtdb_bacteria)

    # get a mapping of what "failed" records we have to species
    gtdb_failed = gtdb_failed_names(gtdb_failed)
    ids_of_focus = set(ids_of_focus)

    # only concern ourselves with the set of IDs we care about
    gtdb_failed = {k: v for k, v in gtdb_failed.items() if k in ids_of_focus}

    # resolve manual mappings
    for k, (oldname, newname, id_) in MANUAL_GTDB_TAXID_ASSESSMENT.items():
        gtdb_taxids[k] = id_

    for k, (name, id_) in MANUAL_GTDB_FAILURES.items():
        gtdb_taxids[k] = id_

    # only fetch what we actually need to fetch
    to_fetch = [gtdb_failed[n] for n in gtdb_failed if n not in gtdb_taxids]
    gtdb_failed_taxids, gtdb_taxid_lookup = names_to_taxid(to_fetch)

    # pull either an already specified tax ID or one from ETE3
    tax_ids = {}
    for k in ids_of_focus:
        if k in gtdb_taxids:
            tax_ids[k] = gtdb_taxids[k]
        else:
            name = gtdb_failed.get(k)
            if name is None:
                print(name)
                raise ValueError()

            if name in gtdb_taxid_lookup:
                name = gtdb_taxid_lookup[name]

            id_ = gtdb_failed_taxids[name]
            tax_ids[k] = id_
    return tax_ids


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
@click.option('--gtdb-failed', type=click.Path(exists=True), required=True,
              help='The GTDB failed entries metadata')
@click.option('--output', type=click.Path(exists=False))
def harmonize(tree, gtdb, ltp, output, gtdb_archaea, gtdb_bacteria,
              gtdb_failed):
    tree = bp.to_skbio_treenode(bp.parse_newick(open(tree).read()))
    tree_tips = {n.name for n in tree.tips()}

    gtdb_tax = pd.read_csv(gtdb, sep='\t', names=['id', 'lineage'])
    gtdb_tax['wol_like'] = gtdb_tax['id']
    ltp_tax = pd.read_csv(ltp, sep='\t', names=['id', 'original_species',
                                                'lineage', 'u0', 'type',
                                                'synonym', 'u2'])

    ltp_tax = ltp_tax[ltp_tax['id'].isin(tree_tips)]
    gtdb_tax = gtdb_tax[gtdb_tax['id'].isin(tree_tips)]

    additional_gtdb_tax = non_wol_taxa_in_gtdb(tree_tips, gtdb_archaea, gtdb_bacteria)
    gtdb_tax = pd.concat([gtdb_tax, additional_gtdb_tax])

    gtdb_unique = gtdb_unique_species_mappings(gtdb_archaea, gtdb_bacteria)
    ncbi_to_gtdb = ncbi_to_gtdb_mappings(gtdb_archaea, gtdb_bacteria)

    gtdb_taxids = preprocess_gtdb(gtdb_bacteria, gtdb_archaea, gtdb_failed, list(gtdb_tax['wol_like']))
    ltp_tax = preprocess_ltp(ltp_tax, gtdb_unique, ncbi_to_gtdb)

    full_set_of_ncbi_tax_ids = [[r.id, r.ncbi_tax_id] for r in ltp_tax.itertuples()]
    full_set_of_ncbi_tax_ids += [[k, gtdb_taxids[k]] for k in gtdb_tax['wol_like']]
    full_set_of_ncbi_tax_ids = pd.DataFrame(full_set_of_ncbi_tax_ids,
                                            columns=['Feature ID', 'ncbi_tax_id'])
    full_set_of_ncbi_tax_ids.to_csv(output + '.ncbi_tax_ids', sep='\t', index=False,
                                    header=True)

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
    if missing:
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
