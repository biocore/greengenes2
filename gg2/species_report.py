import re
import parasail
import pandas as pd
import skbio

ASV = re.compile(r'^[0-9]{8}$')
OPERON = re.compile(r'^MJ[0-9]{3}-')
WOL = re.compile(r"^(G[09][0-9]{8})_[0-9]+$")


#s__Amycolatopsis_D_322092 alkalitolerans
POLY_SPECIES = re.compile(r"^(s__[a-zA-Z0-9-]+)(_[A-Z]+)?(_[0-9]+)? ([a-zA-Z0-9-]+)(_[A-Z]+)?(_[0-9]+)?$")
POLY_LABEL = re.compile(r"^([dpcofg]__[a-zA-Z0-9-]+)(_[A-Z]+)?(_[0-9]+)?$")
def extract_species(name):
    match = POLY_SPECIES.match(name)
    if match is None:
        return "s__"
    else:
        groups = match.groups()
        # just the binomial bits
        return "%s %s" % (groups[0], groups[3])


def extract_label(name):
    if name.startswith('s__'):
        return extract_species(name)
    else:
        match = POLY_LABEL.match(name)
        if match is None:
            return name
        else:
            groups = match.groups()
            return groups[0]


def parse_full_length(in_, limit):
    d = {}
    wol_seen = set()
    print("WARNING: using first WOL record")

    for rec in skbio.read(in_, format='fasta'):
        id_ = rec.metadata['id']
        wol_match = WOL.match(id_)
        if wol_match is not None:
            wol_id = wol_match.groups()[0]

            if wol_id in wol_seen:
                continue
            else:
                wol_seen.add(wol_id)
                id_ = wol_id

        if id_ in limit:
            d[id_] = skbio.DNA(rec)

    return d


def seq_id(aln):
    length = len(aln[0])
    matches = sum([a == b for a, b in aln.iter_positions()])
    return (matches / length) * 100, length


def check_alignment(sequences, in_species, out_species):
    result = []
    for in_ in in_species:
        # only check alignment on isolates
        in_operon = OPERON.match(in_)
        if in_operon is not None:
            continue

        in_sequence = str(sequences[in_])

        for out in out_species:
            # only check alignment on isolates
            out_operon = OPERON.match(out)
            if out_operon is not None:
                continue

            out_sequence = str(sequences[out])
            aln = parasail.nw_stats_striped_sat(in_sequence, out_sequence, 1, 1, parasail.nuc44)
            length = min(len(in_sequence), len(out_sequence))
            id_ = (aln.matches / length) * 100
            score = aln.score
            result.append((in_, out, id_, length, score))
    return pd.DataFrame(result, columns=['Subject', 'Query', 'Sequence ID',
                                         'Length', 'Score'])
