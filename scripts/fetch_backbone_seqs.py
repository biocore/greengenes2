import sys
import skbio
import pandas as pd

tax = pd.read_csv(sys.argv[1], sep='\t', dtype=str, index_col=0)
keep = {i for i in tax.index}
hit = []
wol_hit = set()
with open(sys.argv[3] + '.fna', 'wb') as fp:
    for rec in skbio.read(sys.argv[2], format='fasta'):
        id_ = rec.metadata['id']
        id_part = id_.rsplit('_', 1)[0]
        if id_ in keep:
            s = rec._string.replace(b'-', b'')
            fp.write(b">%s\n%s\n" % (id_.encode('ascii'), s))
            hit.append(id_)
        elif id_part in keep and id_part not in wol_hit:
            s = rec._string.replace(b'-', b'')
            fp.write(b">%s\n%s\n" % (id_part.encode('ascii'), s))
            hit.append(id_part)
            wol_hit.add(id_part)

index = hit
new_tax = pd.DataFrame([tax.loc[i] for i in index],
                       index=index)
new_tax.columns = ['Taxon']
new_tax.index.name = 'Feature ID'
new_tax.to_csv(sys.argv[3] + '.tax', sep='\t', index=True, header=True)
