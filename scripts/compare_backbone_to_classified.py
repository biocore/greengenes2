import pandas as pd
import click


@click.command()
@click.option('--backbone-taxonomy', type=click.Path(exists=True), required=True)
@click.option('--decorated-taxonomy', type=click.Path(exists=True), required=True)
@click.option('--level', type=int, default=1, required=True)
@click.option('--examine', type=str, required=False)
@click.option('--get-records', is_flag=True, required=False, default=False)
def compare(backbone_taxonomy, decorated_taxonomy, level, examine, get_records):
    backbone = pd.read_csv(backbone_taxonomy, sep='\t', names=['id', 'taxon']).set_index('id')
    decorated = pd.read_csv(decorated_taxonomy, sep='\t', names=['id', 'taxon']).set_index('id')

    backbone['target'] = backbone['taxon'].apply(lambda x: x.split('; ')[level])
    decorated['target'] = decorated['taxon'].apply(lambda x: x.split('; ')[level])

    print(len(backbone))
    print(len(decorated))
    decorated = decorated.loc[set(backbone.index) & set(decorated.index)]
    backbone = backbone.loc[set(backbone.index) & set(decorated.index)]
    print(len(backbone))
    print(len(decorated))


    results = []
    for name, grp in backbone.groupby('target'):
        obs = decorated[decorated['target'] == name]
        tp = len(set(obs.index) & set(grp.index))
        fp = len(set(obs.index) - set(grp.index))
        fn = len(set(grp.index) - set(obs.index))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = 2 * ((precision * recall) / (precision + recall))

        results.append((name,
                        len(grp),
                        tp, fp, fn,
                        precision, recall, fmeasure))

    df = pd.DataFrame(results, columns=['name',
                                        'grpsize',
                                        'true positive',
                                        'false positive',
                                        'false negative',
                                        'precision',
                                        'recall',
                                        'fmeasure'])
    df.to_csv(f'{level}.results.tsv', sep='\t', index=False, header=True)
    df = df[df['fmeasure'] < 0.95]
    df.sort_values('grpsize', ascending=False, inplace=True)
    print(df)

    with pd.option_context('display.max_colwidth', 1000, 'display.max_columns', None):
        if examine:
            bb = backbone[backbone['target'] == examine]
            obs = decorated[decorated['target'] == examine]

            fp = set(obs.index) - set(bb.index)
            print('false positive examples:')
            x = list(fp)[:5]
            if x:
                print(backbone.loc[x, 'taxon'])
                print('---')
                print(decorated.loc[x, 'taxon'])
            fn = set(bb.index) - set(obs.index)
            print()
            print('false negative examples:')
            x = list(fn)[:5]
            if x:
                print(backbone.loc[x, 'taxon'])
                print('---')
                print(decorated.loc[x, 'taxon'])

    if get_records:
        results = []
        for n in df['name']:
            bbname = backbone[backbone['target'] == n]
            decname = decorated[decorated['target'] == n]

            for id in set(decname.index) - set(bbname.index):
                exp = backbone.loc[id, 'target']
                results.append((id, n, exp))

        results = pd.DataFrame(results, columns=['id', 'observed', 'expected'])
        results.to_csv(f'{level}.false_positive.records.tsv', sep='\t', index=False, header=True)

if __name__ == '__main__':
    compare()
