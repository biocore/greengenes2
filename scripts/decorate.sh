#!/bin/bash

#SBATCH --mem 64g
#SBATCH --time 4:00:00
#SBATCH --output %x.%N.%j.out
#SBATCH --error %x.%N.%j.err
#SBATCH -J decorate
set -x
set -e

source activate qiime2-2022.2
cd ..

if [[ ! -d backbone ]];
then
    echo "Not in the correct directory"
    exit 1
fi

t=${threshold}
v=${version}
b=data/${version}/${base} 
declabel=${label}
method=${method}

# reroot based on archaea
t2t reroot -n backbone/${v}/archaea.ids \
    -p ${b}/${t}/placement.jplace \
    -o ${b}/${t}/placement.rt.jplace \
    --out-of-target support-files/${v}/arbitrary_bacteria.txt


if [[ -f backbone/${v}/secondary_taxonomy.tsv ]]; then
    # decorate taxonomy
    t2t decorate -m backbone/${v}/taxonomy.tsv \
        -o ${b}/${t}/${declabel} \
        -p ${b}/${t}/placement.rt.jplace \
        --use-node-id \
        --secondary-taxonomy backbone/${v}/secondary_taxonomy.tsv \
        --recover-polyphyletic \
        --correct-binomials \
        --min-count 1 
else
    t2t decorate -m backbone/${v}/taxonomy.tsv \
        -o ${b}/${t}/${declabel} \
        -p ${b}/${t}/placement.rt.jplace \
        --use-node-id \
        --correct-binomials \
        --min-count 1 
fi

# resolve placements
bp placement --placements ${b}/${t}/${declabel}.jplace \
    --output ${b}/${t}/${declabel}.${method}.nwk \
    --method ${method}

if [[ -f support-files/${v}/labels_to_remove ]]; then
    t2t filter -t ${b}/${t}/${declabel}.${method}.nwk \
        -m support-files/${v}/labels_to_remove \
        -o ${b}/${t}/${declabel}.${method}.clean.nwk
else
    cp ${b}/${t}/${declabel}.${method}.nwk ${b}/${t}/${declabel}.${method}.clean.nwk
fi

# 2022.7 has the placement IDs in the entropy file, whereas seqs.fa is all seqs
# grep "^>" ${b}/${t}/seqs.fa | tr -d ">" > ${b}/${t}/placements.ids
cat ${b}/${t}/entropy*.txt > ${b}/${t}/placements.ids
t2t promote-multifurcation -t ${b}/${t}/${declabel}.${method}.clean.nwk \
    -f ${b}/${t}/placements.ids \
    -o ${b}/${t}/${declabel}.${method}.clean.pro.nwk

# extract the full taxonomy
t2t fetch -t ${b}/${t}/${declabel}.${method}.clean.pro.nwk \
    -o ${b}/${t}/${declabel}.${method}.clean.pro.tsv

t2t fetch -t ${b}/${t}/${declabel}.${method}.clean.pro.nwk \
    -o ${b}/${t}/${declabel}.${method}.clean.pro.taxonomy.nwk \
    --as-tree
