#!/bin/bash

#SBATCH --mem 256g
#SBATCH --time 24:00:00
#SBATCH --output %x.%N.%j.out
#SBATCH --error %x.%N.%j.err
#SBATCH -J gg2-release
set -x
set -e

source activate qiime2-2022.2
cd ..

if [[ ! -d backbone ]];
then
    echo "Not in the correct directory"
    exit 1
fi

b=data/${version}/${base} 
label=${label}

mkdir -p release/${version}

cp ${b}/*backbone*.qza release/${version}/

python scripts/release-prep.py \
    --coarse-level ${b}/0.3 \
    --coarse-threshold 0.3 \
    --other-level ${b}/0.1 \
    --other-level ${b}/0.2 \
    --other-level-threshold 0.1 \
    --other-level-threshold 0.2 \
    --basename ${label} \
    --coarse-level-seqs ${b}/0.3/seqs.fa \
    --output release/${version} \
    --release-name ${version}
