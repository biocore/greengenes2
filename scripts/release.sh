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

# oriented to 2022.7
#20596375 ./entropy1.2-gap0.3/entropy1.2-gap0.3.txt
#13122699 ./entropy0.8-gap0.1/entropy0.8-gap0.1.txt
#19551295 ./entropy1-gap0.2/entropy1-gap0.2.txt
#20174774 ./entropy1.2-gap0.2/entropy1.2-gap0.2.txt

python scripts/release-prep.py \
    --coarse-level ${b}/entropy1.2-gap0.3 \
    --coarse-threshold 0.3 \
    --other-level ${b}/entropy0.8-gap0.1 \
    --other-level ${b}/entropy1-gap0.2 \
    --other-level ${b}/entropy1.2-gap0.3 \
    --other-level-threshold 0.1 \
    --other-level-threshold 0.2 \
    --other-level-threshold 0.3 \
    --basename ${label} \
    --coarse-level-seqs ${b}/entropy1.2-gap0.3/seqs.fa \
    --output release/${version} \
    --release-name ${version}

qiime tools import \
    --input-path ${b}/entropy1.2-gap0.3/seqs.fa \
    --output-path release/${version}/${version}.seqs.fna.qza \
    --type FeatureData[Sequence]
     
