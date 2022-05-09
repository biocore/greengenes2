#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --time 48:00:00
#SBATCH --mem 256gb

source activate qiime2-2022.2
cd ..

if [[ ! -d backbone ]];
then
    echo "Not in the correct directory"
    exit 1
fi

v=${version}
b=data/${version}/${base} 

python scripts/fetch_backbone_seqs.py ${b}/0.1/${label}-consensus-strings ${b}/seqs.fa ${b}/backbone

qiime tools import \
    --input-path ${b}/backbone.fna \
    --output-path ${b}/${v}.backbone.full-length.fna.qza \
    --type FeatureData[Sequence] &
qiime tools import \
    --input-path ${b}/backbone.tax \
    --output-path ${b}/${v}.backbone.tax.qza \
    --type FeatureData[Taxonomy] &
wait

qiime feature-classifier extract-reads \
    --i-sequences ${b}/${v}.backbone.full-length.fna.qza \
    --p-f-primer GTGYCAGCMGCCGCGGTAA --p-r-primer GGACTACNVGGGTWTCTAAT \
    --p-read-orientation both \
    --o-reads ${b}/${v}.backbone.v4.fna.qza \
    --p-n-jobs 32 

qiime feature-classifier fit-classifier-naive-bayes \
    --i-reference-reads ${b}/${v}.backbone.v4.fna.qza \
    --i-reference-taxonomy ${b}/${v}.backbone.tax.qza \
    --o-classifier ${b}/${v}.backbone.v4.nb.qza

#qiime feature-classifier fit-classifier-naive-bayes \
#    --i-reference-reads ${b}/${v}.backbone.full-length.fna.qza \
#    --i-reference-taxonomy ${b}/${v}.backbone.tax.qza \
#    --o-classifier ${b}/${v}.backbone.full-length.nb.qza
