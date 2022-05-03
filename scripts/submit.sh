#!/bin/bash

if [[ $# -eq 2 ]];
then
    version=$1
    base=$2
else
    echo "Usage: submit.sh <version> <placement_base>"
    exit 1
fi

label=placement.rt.dec
method=multifurcating

# join strings in an array, see 
# https://stackoverflow.com/a/17841619
function join_by { local IFS="$1"; shift; echo "$*"; }

declare -a jobs
for d in $(/bin/ls -1 ../data/${version}/${base})
do
    if [[ -d "../data/${version}/${base}/${d}" ]]; then
        jobs+=($(sbatch \
            --export version=$version,base=${base},threshold=${d},label=${label},method=${method} \
            --parsable \
            decorate.sh))
    fi
done

dependency=$(join_by : ${jobs[@]})
sbatch \
    --dependency=afterok:${dependency} \
    --export version=${version},label=${label}.${method}.clean.pro \
    release.sh
