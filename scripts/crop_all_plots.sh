#! /bin/bash

set -e


paths="$(ls ../results/plots/*.pdf) $(ls ../results/empirical-plots/*/*.pdf)"

for f in $paths
do
    filesuffix="${f##*-}"
    if [ "$filesuffix" = "cropped.pdf" ]
    then
        continue
    fi
    
    echo "Cropping $f"
    n=${f/\.pdf/-cropped\.pdf}
    # pdfcrop --margins 20 $f $n
    pdfcrop $f $n
done
