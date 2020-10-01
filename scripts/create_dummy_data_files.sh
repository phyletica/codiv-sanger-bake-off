#!/bin/bash

if [ ! -f "../pyenv/bin/activate" ]
then
    echo "ERROR: Python environment '../pyenv/bin/activate' does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi

source "../pyenv/bin/activate"

nspecies=2
ngenomes=10

data_dir="../data"
mkdir -p "$data_dir"

for nchars in 00500 01000 02500 10000
do
    i=1
    while [ "$i" -lt 6 ]
    do
        prefix="c${i}sp"
        comp_str="0${i}"
        if [ ${#i} -gt 1 ]
        then
            comp_str="$i"
        fi
        outfile="${data_dir}/comp${comp_str}-${nspecies}species-${ngenomes}genomes-${nchars}chars.txt"
        pyco-dummy-data \
            --nspecies "$nspecies" \
            --ngenomes "$ngenomes" \
            --ncharacters "$nchars" \
            --prefix "$prefix" \
            > "$outfile"
        i=`expr $i + 1`
    done
done
