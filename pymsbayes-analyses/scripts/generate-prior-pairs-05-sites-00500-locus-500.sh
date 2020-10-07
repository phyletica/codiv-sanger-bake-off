#! /bin/bash

if [ -n "$PBS_JOBNAME" ]
then
    source ${PBS_O_HOME}/.bash_profile
    cd $PBS_O_WORKDIR
fi

nsites="00500"
npairs="05"
locus_length="500"

nprocs=8
nprior=500000
batch_size=6250
nsums=100000
seed=1384268

cfg_name="pairs-${npairs}-sites-${nsites}-locus-${locus_length}"

output_dir="../priors/${cfg_name}"
if [ ! -d "$output_dir" ]
then
    mkdir -p $output_dir
fi

dmc.py --np $nprocs -r 1 -o "../configs/${cfg_name}.cfg" -p "../configs/${cfg_name}.cfg" -n $nprior --num-posterior-samples $batch_size --prior-batch-size $batch_size --num-standardizing-samples $nsums --output-dir "$output_dir" --seed $seed --generate-samples-only 1>generate-prior-${cfg_name}.sh.out 2>&1
