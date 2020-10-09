#! /bin/bash

set -e

if [ -n "$PBS_JOBNAME" ]
then
    if [ -f "${PBS_O_HOME}/.bashrc" ]
    then
        source "${PBS_O_HOME}/.bashrc"
    fi
    cd $PBS_O_WORKDIR
fi

project_dir="../.."

if [ ! -f "${project_dir}/pyenv-abc/bin/activate" ]
then
    echo "ERROR: Python environment \"${project_dir}/pyenv-abc\" does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi
source "${project_dir}/pyenv-abc/bin/activate"

nsites="02500"
npairs="05"
locus_length="500"

reps=1000
nprocs=8
nprior=1000000
batch_size=12500
nsums=100000
npost=2000
nquantiles=1000
sortindex=0
seed=37851841

prior_name="pairs-${npairs}-sites-${nsites}-locus-${locus_length}"
cfg_name="$prior_name"

output_dir="../results/${cfg_name}"
if [ ! -d "$output_dir" ]
then
    mkdir -p $output_dir
fi

dmc.py --np $nprocs -r $reps -o "../configs/${cfg_name}.cfg" -p ../priors/${prior_name}/pymsbayes-results/pymsbayes-output/prior-stats-summaries -n $nprior --prior-batch-size $batch_size --num-posterior-samples $npost --num-standardizing-samples $nsums -q $nquantiles --sort-index $sortindex --output-dir $output_dir --seed $seed --no-global-estimate --compress 1>run-reject-${cfg_name}.sh.out 2>&1
