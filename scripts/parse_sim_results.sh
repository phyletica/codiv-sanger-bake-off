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

project_dir=".."

if [ ! -f "${project_dir}/pyenv/bin/activate" ]
then
    echo "ERROR: Python environment \"${project_dir}/pyenv\" does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi
source "${project_dir}/pyenv/bin/activate"


./parse_sim_results.py $@
