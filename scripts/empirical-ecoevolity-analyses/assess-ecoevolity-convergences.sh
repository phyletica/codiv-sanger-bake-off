#!/bin/bash

function run_pyco_sumchains () {
    gzip -d -k run-?-${ecoevolity_config_prefix}-config-state-run-1.log.gz
    echo "pyco-sumchains run-?-${ecoevolity_config_prefix}-config-state-run-1.log"
    pyco-sumchains run-?-${ecoevolity_config_prefix}-config-state-run-1.log 1>pyco-sumchains-${ecoevolity_config_prefix}-table.txt 2>pyco-sumchains-${ecoevolity_config_prefix}-stderr.txt
    rm run-?-${ecoevolity_config_prefix}-config-state-run-1.log
}

set -e

current_dir="$(pwd)"
function return_on_exit () {
    cd "$current_dir"
}
trap return_on_exit EXIT

# Get path to project directory
project_dir="$( cd ../.. && pwd )"
if [ ! -f "${project_dir}/pyenv/bin/activate" ]
then
    echo "ERROR: Python environment \"${project_dir}/pyenv\" does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi
source "${project_dir}/pyenv/bin/activate"

ecoevolity_output_dir="${project_dir}/empirical-ecoevolity-output"

config_prefixes=( "mediterranean-all-pairs" "mediterranean-soil-pairs" "philippines-all-pairs" "philippines-negros-panay-pairs" )

for ecoevolity_config_prefix in "${config_prefixes[@]}"
do
    target_dir="${ecoevolity_output_dir}/${ecoevolity_config_prefix}"
    cd "${target_dir}"
    run_pyco_sumchains
done

cd "$current_dir"
