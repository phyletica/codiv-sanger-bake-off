#! /bin/bash

set -e

if [ -n "$PBS_JOBNAME" ]
then
    if [ -f "${PBS_O_HOME}/.bashrc" ]
    then
        source "${PBS_O_HOME}/.bashrc"
    fi
    cd /gpfs01/home/tcm0036/codiv-sanger-bake-off/scripts/simcoevolity-scripts
else
    cd "$( dirname "\${BASH_SOURCE[0]}" )"
fi

project_dir="../.."
exe_path="${project_dir}/bin/simcoevolity"

if [ ! -x "$exe_path" ]
then
    echo "ERROR: No executable '${exe_path}'."
    echo "       You probably need to run the project setup script."
    exit 1
fi

source "${project_dir}/modules-to-load.sh" >/dev/null 2>&1 || echo "    No modules loaded"

if [ ! -f "${project_dir}/pyenv/bin/activate" ]
then
    echo "ERROR: Python environment \"${project_dir}/pyenv\" does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi
source "${project_dir}/pyenv/bin/activate"

rng_seed=687578308
number_of_reps=20
locus_size=500
config_path="../../configs/pairs-05-sites-00500.yml"
prior_config_path="../../configs/pairs-05-sites-00500.yml"
output_dir="../../simulations/pairs-05-sites-00500-locus-500/batch-687578308"
qsub_set_up_script_path="../set_up_ecoevolity_qsubs.py"
mkdir -p "$output_dir"

"$exe_path" --seed="$rng_seed" -n "$number_of_reps" -p "$prior_config_path" -l "$locus_size" -o "$output_dir" "$config_path" && "$qsub_set_up_script_path" "$output_dir"
