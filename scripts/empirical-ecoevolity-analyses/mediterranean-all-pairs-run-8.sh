#! /bin/bash

set -e

run=8
config_prefix="mediterranean-all-pairs"

# Get path to project directory
project_dir="$( cd ../.. && pwd )"

# Load modules
echo "Loading modules..."
source "${project_dir}/modules-to-load.sh" >/dev/null 2>&1 || echo "    No modules loaded"

emp_output_dir="${project_dir}/empirical-ecoevolity-output"
if [ ! -d "$emp_output_dir" ]
then
    mkdir "$emp_output_dir"
fi

output_dir="${emp_output_dir}/${config_prefix}"
if [ ! -d "$output_dir" ]
then
    mkdir "$output_dir"
fi

config_path="${project_dir}/data/empirical/ecoevolity-format/Papadopoulou-Knowles-2015/configs/${config_prefix}-config.yml"
prefix="${output_dir}/run-${run}-"

"${project_dir}/bin/ecoevolity" --seed "$run" --prefix "$prefix" "$config_path" 1>"${prefix}${config_prefix}.out" 2>&1
