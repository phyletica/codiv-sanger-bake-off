#! /bin/bash

set -e

# Get path to directory of this script
script_dir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
project_dir="$(dirname "$script_dir")"

nex2yml_path="${project_dir}/bin/nex2yml"

all_msbayes_config_path="${project_dir}/data/empirical/msbayes-format/Papadopoulou-Knowles-2015/hABC_dppmsbayes_InputFiles/All13taxa/13taxa_dppbayes.cfg"
soil_msbayes_config_path="${project_dir}/data/empirical/msbayes-format/Papadopoulou-Knowles-2015/hABC_dppmsbayes_InputFiles/Only6Soiltaxa/SoilTaxa_dppbayes.cfg"

base_output_dir="${project_dir}/data/empirical/ecoevolity-format"
if [ ! -e "$base_output_dir" ]
then
    mkdir "$base_output_dir"
fi
output_dir="${base_output_dir}/Papadopoulou-Knowles-2015"
if [ ! -e "$output_dir" ]
then
    mkdir "$output_dir"
fi
config_out_dir="${output_dir}/configs"
if [ ! -e "$config_out_dir" ]
then
    mkdir "$config_out_dir"
fi
data_out_dir="${output_dir}/data"
if [ ! -e "$data_out_dir" ]
then
    mkdir "$data_out_dir"
fi

if [ ! -f "${project_dir}/pyenv/bin/activate" ]
then
    echo "ERROR: Python environment \"${project_dir}/pyenv\" does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi
source "${project_dir}/pyenv/bin/activate"

tmp_dir=$(mktemp -d 2>/dev/null || mktemp -d -t 'convert-medi-tmp')
tmp_cfg_path="${tmp_dir}/mediterranean-all-pairs-config.yml"

msb2nex_stderr_path="${output_dir}/msb2nex-stderr.txt"
nex2yml_stderr_path="${output_dir}/nex2yml-stderr.txt"
pop_label_path="${output_dir}/population-labels.sh"

# Convert msbayes config and fasta files to a ecoevolity config and nexus files
msb2nex --charsets --remove-triallelic-sites --recode-ambig-states-as-missing \
    --output-dir "$tmp_dir" \
    "$all_msbayes_config_path" 1> "$tmp_cfg_path" 2> "$msb2nex_stderr_path"

# Convert nexus files to yaml allele count files
"$nex2yml_path" --relax-missing-sites "$tmp_cfg_path" > "$nex2yml_stderr_path"

# Copy yaml allele count files to output directory and change '.nex.yml' file
# name suffix to just '.yml'
for yml_path in "$tmp_dir"/*.nex.yml
do
    yml_name="$(basename $yml_path)"
    yml_out_name="$(echo "$yml_name" | sed -e "s/\.nex//g")"
    yml_out_path="${data_out_dir}/$yml_out_name"
    cp "$yml_path" "$yml_out_path"
done

# Create a mapping of the labels created by msb2nex to the population labels in
# original fasta files
echo "labels='\\" > "$pop_label_path"
for s in $(grep --no-filename "^ *[A-Za-z0-9_-]*pop[12]$" "$tmp_dir"/*.nex)
do
    sp=${s%%_*}
    l=${s##*_}
    x=${s#*_}
    p=${x%%_*}
    echo "-l \"$l\" \"$sp $p\""
done | uniq >> "$pop_label_path"
echo "'" >> "$pop_label_path"

# Create ecoevolity yaml config with correct directory paths to yaml allele
# count files
msb2nex --charsets --remove-triallelic-sites --recode-ambig-states-as-missing \
    --output-dir "$tmp_dir" \
    --comparison-prefix "../data/" \
    --yaml-config-only \
    "$all_msbayes_config_path" > "$tmp_cfg_path"

# Copy ecoevolity yaml config to output directory and replace '.nex' extension
# in comparison paths to '.yml'
sed -e "s/\.nex/\.yml/g" "$tmp_cfg_path" > "${config_out_dir}/$(basename "$tmp_cfg_path")"

# Create another ecoevolity yaml config for only the soil taxa
tmp_cfg_path="${tmp_dir}/mediterranean-soil-pairs-config.yml"
msb2nex --charsets --remove-triallelic-sites --recode-ambig-states-as-missing \
    --output-dir "$tmp_dir" \
    --comparison-prefix "../data/" \
    --yaml-config-only \
    "$soil_msbayes_config_path" > "$tmp_cfg_path"

# Copy ecoevolity yaml soil config to output directory and replace '.nex'
# extension in comparison paths to '.yml'
sed -e "s/\.nex/\.yml/g" "$tmp_cfg_path" > "${config_out_dir}/$(basename "$tmp_cfg_path")"

# Remove all the temp files
rm -r "$tmp_dir"
