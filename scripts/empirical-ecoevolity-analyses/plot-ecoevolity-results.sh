#!/bin/bash

function run_summary_tools () {
    gzip -d -k run-?-${ecoevolity_config_prefix}-config-state-run-1.log.gz
    pyco-sumtimes -f -w "$plot_width" --violin -y "$time_ylabel" -b $burnin "${label_array[@]}" -p "${plot_dir}/pyco-sumtimes-${ecoevolity_config_prefix}-" run-?-${ecoevolity_config_prefix}-config-state-run-1.log
    pyco-sumsizes -f -w "$plot_width" --violin --base-font-size $size_base_font -y "$size_ylabel" -b $burnin "${label_array[@]}" -p "${plot_dir}/pyco-sumsizes-${ecoevolity_config_prefix}-" run-?-${ecoevolity_config_prefix}-config-state-run-1.log
    if [ -e "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-nevents.txt" ]
    then
        rm "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-nevents.txt"
    fi
    if [ -e "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-model.txt" ]
    then
        rm "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-model.txt"
    fi
    "$sumco_exe_path" -b $burnin -n 1000000 -p "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-" -c "${config_dir}/${ecoevolity_config_prefix}-config.yml" run-?-${ecoevolity_config_prefix}-config-state-run-1.log
    pyco-sumevents -f -w "$plot_width" --bf-font-size $bf_font_size -p "${plot_dir}/pyco-sumevents-${ecoevolity_config_prefix}-" --no-legend "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-nevents.txt"
    rm run-?-${ecoevolity_config_prefix}-config-state-run-1.log
}

set -e

label_array=()
convert_labels_to_array() {
    local concat=""
    local t=""
    label_array=()

    for word in $@
    do
        local len=`expr "$word" : '.*"'`

        [ "$len" -eq 1 ] && concat="true"

        if [ "$concat" ]
        then
            t+=" $word"
        else
            word=${word#\"}
            word=${word%\"}
            label_array+=("$word")
        fi

        if [ "$concat" -a "$len" -gt 1 ]
        then
            t=${t# }
            t=${t#\"}
            t=${t%\"}
            label_array+=("$t")
            t=""
            concat=""
        fi
    done
}

burnin=101

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
ecoevolity_data_dir="${project_dir}/data/empirical/ecoevolity-format"
sumco_exe_path="${project_dir}/bin/sumcoevolity"

plot_base_dir="${project_dir}/results/empirical-plots"
if [ ! -d "$plot_base_dir" ]
then
    mkdir -p "$plot_base_dir"
fi

time_ylabel="Population"
size_ylabel="Population"

bf_font_size=2.0

size_base_font=9.0

plot_width=6.0

config_prefixes=( "mediterranean-all-pairs" "mediterranean-soil-pairs" "philippines-all-pairs" "philippines-negros-panay-pairs" )

for ecoevolity_config_prefix in "${config_prefixes[@]}"
do
    config_dir="${ecoevolity_data_dir}/philippines/configs"
    source "${ecoevolity_data_dir}/philippines/population-labels.sh"
    if [[ $ecoevolity_config_prefix == medi* ]]
    then
        config_dir="${ecoevolity_data_dir}/Papadopoulou-Knowles-2015/configs"
    source "${ecoevolity_data_dir}/Papadopoulou-Knowles-2015/population-labels.sh"
    fi
    convert_labels_to_array $labels
    input_dir="${ecoevolity_output_dir}/${ecoevolity_config_prefix}"
    plot_dir="${plot_base_dir}/${ecoevolity_config_prefix}"
    if [ ! -d "$plot_dir" ]
    then
        mkdir "$plot_dir"
    fi
    cd "$input_dir"

    run_summary_tools

    cd "$plot_dir"
    
    for p in pyco-*.pdf
    do
        pdfcrop "$p" "$p"
    done
done

cd "$current_dir"
