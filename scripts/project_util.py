#! /usr/bin/env python

import os
import sys
import re
import argparse

# Project paths
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SIM_DIR = os.path.join(PROJECT_DIR, 'simulations')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'configs')
BIN_DIR = os.path.join(PROJECT_DIR, 'bin')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
SCRIPTS_DIR = os.path.join(PROJECT_DIR, 'scripts')
SIMCO_SCRIPTS_DIR = os.path.join(SCRIPTS_DIR, 'simcoevolity-scripts')
PY_ENV_ACTIVATE_PATH = os.path.join(PROJECT_DIR, 'pyenv/bin/activate')
PYMSBAYES_DIR = os.path.join(PROJECT_DIR, "pymsbayes-analyses")

# Project regular expressions
SIMCOEVOLITY_CONFIG_NAME_PATTERN_STR = (
        # r"simcoevolity-sim-(?P<sim_num>\d+)-config.yml")
        r"(?P<var_only>var-only-)?"
        r"(?P<config_name>\S+)"
        r"-sim-(?P<sim_num>\d+)"
        r"-config.yml")
SIMCOEVOLITY_CONFIG_NAME_PATTERN = re.compile(
        r"^" + SIMCOEVOLITY_CONFIG_NAME_PATTERN_STR + r"$")

SIM_STATE_LOG_PATTERN_STR = (
        r"run-(?P<run_num>\d+)-"
        r"(?P<var_only>var-only-)?"
        r"(?P<config_name>\S+)"
        r"-sim-(?P<sim_num>\d+)"
        r"-config-state-run-(?P<dummy_run_num>\d+).log")
SIM_STATE_LOG_PATTERN = re.compile(
        r"^" + SIM_STATE_LOG_PATTERN_STR + r"$")

SIM_STDOUT_PATTERN_STR = (
        r"run-(?P<run_num>\d+)-"
        r"(?P<var_only>var-only-)?"
        r"(?P<config_name>\S+)"
        r"-sim-(?P<sim_num>\d+)"
        r"-config\.yml\.out")
SIM_STDOUT_PATTERN = re.compile(
        r"^" + SIM_STDOUT_PATTERN_STR + r"$")

BATCH_DIR_PATTERN_STR = r"batch-(?P<batch_num>\d+)"
BATCH_DIR_PATTERN = re.compile(
        r"^" + BATCH_DIR_PATTERN_STR + r"$")
BATCH_DIR_ENDING_PATTERN = re.compile(
            r"^.*" + BATCH_DIR_PATTERN_STR + r"(" + os.sep + r")?$")


def get_pbs_header(pbs_script_path,
        exe_name = "ecoevolity",
        exe_var_name = "exe_path",
        py_env = None):
    script_dir = os.path.dirname(pbs_script_path)
    relative_project_dir = os.path.relpath(PROJECT_DIR,
            script_dir)
    h = """#! /bin/bash

set -e

if [ -n "$PBS_JOBNAME" ]
then
    if [ -f "${{PBS_O_HOME}}/.bashrc" ]
    then
        source "${{PBS_O_HOME}}/.bashrc"
    fi
    cd {script_dir}
else
    cd "$( dirname "\${{BASH_SOURCE[0]}}" )"
fi

project_dir="{rel_project_dir}"
{exe_var_name}="${{project_dir}}/bin/{exe_name}"

if [ ! -x "${exe_var_name}" ]
then
    echo "ERROR: No executable '${{{exe_var_name}}}'."
    echo "       You probably need to run the project setup script."
    exit 1
fi

source "${{project_dir}}/modules-to-load.sh" >/dev/null 2>&1 || echo "    No modules loaded"

""".format(
        script_dir = script_dir,
        rel_project_dir = relative_project_dir,
        exe_name = exe_name,
        exe_var_name = exe_var_name,
        )
    if py_env:
        h += """if [ ! -f "${{project_dir}}/{py_env}/bin/activate" ]
then
    echo "ERROR: Python environment \\"${{project_dir}}/{py_env}\\" does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi
source "${{project_dir}}/{py_env}/bin/activate"

""".format(py_env = py_env)
    return h

def file_path_iter(directory, regex_pattern, include_match = False):
    for dir_path, dir_names, file_names in os.walk(directory):
        for f_name in file_names:
            match = regex_pattern.match(f_name)
            if match:
                path = os.path.join(dir_path, f_name)
                if include_match:
                    yield path, match
                else:
                    yield path

def flat_file_path_iter(directory, regex_pattern, include_match = False):
    for file_name in os.listdir(directory):
        match = regex_pattern.match(file_name)
        if match:
            path = os.path.join(directory, file_name)
            if include_match:
                yield path, match
            else:
                yield path

def dir_path_iter(directory, regex_pattern, include_match = False):
    for dir_path, dir_names, file_names in os.walk(directory):
        for d_name in dir_names:
            match = regex_pattern.match(d_name)
            if match:
                path = os.path.join(dir_path, d_name)
                if include_match:
                    yield path, match
                else:
                    yield path

def simcoevolity_config_iter(sim_directory = None):
    if sim_directory is None:
        sim_directory = SIM_DIR
    for path in file_path_iter(sim_directory, SIMCOEVOLITY_CONFIG_NAME_PATTERN):
        yield path

def batch_dir_iter(directory = None):
    if directory is None:
        directory = SIM_DIR
    for path in dir_path_iter(directory, BATCH_DIR_PATTERN):
        yield path

def sim_stdout_iter(sim_directory):
    for path in file_path_iter(sim_directory, SIM_STDOUT_PATTERN):
        yield path

def get_sim_state_log_paths(sim_directory = None):
    if sim_directory is None:
        sim_directory = SIM_DIR
    log_paths = {}
    log_path_iter = flat_file_path_iter(sim_directory, SIM_STATE_LOG_PATTERN, True)
    for path, match in log_path_iter:
        sim_number = match.group("sim_num")
        run_number = match.group("run_num")
        config_name = match.group("config_name")
        dummy_run_number = int(match.group("dummy_run_num"))
        if dummy_run_number != 1:
            sys.stderr.write(
                    "ERROR: Unexpected second run number '{0}' in state "
                    "log '{1}'\n".format(dummy_run_number, path))
            raise Exception("Unexpected second run number in state log path")
        is_var_only = bool(match.group("var_only"))
        if is_var_only:
            config_name = "var-only-" + config_name
        if not config_name in log_paths:
            log_paths[config_name] = {}
        if not sim_number in log_paths[config_name]:
            log_paths[config_name][sim_number] = {}
        log_paths[config_name][sim_number][run_number] = path
    return log_paths

# Utility functions for argparse
def arg_is_path(path):
    try:
        if not os.path.exists(path):
            raise
    except:
        msg = 'path {0!r} does not exist'.format(path)
        raise argparse.ArgumentTypeError(msg)
    return path

def arg_is_file(path):
    try:
        if not os.path.isfile(path):
            raise
    except:
        msg = '{0!r} is not a file'.format(path)
        raise argparse.ArgumentTypeError(msg)
    return path

def arg_is_dir(path):
    try:
        if not os.path.isdir(path):
            raise
    except:
        msg = '{0!r} is not a directory'.format(path)
        raise argparse.ArgumentTypeError(msg)
    return path

def arg_is_nonnegative_int(i):
    try:
        if int(i) < 0:
            raise
    except:
        msg = '{0!r} is not a non-negative integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)

def arg_is_positive_int(i):
    try:
        if int(i) < 1:
            raise
    except:
        msg = '{0!r} is not a positive integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)

def arg_is_positive_float(i):
    try:
        if float(i) <= 0.0:
            raise
    except:
        msg = '{0!r} is not a positive real number'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return float(i)

def arg_is_nonnegative_float(i):
    try:
        if float(i) < 0.0:
            raise
    except:
        msg = '{0!r} is not a non-negative real number'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return float(i)


def main():
    sys.stdout.write("{0}".format(PROJECT_DIR))


if __name__ == '__main__':
    main()
