#! /usr/bin/env python

import os
import sys
import random
import re
import argparse

import project_util


def number_of_existing_sims(out_dir, config_path, locus_size):
    nreps_regex = re.compile(r"^\s*number_of_reps=(?P<nreps>\d+)\s*$")
    script_path, sim_name = get_script_path(out_dir, config_path,
            "bogus", locus_size)
    nsims = 0
    if os.path.exists(out_dir):
        for file_name in os.listdir(out_dir):
            if file_name.startswith(sim_name) and file_name.endswith(".sh"):
                nreps_found = False
                file_path = os.path.join(out_dir, file_name)
                with open(file_path, "r") as in_stream:
                    for line in in_stream:
                        match = nreps_regex.match(line)
                        if match:
                            nreps = int(match.group("nreps"))
                            nsims += nreps
                            nreps_found = True
                if not nreps_found:
                    raise Exception(
                            "\'{0}\' did not have \'number_of_reps\' line".format(
                                file_path))
    return nsims


def get_script_path(out_dir, config_path, batch_id_string,
        locus_size):
    config_file_name = os.path.basename(config_path)
    config_file_prefix, extension = os.path.splitext(config_file_name)
    sim_name = "{0}-locus-{1}".format(config_file_prefix, locus_size)
    out_file_name = "{prefix}-batch-{batch_id}.sh".format(
            prefix = sim_name,
            batch_id = batch_id_string)
    out_path = os.path.join(out_dir, out_file_name)
    return out_path, sim_name

def write_simcoevolity_script(out_dir,
        config_path,
        batch_id_string,
        number_of_reps,
        locus_size,
        fixed_config_prefixes):
    script_path, sim_name = get_script_path(out_dir, config_path,
            batch_id_string, locus_size)
    script_dir = os.path.dirname(script_path)
    if os.path.exists(script_path):
        raise Exception("Script path '{0}' already exists!".format(script_path))
    relative_config_path = os.path.relpath(
            config_path,
            script_dir)
    relative_prior_config_path = relative_config_path
    for fixed_prefix in fixed_config_prefixes:
        if sim_name.startswith(fixed_prefix):
            relative_prior_config_path = os.path.join(
                    os.path.dirname(relative_prior_config_path),
                    os.path.basename(relative_prior_config_path)[len(fixed_prefix):]
                    )
    batch_dir = os.path.join(project_util.SIM_DIR,
            sim_name,
            "batch-{0}".format(batch_id_string))
    relative_batch_dir = os.path.relpath(
            batch_dir,
            script_dir)
    qsub_script = os.path.join(project_util.SCRIPTS_DIR,
            "set_up_ecoevolity_qsubs.py")
    relative_qsub_script = os.path.relpath(
            qsub_script,
            script_dir)
    with open(script_path, "w") as out_stream:
        out_stream.write("{0}".format(
                project_util.get_pbs_header(
                    script_path,
                    exe_name = "simcoevolity",
                    py_env = "pyenv")
                ))
        out_stream.write(
                "rng_seed={seed}\n"
                "number_of_reps={number_of_reps}\n"
                "locus_size={locus_size}\n"
                "config_path=\"{relative_config_path}\"\n"
                "prior_config_path=\"{relative_prior_config_path}\"\n"
                "output_dir=\"{relative_batch_dir}\"\n"
                "qsub_set_up_script_path=\"{qsub_script}\"\n"
                "mkdir -p \"$output_dir\"\n\n"
                "\"$exe_path\" --seed=\"$rng_seed\" "
                "-n \"$number_of_reps\" "
                "-p \"$prior_config_path\" "
                "-l \"$locus_size\" "
                "-o \"$output_dir\" "
                "\"$config_path\" "
                "&& \"$qsub_set_up_script_path\" \"$output_dir\"\n".format(
                    seed = batch_id_string,
                    number_of_reps = number_of_reps,
                    locus_size = locus_size,
                    relative_config_path = relative_config_path,
                    relative_prior_config_path = relative_prior_config_path,
                    relative_batch_dir = relative_batch_dir,
                    qsub_script = relative_qsub_script))
    sys.stdout.write("Script written to '{0}'\n".format(script_path))

def main_cli():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config_paths',
            metavar = 'ECOEVOLITY-CONFIG-PATH',
            nargs = '+',
            type = project_util.arg_is_file,
            help = ('Path to ecoevolity config file.'))
    parser.add_argument('-n', '--number-of-reps',
            action = 'store',
            default = 10,
            type = project_util.arg_is_positive_int,
            help = ('Number of simulation replicates for the new batch.'))
    parser.add_argument('-m', '--max-number-of-reps',
            action = 'store',
            default = 500,
            type = project_util.arg_is_positive_int,
            help = ('Maximum number of simulation replicates across all '
                    'batches.'))
    parser.add_argument('-l', '--locus-size',
            action = 'store',
            default = 500,
            type = project_util.arg_is_positive_int,
            help = ('The number of bases in each locus.'))
    parser.add_argument('--seed',
            action = 'store',
            type = project_util.arg_is_positive_int,
            help = ('Seed for random number generator.'))

    args = parser.parse_args()

    max_random_int = 999999999
    max_num_digits = len(str(max_random_int))

    rng = random.Random()
    if not args.seed:
        args.seed = random.randint(1, max_random_int)
    rng.seed(args.seed)

    batch_num = rng.randint(1, max_random_int)
    batch_num_str = str(batch_num).zfill(max_num_digits)

    fixed_config_prefixes = ("fixed-independent-", "fixed-simultaneous-")

    out_dir = project_util.SIMCO_SCRIPTS_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_sim_scripts_created = 0
    for config_path in args.config_paths:
        n_existing_sims = number_of_existing_sims(out_dir, config_path,
                args.locus_size)
        if n_existing_sims >= args.max_number_of_reps:
            sys.stdout.write(
                    "No additional simulation replicates needed for \'{0}\'... "
                    "Skipping!\n".format(
                        os.path.basename(config_path)))
            continue
        elif (n_existing_sims + args.number_of_reps) > args.max_number_of_reps:
            nreps_needed = args.max_number_of_reps - n_existing_sims
            sys.stdout.write(
                    "Only {0} simulation reps needed for \'{1}\'; changing "
                    "number of reps to {2}\n".format(
                        nreps_needed,
                        os.path.basename(config_path),
                        nreps_needed))
            args.number_of_reps = nreps_needed

        write_simcoevolity_script(out_dir, config_path, batch_num_str,
                number_of_reps =  args.number_of_reps,
                locus_size = args.locus_size,
                fixed_config_prefixes = fixed_config_prefixes)
        num_sim_scripts_created += 1

    if num_sim_scripts_created < 1:
        sys.stdout.write("No simcoevolity scripts were written.\n")
    else:
        sys.stdout.write("\nSimcoevolity scripts successfully written.\n")
        sys.stdout.write("\nBatch ID:\n")
        sys.stdout.write("\t{0}\n".format(batch_num_str))

if __name__ == "__main__":
    main_cli()
