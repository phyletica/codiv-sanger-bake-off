#! /usr/bin/env python

import sys
import os
import glob
import gzip

import project_util

def count_sims(results_table_path):
    n = 0
    with gzip.open(results_table_path, 'r') as in_stream:
        for line in in_stream:
            n += 1
    # Subtract 1 for the header row
    return n - 1

def eco_break_down(sim_names):
    sim_counts = {}
    for sim_name in sim_names:
        assert sim_name not in sim_counts
        sim_counts[sim_name] = []
        # sys.stdout.write(f"{sim_name}:\n")
        for batch_dir in glob.glob(
            os.path.join(
                project_util.SIM_DIR,
                sim_name,
                "batch*")
        ):
            batch_name = os.path.basename(batch_dir)
            # sys.stdout.write(f"  {batch_name}:\n")
            result_path = os.path.join(
                batch_dir,
                "simcoevolity-results.tsv.gz",
            )
            n = 0
            if os.path.exists(result_path):
                n = count_sims(result_path)
            # sys.stdout.write(f"    {n}\n")
            batch = batch_name.split('-')[1]
            assert batch not in sim_counts[sim_name]
            sim_counts[sim_name].append((batch, n))
    return sim_counts


def summarize_sim_counts(sim_anmes):
    for sim_name in sim_names:
        eco_count = 0
        for result_path in glob.glob(
            os.path.join(
                project_util.SIM_DIR,
                sim_name,
                "batch*",
                "simcoevolity-results.tsv.gz")
        ):
            eco_count += count_sims(result_path)
        pymsb_count = 0
        for result_path in glob.glob(
            os.path.join(
                project_util.PYMSBAYES_DIR,
                "results",
                "results-{0}.tsv.gz".format(sim_name)
            )
        ):
            pymsb_count += count_sims(result_path)
        
        sys.stderr.write(f"{sim_name}:\n")
        sys.stderr.write(f"    ecoevolity: {eco_count}:\n")
        sys.stderr.write(f"    pymsbayes: {pymsb_count}:\n")

if __name__ == "__main__":
    sim_names = (
        "fixed-independent-pairs-05-sites-00500-locus-500",
        "fixed-independent-pairs-05-sites-01000-locus-500",
        "fixed-independent-pairs-05-sites-02500-locus-500",
        "fixed-independent-pairs-05-sites-10000-locus-500",
        "fixed-simultaneous-pairs-05-sites-00500-locus-500",
        "fixed-simultaneous-pairs-05-sites-01000-locus-500",
        "fixed-simultaneous-pairs-05-sites-02500-locus-500",
        "fixed-simultaneous-pairs-05-sites-10000-locus-500",
        "pairs-05-sites-00500-locus-500",
        "pairs-05-sites-01000-locus-500",
        "pairs-05-sites-02500-locus-500",
        "pairs-05-sites-10000-locus-500",
    )

    summarize_sim_counts(sim_names)

    sim_counts = eco_break_down(sim_names)

    sys.stderr.write("\nMissing evoevolity simulations:\n\n")
    for sim_name, batch_counts in sim_counts.items():
        for batch, count in batch_counts:
            if count < 1:
                sys.stdout.write(
                    f"simulations/{sim_name}/batch-{batch}\n"
                    # f"  n={count}\n"
                )
        
