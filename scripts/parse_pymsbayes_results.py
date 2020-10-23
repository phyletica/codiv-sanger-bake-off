#! /usr/bin/env python

import sys
import os
import math
import tarfile
import re
import glob
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
_LOG = logging.getLogger(os.path.basename(__file__))

import pycoevolity
import project_util

def parse_msbayes_results(sim_posterior_path,
        out_path):
    sim_dir = os.path.dirname(sim_posterior_path)
    sim_file = os.path.basename(sim_posterior_path)
    sim_file_pattern = re.compile(r'.*d1-m1-s(?P<sim_number>\d+)-(?P<nprior>\d+)-.*')
    sim_file_match = sim_file_pattern.match(sim_file) 
    prior_sample_size = int(sim_file_match.group("nprior"))
    data_model_dir = os.path.dirname(sim_dir)
    output_dir = os.path.dirname(data_model_dir)
    results_dir = os.path.dirname(output_dir)
    true_path = os.path.join(results_dir,
            "observed-summary-stats",
            "observed-1.txt")
    post_path_wild_card = os.path.join(
            sim_dir,
            "d1-m1-s*-{prior_sample_size}-posterior-sample.txt.gz".format(
                    prior_sample_size = prior_sample_size))
    posterior_paths = glob.glob(post_path_wild_card)
    number_of_sims = len(posterior_paths)
    number_of_pairs = 0
    for header in pycoevolity.parsing.parse_header_from_path(sim_posterior_path):
        if header.startswith("PRI.t."):
            number_of_pairs += 1

    _LOG.info("Parsing {0} posterior files with {1} pairs in {2!r}...".format(
            number_of_sims,
            number_of_pairs,
            sim_dir))

    column_header_prefixes = {
            'root_height': ("PRI.t.",),
            'pop_size_root': ("PRI.aTheta.",),
            'pop_size': ("PRI.d1Theta.", "PRI.d2Theta."),
            }
    parameter_prefixes = [
            "true",
            "mean",
            "eti_95_lower",
            "eti_95_upper",
            ]
    results = {
            "true_num_events": [],
            "map_num_events": [],
            "true_num_events_cred_level": [],
            "true_model": [],
            "map_model": [],
            "true_model_cred_level": [],
            "mean_map_model_distance": [],
            "median_map_model_distance": [],
            "mean_model_distance": [],
            "median_model_distance": [],
            "stddev_model_distance": [],
            "hpdi_95_lower_model_distance": [],
            "hpdi_95_upper_model_distance": [],
            "eti_95_lower_model_distance": [],
            "eti_95_upper_model_distance": [],
            
            }
    for pair_idx in range(number_of_pairs):
        results["num_events_{0}_p".format(pair_idx + 1)] = []
        for parameter_key, header_prefixes in column_header_prefixes.items():
            result_keys = ["{param}_c{comparison}sp1".format(
                            param = parameter_key,
                            comparison = pair_idx + 1)]
            if parameter_key== "pop_size":
                result_keys.append("{param}_c{comparison}sp2".format(
                                param = parameter_key,
                                comparison = pair_idx + 1))
            for suffix in result_keys:
                for prefix in parameter_prefixes:
                    k = "{0}_{1}".format(prefix, suffix)
                    results[k] = []
    true_values = pycoevolity.parsing.get_dict_from_spreadsheets(
            [true_path],
            sep = '\t')
    nsims = len(true_values["PRI.Psi"])
    assert (nsims == number_of_sims)
    for sim_idx in range(nsims):
        posterior_path = os.path.join(sim_dir,
                "d1-m1-s{sim_num}-{prior_sample_size}-posterior-sample.txt.gz".format(
                        sim_num = sim_idx + 1,
                        prior_sample_size = prior_sample_size))
        _LOG.info("Parsing {0}".format(posterior_path))
        posterior = pycoevolity.parsing.get_dict_from_spreadsheets(
                [posterior_path],
                sep = '\t')
        for pair_idx in range(number_of_pairs):
            for parameter_key, header_prefixes in column_header_prefixes.items():
                for header_prefix in header_prefixes:
                    header = "{0}{1}".format(header_prefix, pair_idx + 1)
                    pop_number = 1
                    if "d2Theta" in header:
                        pop_number = 2
                    result_key = "{param}_c{comparison}sp{pop}".format(
                            param = parameter_key,
                            comparison = pair_idx + 1,
                            pop = pop_number)
                    if parameter_key.startswith("pop_size"):
                        true_val = float(true_values[header][sim_idx]) / 4.0
                        post_sum = pycoevolity.stats.get_summary((float(x) / 4.0) for x in posterior[header])
                    else:
                        true_val = float(true_values[header][sim_idx])
                        post_sum = pycoevolity.stats.get_summary(float(x) for x in posterior[header])
                    results["true_" + result_key].append(true_val)
                    results["mean_" + result_key].append(post_sum['mean'])
                    results["eti_95_lower_" + result_key].append(post_sum['qi_95'][0])
                    results["eti_95_upper_" + result_key].append(post_sum['qi_95'][1])
        true_nevents = int(true_values["PRI.Psi"][sim_idx])
        true_div_times = []
        for pair_idx in range(number_of_pairs):
            time_key = "PRI.t.{0}".format(pair_idx + 1)
            true_div_times.append(float(true_values[time_key][sim_idx]))
        true_model, model_dict= pycoevolity.partition.standardize_partition(
                true_div_times)

        nevent_samples = tuple(int(x) for x in posterior["PRI.Psi"])
        div_model_samples = []
        for sample_idx in range(len(nevent_samples)):
            div_times = []
            for pair_idx in range(number_of_pairs):
                time_key = "PRI.t.{0}".format(pair_idx + 1)
                div_times.append(float(posterior[time_key][sample_idx]))
            div_model, div_dict = pycoevolity.partition.standardize_partition(
                    div_times)
            assert len(set(div_model)) == nevent_samples[sample_idx]
            div_model_samples.append(div_model)
        div_set_partitions = pycoevolity.partition.SetPartitionCollection.get_from_indices(
                div_model_samples)

        posterior_model_summary = pycoevolity.posterior.PosteriorModelSummary(
                nevent_samples = nevent_samples,
                model_samples = div_model_samples,
                set_partitions = div_set_partitions)

        true_model_cred = posterior_model_summary.get_model_credibility_level(
                true_model)
        map_models = posterior_model_summary.get_map_models()
        map_model = map_models[0]
        if len(map_models) > 1:
            if true_model in map_models:
                map_model = true_model
        true_nevents_cred = posterior_model_summary.get_number_of_events_credibility_level(
                true_nevents)
        map_numbers_of_events = posterior_model_summary.get_map_numbers_of_events()
        map_nevents = map_numbers_of_events[0]
        if len(map_numbers_of_events) > 1:
            if true_nevents in map_numbers_of_events:
                map_nevents = true_nevents

        results["true_num_events"].append(true_nevents)
        results["map_num_events"].append(map_nevents)
        results["true_num_events_cred_level"].append(true_nevents_cred)
        for i in range(number_of_pairs):
            results["num_events_{0}_p".format(i + 1)].append(
                    posterior_model_summary.get_number_of_events_probability(
                        i + 1))
        results["true_model"].append("".join((str(i) for i in true_model)))
        results["map_model"].append("".join((str(i) for i in map_model)))
        results["true_model_cred_level"].append(true_model_cred)

        model_dist_summary = pycoevolity.stats.get_summary(
                    posterior_model_summary.distances_from(true_model))
        results["mean_model_distance"].append(model_dist_summary["mean"])
        results["median_model_distance"].append(model_dist_summary["median"])
        results["stddev_model_distance"].append(math.sqrt(model_dist_summary["variance"]))
        results["hpdi_95_lower_model_distance"].append(model_dist_summary["hpdi_95"][0])
        results["hpdi_95_upper_model_distance"].append(model_dist_summary["hpdi_95"][1])
        results["eti_95_lower_model_distance"].append(model_dist_summary["qi_95"][0])
        results["eti_95_upper_model_distance"].append(model_dist_summary["qi_95"][1])
        map_model_distances = posterior_model_summary.get_map_model_distances_from(
                true_model)
        if len(map_model_distances) > 1:
            map_model_dist_summary = pycoevolity.stats.get_summary(
                    map_model_distances)
            results["mean_map_model_distance"].append(map_model_dist_summary["mean"])
            results["median_map_model_distance"].append(map_model_dist_summary["median"])
        else:
            results["mean_map_model_distance"].append(map_model_distances[0])
            results["median_map_model_distance"].append(map_model_distances[0])
        
    for parameter, values in results.items():
        assert len(values) == number_of_sims

    header = sorted(results.keys())

    with open(out_path, 'w') as stream:
        stream.write("{0}\n".format("\t".join(header)))
        for sim_index in range(number_of_sims):
            stream.write("{0}\n".format(
                    "\t".join(str(results[k][sim_index]) for k in header)))

def posterior_files(members):
    for tar_info in members:
        if (("1000000-posterior-sample.txt" in tar_info.name) or
                ("observed-1.txt" in tar_info.name)):
            yield tar_info


def main_cli(argv = sys.argv):
    base_results_dir = os.path.join(project_util.PYMSBAYES_DIR, "results")
    results_directories = (
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
    for results_dir_name in results_directories:
        results_dir = os.path.join(base_results_dir, results_dir_name)
        if not os.path.isdir(results_dir):
            tar_path = results_dir + ".tar.xz"
            tar = tarfile.open(tar_path, "r:xz")
            tar.extractall(
                    path = os.path.dirname(tar_path),
                    members = posterior_files(tar))
            tar.close()
        sim_posterior_path = os.path.join(
                results_dir,
                "pymsbayes-results",
                "pymsbayes-output",
                "d1",
                "m1",
                "d1-m1-s1-1000000-posterior-sample.txt.gz")
        out_path = os.path.join(base_results_dir,
                "results-{0}.csv".format(results_dir_name))
        parse_msbayes_results(
                sim_posterior_path = sim_posterior_path,
                out_path = out_path)


if __name__ == "__main__":
    main_cli()
