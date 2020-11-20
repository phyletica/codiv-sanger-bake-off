#! /usr/bin/env python

import sys
import os
import re
import configobj

import pycoevolity

import project_util


def correct_machine():
    machine_cpuinfo = "/proc/cpuinfo"
    correct_cpuinfo = os.path.join(project_util.PYMSBAYES_DIR, "cpuinfo")
    if not os.path.isfile(correct_cpuinfo):
        return False
    if not os.path.isfile(machine_cpuinfo):
        return False
    mhz_pattern = re.compile(r"^\s*cpu\s+mhz\s*:\s*[0123456789.]+\s*", re.IGNORECASE)
    machine_str = ""
    correct_str = ""
    with open(machine_cpuinfo, "r") as in_stream:
        for line in in_stream:
            m = mhz_pattern.match(line)
            if not m:
                machine_str += line
    with open(correct_cpuinfo, "r") as in_stream:
        for line in in_stream:
            m = mhz_pattern.match(line)
            if not m:
                correct_str += line
    return machine_str == correct_str

class EcoevolityClocker(object):
    def __init__(self, sim_dirs):
        self.runtime_summarizer = pycoevolity.stats.SampleSummarizer()
        for sim_dir in sim_dirs:
            for stdout_path in project_util.sim_stdout_iter(sim_dir):
                stdout = pycoevolity.parsing.EcoevolityStdOut(stdout_path)
                self.runtime_summarizer.add_sample(stdout.run_time)

    def __str__(self):
        s = ("Ecoevolity runtime summary:\n" 
             "    n = {n}\n"
             "    mean = {mean}\n"
             "    std_deviation = {std_deviation}\n"
             "    min = {mn}\n"
             "    max = {mx}\n".format(
                 n = self.runtime_summarizer.n,
                 mean = self.runtime_summarizer.mean,
                 std_deviation = self.runtime_summarizer.std_deviation,
                 mn = self.runtime_summarizer.minimum,
                 mx = self.runtime_summarizer.maximum)
             )
        return s

class PyMsBayesClocker(object):
    duration_pattern_str = (
            r"((?P<days>\d+)\s+day[s]*[,]*\s+)?"
            r"(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>[0123456789.]+)"
            )
    duration_pattern = re.compile(
        r"^\s*" + duration_pattern_str + "\s*$")
    def __init__(self,
            prior_pymsbayes_info_path,
            reject_pymsbayes_info_path):
        times = []
        for p in (prior_pymsbayes_info_path, reject_pymsbayes_info_path):
            data = configobj.ConfigObj(p)
            runtime = data["pymsbayes"]["run_stats"]["total_duration"]
            nprocs = int(data["pymsbayes"]["num_processors"])
            nreps = int(data["pymsbayes"]["simulation_reps"])

            if not isinstance(runtime, str):
                runtime = " ".join(runtime)
            m = self.duration_pattern.match(runtime)
            if not m:
                raise Exception(
                        "Could not match total_duration in {0!r}".format(
                            p))
            d = {}
            for k, v in m.groupdict(0.0).items():
                d[k] = float(v)
            t = (
                    (d["days"] * 24 * 60 * 60 ) +
                    (d["hours"] * 60 * 60 ) +
                    (d["minutes"] * 60 ) +
                    d["seconds"]
                )
            times.append(t)
        self._prior_time, self._reject_time = times

    def _get_prior_time(self):
        return self._prior_time
    def _get_reject_time(self):
        return self._reject_time
    def _get_time(self):
        return self._prior_time + self._reject_time
    prior_time = property(_get_prior_time)
    reject_time = property(_get_reject_time)
    time = property(_get_time)

def write_results(
        sim_name,
        pymsbayes_clocker,
        ecoevolity_clocker,
        out = sys.stdout):
    s = ecoevolity_clocker.runtime_summarizer
    out.write("{0}:\n".format(sim_name))
    out.write("    pymsbayes:\n".format(sim_name))
    out.write("        prior_time: {0}\n".format(
        pymsbayes_clocker.prior_time))
    out.write("        mean_reject_time: {0}\n".format(
        pymsbayes_clocker.reject_time))
    out.write("        mean: {0}\n".format(
        pymsbayes_clocker.time))
    out.write("    ecoevolity:\n".format(sim_name))
    out.write("        mean: {0}\n".format(s.mean))
    out.write("        std_deviation: {0}\n".format(s.std_deviation))
    out.write("        min: {0}\n".format(s.minimum))
    out.write("        max: {0}\n".format(s.maximum))
    out.write("        n: {0}\n".format(s.n))

def main_cli():

    if not correct_machine():
        sys.stderr.write(
                "\n"
                "ERROR: This is not meant to run on this computer\n"
                "It is only meant to run on the standard machine that both\n"
                "PyMsBayes and ecoevolity were run on.\n\n")
        sys.exit(1)

    out_path = os.path.join(project_util.RESULTS_DIR,
            "run-time-comparison.yml")

    if os.path.exists(out_path):
        sys.stderr.write(
                "\n"
                "The output file:\n"
                "  {0!r}\n"
                "already exists. If you intend to overwrite it, please delete\n"
                "the existing copy and re-run this script.\n\n".format(
                    out_path))
        sys.exit(1)

    try:
        os.makedirs(project_util.RESULTS_DIR)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    sim_names = [
            "pairs-05-sites-10000-locus-500",
            ]

    with open(out_path, "w") as out_stream:
        out_stream.write("---\n")

        for sim_name in sim_names:
            pymsbayes_prior_path = os.path.join(project_util.PYMSBAYES_DIR,
                    "priors",
                    sim_name,
                    "pymsbayes-results",
                    "pymsbayes-info.txt")
            pymsbayes_reject_path = os.path.join(project_util.PYMSBAYES_DIR,
                    "results",
                    sim_name,
                    "pymsbayes-results",
                    "pymsbayes-info.txt")
            sim_dir_path = os.path.join(project_util.SIM_DIR, sim_name)

            pymsbayes_clocker = PyMsBayesClocker(
                    prior_pymsbayes_info_path = pymsbayes_prior_path,
                    reject_pymsbayes_info_path = pymsbayes_reject_path)
            ecoevolity_clocker = EcoevolityClocker([sim_dir_path])

            write_results(
                    sim_name = sim_name,
                    pymsbayes_clocker = pymsbayes_clocker,
                    ecoevolity_clocker = ecoevolity_clocker,
                    out = sys.stdout)
            write_results(
                    sim_name = sim_name,
                    pymsbayes_clocker = pymsbayes_clocker,
                    ecoevolity_clocker = ecoevolity_clocker,
                    out = out_stream)

    sys.stdout.write("Results written to {0!r}\n".format(out_path))


if __name__ == "__main__":
    main_cli()
