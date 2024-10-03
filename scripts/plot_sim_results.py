#! /usr/bin/env python

import sys
import os
import re
import math
import errno
import glob
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
_LOG = logging.getLogger(os.path.basename(__file__))

import pycoevolity
import project_util

import scipy.stats as st
import matplotlib as mpl

# Use TrueType (42) fonts rather than Type 3 fonts
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
tex_font_settings = {
        "text.usetex": True,
        "font.family": "sans-serif",
        # "font.serif": [
        #         "Computer Modern Roman",
        #         "Times",
        #         ],
        # "font.sans-serif": [
        #         "Computer Modern Sans serif",
        #         "Helvetica",
        #         ],
        # "font.cursive": [
        #         "Zapf Chancery",
        #         ],
        # "font.monospace": [
        #         "Computer Modern Typewriter",
        #         "Courier",
        #         ],
        "text.latex.preamble" : [
                "\\usepackage[T1]{fontenc}",
                "\\usepackage[cm]{sfmath}",
                ]
}

mpl.rcParams.update(tex_font_settings)

import matplotlib.pyplot as plt
from matplotlib import gridspec

def mwu(values1, values2):
    res = st.mannwhitneyu(
        values1,
        values2,
    )
    mwu_stat = res.statistic
    mwu_pval = res.pvalue
    num_comps = len(values1) * len(values2)
    mwu_auc = mwu_stat / num_comps
    if mwu_auc < 0.5:
        mwu_auc = 1.0 - mwu_auc
    return mwu_stat, mwu_auc, mwu_pval

def append_sum_of_mean_errors_column(
    results_dict,
    parameter_prefix = "root_height",
):
    mean_key_prefix = f"mean_{parameter_prefix}_"
    true_key_prefix = f"true_{parameter_prefix}_"
    mean_keys = []
    true_keys = []
    mean_key_suffixes = []
    true_key_suffixes = []
    for k in results_dict.keys():
        if k.startswith(mean_key_prefix):
            assert k not in mean_keys
            mean_keys.append(k)
            suffix = k.split('_')[-1]
            assert suffix not in mean_key_suffixes
            mean_key_suffixes.append(suffix)
        if k.startswith(true_key_prefix):
            if k.endswith("rank"):
                continue
            assert k not in true_keys
            true_keys.append(k)
            suffix = k.split('_')[-1]
            assert suffix not in true_key_suffixes
            true_key_suffixes.append(suffix)
    assert len(mean_keys) == len(true_keys)
    assert len(mean_keys) == len (mean_key_suffixes)
    assert mean_key_suffixes == true_key_suffixes

    n = len(results_dict[mean_keys[0]])
    sum_of_mean_errors = []
    for i in range(n):
        error_sum = 0.0
        for suffix in mean_key_suffixes:
            true_val = results_dict[f"true_{parameter_prefix}_{suffix}"][i]
            mean_val = results_dict[f"mean_{parameter_prefix}_{suffix}"][i]
            error = math.fabs(float(true_val) - float(mean_val))
            error_sum += error
        sum_of_mean_errors.append(error_sum)
    assert len(sum_of_mean_errors) == n
    res_key = f"sum_of_mean_errors_{parameter_prefix}"
    assert res_key not in results_dict
    results_dict[res_key] = sum_of_mean_errors

def get_nevents_probs(
        results_paths,
        nevents = 1):
    probs = []
    prob_key = "num_events_{0}_p".format(nevents)
    for d in pycoevolity.parsing.spreadsheet_iter(results_paths):
        probs.append((
                float(d[prob_key]),
                int(int(d["true_num_events"]) == nevents)
                ))
    return probs

def bin_prob_correct_tuples(probability_correct_tuples, nbins = 20):
    bin_upper_limits = list(get_sequence_iter(0.0, 1.0, nbins+1))[1:]
    bin_width = (bin_upper_limits[1] - bin_upper_limits[0]) / 2.0
    bins = [[] for i in range(nbins)]
    n = 0
    for (p, t) in probability_correct_tuples:
        n += 1
        binned = False
        for i, l in enumerate(bin_upper_limits):
            if p < l:
                bins[i].append((p, t))
                binned = True
                break
        if not binned:
            bins[i].append((p, t))
    total = 0
    for b in bins:
        total += len(b)
    assert total == n
    assert len(bins) == nbins
    est_true_tups = []
    for i, b in enumerate(bins):
        ests = [p for (p, t) in b]
        est = sum(ests) / float(len(ests))
        correct = [t for (p, t) in b]
        true = sum(correct) / float(len(correct))
        est_true_tups.append((est, true))
    return bins, est_true_tups

def get_nevents_estimated_true_probs(
        results_paths,
        nevents = 1,
        nbins = 20):
    nevent_probs = get_nevents_probs(
            results_paths = results_paths,
            nevents = nevents)
    _LOG.info("\tparsed results for {0} simulations".format(len(nevent_probs)))
    bins, tups = bin_prob_correct_tuples(nevent_probs, nbins = nbins)
    _LOG.info("\tbin sample sizes: {0}".format(
            ", ".join(str(len(b)) for b in bins)
            ))
    return bins, tups

def plot_nevents_estimated_vs_true_probs(
        results_paths,
        nevents = 1,
        nbins = 20,
        plot_file_prefix = ""):
    bins, est_true_probs = get_nevents_estimated_true_probs(
            results_paths = results_paths,
            nevents = nevents,
            nbins = nbins)

    plt.close('all')
    fig = plt.figure(figsize = (4.0, 3.5))
    ncols = 1
    nrows = 1
    gs = gridspec.GridSpec(nrows, ncols,
            wspace = 0.0,
            hspace = 0.0)

    ax = plt.subplot(gs[0, 0])
    x = [e for (e, t) in est_true_probs]
    y = [t for (e, t) in est_true_probs]
    sample_sizes = [len(b) for b in bins]
    line, = ax.plot(x, y)
    plt.setp(line,
            marker = 'o',
            markerfacecolor = 'none',
            markeredgecolor = '0.35',
            markeredgewidth = 0.7,
            markersize = 3.5,
            linestyle = '',
            zorder = 100,
            rasterized = False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    for i, (label, lx, ly) in enumerate(zip(sample_sizes, x, y)):
        if i == 0:
            ax.annotate(
                    str(label),
                    xy = (lx, ly),
                    xytext = (1, 1),
                    textcoords = "offset points",
                    horizontalalignment = "left",
                    verticalalignment = "bottom")
        elif i == len(x) - 1:
            ax.annotate(
                    str(label),
                    xy = (lx, ly),
                    xytext = (-1, -1),
                    textcoords = "offset points",
                    horizontalalignment = "right",
                    verticalalignment = "top")
        else:
            ax.annotate(
                    str(label),
                    xy = (lx, ly),
                    xytext = (-1, 1),
                    textcoords = "offset points",
                    horizontalalignment = "right",
                    verticalalignment = "bottom")
    ylabel_text = ax.set_ylabel("True probability", size = 14.0)
    ax.text(0.5, -0.14,
            "Posterior probability of one divergence",
            horizontalalignment = "center",
            verticalalignment = "top",
            size = 14.0)
    identity_line, = ax.plot(
            [0.0, 1.0],
            [0.0, 1.0])
    plt.setp(identity_line,
            color = '0.8',
            linestyle = '-',
            linewidth = 1.0,
            marker = '',
            zorder = 0)

    gs.update(left = 0.10, right = 0.995, bottom = 0.18, top = 0.91)

    plot_dir = os.path.join(project_util.PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_file_name = "est-vs-true-prob-nevent-1.pdf"
    if plot_file_prefix:
        plot_file_name = plot_file_prefix + "-" + plot_file_name
    plot_path = os.path.join(plot_dir,
            plot_file_name)
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))

def get_sequence_iter(start = 0.0, stop = 1.0, n = 10):
    assert(stop > start)
    step = (stop - start) / float(n - 1)
    return ((start + (i * step)) for i in range(n))

def truncate_color_map(cmap, min_val = 0.0, max_val = 10, n = 100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                    n = cmap.name,
                    a = min_val,
                    b = max_val),
            cmap(list(get_sequence_iter(min_val, max_val, n))))
    return new_cmap

def get_errors(values, lowers, uppers):
    n = len(values)
    assert(n == len(lowers))
    assert(n == len(uppers))
    return [[values[i] - lowers[i] for i in range(n)],
            [uppers[i] - values[i] for i in range(n)]]

def ci_width_iter(results, parameter_str):
    n = len(results["eti_95_upper_{0}".format(parameter_str)])
    for i in range(n):
        upper = float(results["eti_95_upper_{0}".format(parameter_str)][i])
        lower = float(results["eti_95_lower_{0}".format(parameter_str)][i])
        yield upper - lower

def absolute_error_iter(results, parameter_str):
    n = len(results["true_{0}".format(parameter_str)])
    for i in range(n):
        t = float(results["true_{0}".format(parameter_str)][i])
        e = float(results["mean_{0}".format(parameter_str)][i])
        yield math.fabs(t - e)


def plot_ess_versus_error(
        parameters,
        results_grid,
        column_labels = None,
        row_labels = None,
        parameter_label = "divergence time",
        plot_file_prefix = None):
    _LOG.info("Generating ESS vs CI scatter plots for {0}...".format(parameter_label))

    assert(len(parameters) == len(set(parameters)))
    if row_labels:
        assert len(row_labels) ==  len(results_grid)
    if column_labels:
        assert len(column_labels) == len(results_grid[0])

    nrows = len(results_grid)
    ncols = len(results_grid[0])

    if not plot_file_prefix:
        plot_file_prefix = parameters[0] 
    plot_file_prefix_ci = plot_file_prefix + "-ess-vs-ci-width"
    plot_file_prefix_error = plot_file_prefix + "-ess-vs-error"

    # Very inefficient, but parsing all results to get min/max for parameter
    ess_min = float('inf')
    ess_max = float('-inf')
    ci_width_min = float('inf')
    ci_width_max = float('-inf')
    error_min = float('inf')
    error_max = float('-inf')
    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            for parameter_str in parameters:
                ci_widths = tuple(ci_width_iter(results, parameter_str))
                errors = tuple(absolute_error_iter(results, parameter_str))
                ess_min = min(ess_min,
                        min(float(x) for x in results["ess_sum_{0}".format(parameter_str)]))
                ess_max = max(ess_max,
                        max(float(x) for x in results["ess_sum_{0}".format(parameter_str)]))
                ci_width_min = min(ci_width_min, min(ci_widths))
                ci_width_max = max(ci_width_max, max(ci_widths))
                error_min = min(error_min, min(errors))
                error_max = max(error_max, max(errors))
    ess_axis_buffer = math.fabs(ess_max - ess_min) * 0.05
    ess_axis_min = ess_min - ess_axis_buffer
    ess_axis_max = ess_max + ess_axis_buffer
    ci_width_axis_buffer = math.fabs(ci_width_max - ci_width_min) * 0.05
    ci_width_axis_min = ci_width_min - ci_width_axis_buffer
    ci_width_axis_max = ci_width_max + ci_width_axis_buffer
    error_axis_buffer = math.fabs(error_max - error_min) * 0.05
    error_axis_min = error_min - error_axis_buffer
    error_axis_max = error_max + error_axis_buffer

    plt.close('all')
    w = 1.6
    h = 1.5
    fig_width = (ncols * w) + 1.0
    fig_height = (nrows * h) + 0.7
    fig = plt.figure(figsize = (fig_width, fig_height))
    gs = gridspec.GridSpec(nrows, ncols,
            wspace = 0.0,
            hspace = 0.0)

    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):

            x = []
            y = []
            for parameter_str in parameters:
                x.extend(float(x) for x in results["ess_sum_{0}".format(parameter_str)])
                y.extend(ci_width_iter(results, parameter_str))

            assert(len(x) == len(y))
            ax = plt.subplot(gs[row_index, column_index])
            line, = ax.plot(x, y)
            plt.setp(line,
                    marker = 'o',
                    markerfacecolor = 'none',
                    markeredgecolor = '0.35',
                    markeredgewidth = 0.7,
                    markersize = 2.5,
                    linestyle = '',
                    zorder = 100,
                    rasterized = True)
            ax.set_xlim(ess_axis_min, ess_axis_max)
            ax.set_ylim(ci_width_axis_min, ci_width_axis_max)
            if column_labels and (row_index == 0):
                col_header = column_labels[column_index]
                ax.text(0.5, 1.015,
                        col_header,
                        horizontalalignment = "center",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
            if row_labels and (column_index == (ncols - 1)):
                row_label = row_labels[row_index]
                ax.text(1.015, 0.5,
                        row_label,
                        horizontalalignment = "left",
                        verticalalignment = "center",
                        rotation = 270.0,
                        transform = ax.transAxes)

    # show only the outside ticks
    all_axes = fig.get_axes()
    for ax in all_axes:
        if not ax.is_last_row():
            ax.set_xticks([])
        if not ax.is_first_col():
            ax.set_yticks([])

    # show tick labels only for lower-left plot 
    # all_axes = fig.get_axes()
    # for ax in all_axes:
    #     if ax.is_last_row() and ax.is_first_col():
    #         continue
    #     xtick_labels = ["" for item in ax.get_xticklabels()]
    #     ytick_labels = ["" for item in ax.get_yticklabels()]
    #     ax.set_xticklabels(xtick_labels)
    #     ax.set_yticklabels(ytick_labels)
    all_axes = fig.get_axes()
    for ax in all_axes:
        if not ax.is_last_row():
            xtick_labels = ["" for item in ax.get_xticklabels()]
            ax.set_xticklabels(xtick_labels)
        if not ax.is_first_col():
            ytick_labels = ["" for item in ax.get_yticklabels()]
            ax.set_yticklabels(ytick_labels)

    # avoid doubled spines
    all_axes = fig.get_axes()
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
            sp.set_linewidth(2)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        else:
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['right'].set_visible(True)

    fig.text(0.5, 0.001,
            "Effective sample size of {0}".format(parameter_label),
            horizontalalignment = "center",
            verticalalignment = "bottom",
            size = 18.0)
    fig.text(0.005, 0.5,
            "CI width {0}".format(parameter_label),
            horizontalalignment = "left",
            verticalalignment = "center",
            rotation = "vertical",
            size = 18.0)

    gs.update(left = 0.08, right = 0.98, bottom = 0.08, top = 0.97)

    plot_dir = os.path.join(project_util.PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(plot_dir,
            "{0}-scatter.pdf".format(plot_file_prefix_ci))
    plt.savefig(plot_path, dpi=600)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))


    _LOG.info("Generating ESS vs error scatter plots for {0}...".format(parameter_label))
    plt.close('all')
    w = 1.6
    h = 1.5
    fig_width = (ncols * w) + 1.0
    fig_height = (nrows * h) + 0.7
    fig = plt.figure(figsize = (fig_width, fig_height))
    gs = gridspec.GridSpec(nrows, ncols,
            wspace = 0.0,
            hspace = 0.0)

    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            x = []
            y = []
            for parameter_str in parameters:
                x.extend(float(x) for x in results["ess_sum_{0}".format(parameter_str)])
                y.extend(absolute_error_iter(results, parameter_str))
                

            assert(len(x) == len(y))
            ax = plt.subplot(gs[row_index, column_index])
            line, = ax.plot(x, y)
            plt.setp(line,
                    marker = 'o',
                    markerfacecolor = 'none',
                    markeredgecolor = '0.35',
                    markeredgewidth = 0.7,
                    markersize = 2.5,
                    linestyle = '',
                    zorder = 100,
                    rasterized = True)
            ax.set_xlim(ess_axis_min, ess_axis_max)
            ax.set_ylim(error_axis_min, error_axis_max)
            if column_labels and (row_index == 0):
                col_header = column_labels[column_index]
                ax.text(0.5, 1.015,
                        col_header,
                        horizontalalignment = "center",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
            if row_labels and (column_index == (ncols - 1)):
                row_label = row_labels[row_index]
                ax.text(1.015, 0.5,
                        row_label,
                        horizontalalignment = "left",
                        verticalalignment = "center",
                        rotation = 270.0,
                        transform = ax.transAxes)

    # show only the outside ticks
    all_axes = fig.get_axes()
    for ax in all_axes:
        if not ax.is_last_row():
            ax.set_xticks([])
        if not ax.is_first_col():
            ax.set_yticks([])

    # show tick labels only for lower-left plot 
    # all_axes = fig.get_axes()
    # for ax in all_axes:
    #     if ax.is_last_row() and ax.is_first_col():
    #         continue
    #     xtick_labels = ["" for item in ax.get_xticklabels()]
    #     ytick_labels = ["" for item in ax.get_yticklabels()]
    #     ax.set_xticklabels(xtick_labels)
    #     ax.set_yticklabels(ytick_labels)
    all_axes = fig.get_axes()
    for ax in all_axes:
        if not ax.is_last_row():
            xtick_labels = ["" for item in ax.get_xticklabels()]
            ax.set_xticklabels(xtick_labels)
        if not ax.is_first_col():
            ytick_labels = ["" for item in ax.get_yticklabels()]
            ax.set_yticklabels(ytick_labels)

    # avoid doubled spines
    all_axes = fig.get_axes()
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
            sp.set_linewidth(2)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        else:
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['right'].set_visible(True)

    fig.text(0.5, 0.001,
            "Effective sample size of {0}".format(parameter_label),
            horizontalalignment = "center",
            verticalalignment = "bottom",
            size = 18.0)
    fig.text(0.005, 0.5,
            "Absolute error of {0}".format(parameter_label),
            horizontalalignment = "left",
            verticalalignment = "center",
            rotation = "vertical",
            size = 18.0)

    gs.update(left = 0.08, right = 0.98, bottom = 0.08, top = 0.97)

    plot_path = os.path.join(plot_dir,
            "{0}-scatter.pdf".format(plot_file_prefix_error))
    plt.savefig(plot_path, dpi=600)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))


def generate_violin_plots(
        parameters,
        results_grid,
        comparison_labels,
        comparison_colors = None,
        column_labels = None,
        plot_width = 1.9,
        plot_height = 1.8,
        pad_left = 0.1,
        pad_right = 0.98,
        pad_bottom = 0.12,
        pad_top = 0.92,
        y_label = None,
        y_label_size = 18.0,
        x_tick_rotation = None,
        force_shared_y_range = True,
        force_shared_spines = True,
        show_means = True,
        mean_decimals = 2,
        show_sample_sizes = False,
        show_mwu = False,
        plot_file_prefix = None,
        show_legend = True):
    if force_shared_spines:
        force_shared_y_range = True
    num_comparisons = len(results_grid)
    assert len(comparison_labels) == num_comparisons
    if num_comparisons != 2:
        show_mwu = False
    num_plots = len(results_grid[0])
    if column_labels:
        assert len(column_labels) == num_plots
    for r in results_grid:
        assert len(r) == num_plots
    if comparison_colors:
        assert len(comparison_colors) == num_comparisons

    ncols = num_plots
    nrows = 1

    # Very inefficient, but parsing all results to get min/max for parameter
    y_min = float('inf')
    y_max = float('-inf')
    for comparison_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            for parameter_str in parameters:
                y_min = min(y_min,
                        min(float(x) for x in results[parameter_str]))
                y_max = max(y_max,
                        max(float(x) for x in results[parameter_str]))
    buff = 0.05
    y_buffer = math.fabs(y_max - y_min) * buff
    y_axis_min = y_min - y_buffer
    y_axis_max = y_max + y_buffer
    if show_sample_sizes:
        y_axis_min = y_min - (2 * y_buffer)
    if show_means:
        y_axis_max = y_max + (2 * y_buffer)

    plt.close('all')
    w = plot_width
    h = plot_height
    fig_width = (ncols * w)
    fig_height = (nrows * h)
    fig = plt.figure(figsize = (fig_width, fig_height))
    if force_shared_spines:
        gs = gridspec.GridSpec(nrows, ncols,
                wspace = 0.0,
                hspace = 0.0)
    else:
        gs = gridspec.GridSpec(nrows, ncols)

    for plot_idx in range(num_plots):
        ax = plt.subplot(gs[0, plot_idx])
        values = []
        for comparison_idx in range(num_comparisons):
            values.append([float(x) for x in results_grid[comparison_idx][plot_idx][parameter_str]])
        sample_sizes = [len(vals) for vals in values]

        positions = range(1, num_comparisons + 1)
        v = ax.violinplot(values,
                positions = positions,
                vert = True,
                widths = 0.9,
                showmeans = False,
                showextrema = False,
                showmedians = False,
                # points = 100,
                # bw_method = None,
                )

        using_colors = False
        colors = ["gray"] * num_comparisons
        if comparison_colors:
            colors = comparison_colors
            using_colors = True
        for i in range(len(v["bodies"])):
            v["bodies"][i].set_alpha(1)
            v["bodies"][i].set_facecolor(colors[i])
            v["bodies"][i].set_edgecolor(colors[i])

        means = []
        ci_lower = []
        ci_upper = []
        q1 = []
        q3 = []
        for sample in values:
            summary = pycoevolity.stats.get_summary(sample)
            means.append(summary["mean"])
            ci_lower.append(summary["qi_95"][0])
            ci_upper.append(summary["qi_95"][1])
            q1.append(pycoevolity.stats.quantile(sample, 0.25))
            q3.append(pycoevolity.stats.quantile(sample, 0.75))
        ax.vlines(positions, q1, q3,
                colors = "black",
                linestyle = "solid",
                linewidth = 8,
                zorder = 1)
        ax.vlines(positions, ci_lower, ci_upper,
                colors = "black",
                linestyle = "solid",
                zorder = 100)
        ax.scatter(positions, ci_lower,
                marker = "_",
                color = "black",
                s = 120,
                zorder = 200,
                )
        ax.scatter(positions, ci_upper,
                marker = "_",
                color = "black",
                s = 120,
                zorder = 200,
                )
        ax.scatter(positions, means,
                marker = ".",
                color = "white",
                s = 50,
                zorder = 300,
                )

        ax.xaxis.set_ticks(range(1, len(comparison_labels) + 1))
        xtick_labels = [item for item in ax.get_xticklabels()]
        assert(len(xtick_labels) == len(comparison_labels))
        for i in range(len(xtick_labels)):
            xtick_labels[i].set_text(comparison_labels[i])
        ax.set_xticklabels(xtick_labels)

        if force_shared_y_range:
            ax.set_ylim(y_axis_min, y_axis_max)
        else:
            y_mn = min(min(x) for x in values)
            y_mx = max(max(x) for x in values)
            y_buf = math.fabs(y_mx - y_mn) * buff
            y_ax_mn = y_mn - y_buf
            y_ax_mx = y_mx + y_buf
            if show_sample_sizes:
                y_ax_mn = y_mn - (y_buf * 2)
            if show_means:
                y_ax_mx = y_mx + (y_buf * 2)
            ax.set_ylim(y_ax_mn, y_ax_mx)
        if column_labels:
            col_header = column_labels[plot_idx]
            ax.text(0.5, 1.015,
                    col_header,
                    horizontalalignment = "center",
                    verticalalignment = "bottom",
                    transform = ax.transAxes)
        if show_sample_sizes:
            y_min, y_max = ax.get_ylim()
            y_n = y_min + ((y_max - y_min) * 0.001)
            for i in range(len(sample_sizes)):
                ax.text(i + 1, y_n,
                        "\\scriptsize {ss}".format(
                            ss = sample_sizes[i]),
                        horizontalalignment = "center",
                        verticalalignment = "bottom")
        if show_means:
            y_min, y_max = ax.get_ylim()
            y_mean = y_min + ((y_max - y_min) * 0.999)
            for i in range(len(means)):
                ax.text(i + 1, y_mean,
                        "\\scriptsize {mean:,.{ndigits}f}".format(
                            mean = means[i],
                            ndigits = mean_decimals),
                        horizontalalignment = "center",
                        verticalalignment = "top")

        if show_mwu:
            mwu_stat, mwu_auc, mwu_pval = mwu(values[0], values[1])
            y_min, y_max = ax.get_ylim()
            y_pos = y_min + ((y_max - y_min) * 0.93)
            ax.text(1.5, y_pos,
                    "\\scriptsize PoS = {x:,.{ndigits}f}".format(
                        x = mwu_auc,
                        ndigits = 3),
                    horizontalalignment = "center",
                    verticalalignment = "top")
            y_pos = y_min + ((y_max - y_min) * 0.87)
            p_str = f"{mwu_pval:,.3f}"
            if mwu_pval < 0.001:
                p_str = f"{mwu_pval:,.1g}"
            ax.text(1.5, y_pos,
                    f"\\scriptsize \\textit{{p}} = {p_str}",
                    horizontalalignment = "center",
                    verticalalignment = "top")

    if force_shared_spines:
        # show only the outside ticks
        all_axes = fig.get_axes()
        for ax in all_axes:
            if not ax.is_last_row():
                ax.set_xticks([])
            if not ax.is_first_col():
                ax.set_yticks([])

        # show tick labels only for lower-left plot 
        all_axes = fig.get_axes()
        for ax in all_axes:
            if ax.is_last_row() and ax.is_first_col():
                continue
            # xtick_labels = ["" for item in ax.get_xticklabels()]
            ytick_labels = ["" for item in ax.get_yticklabels()]
            # ax.set_xticklabels(xtick_labels)
            ax.set_yticklabels(ytick_labels)

        # avoid doubled spines
        all_axes = fig.get_axes()
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
                sp.set_linewidth(2)
            if ax.is_first_row():
                ax.spines['top'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
            else:
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(True)
            else:
                ax.spines['right'].set_visible(True)

    if x_tick_rotation:
        all_axes = fig.get_axes()
        for ax in all_axes:
            for tick in ax.get_xticklabels():
                tick.set_rotation(x_tick_rotation)

    if using_colors and show_legend:
        fig.legend(comparison_colors,
                labels = comparison_labels,
                loc = "upper center",
                mode = "expand",
                ncol = len(comparison_colors),
                # borderaxespad = -0.5,
                title = None)

    if y_label:
        fig.text(0.005, 0.5,
                y_label,
                horizontalalignment = "left",
                verticalalignment = "center",
                rotation = "vertical",
                size = y_label_size)

    gs.update(left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_dir = os.path.join(project_util.PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(plot_dir,
            "{0}-violin.pdf".format(plot_file_prefix))
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))

def generate_scatter_plots(
        parameters,
        results_grid,
        column_labels = None,
        row_labels = None,
        parameter_label = "divergence time",
        parameter_symbol = "\\tau",
        max_psrf = 1.2,
        min_ess = 200,
        highlight_color = "red",
        plot_width = 1.9,
        plot_height = 1.8,
        pad_left = 0.1,
        pad_right = 0.98,
        pad_bottom = 0.12,
        pad_top = 0.92,
        x_label_size = 18.0,
        y_label_size = 18.0,
        y_label = None,
        plot_file_prefix = None):
    _LOG.info("Generating scatter plots for {0}...".format(parameter_label))

    assert(len(parameters) == len(set(parameters)))
    if row_labels:
        assert len(row_labels) ==  len(results_grid)
    if column_labels:
        assert len(column_labels) == len(results_grid[0])

    nrows = len(results_grid)
    ncols = len(results_grid[0])

    if not plot_file_prefix:
        plot_file_prefix = parameters[0] 

    # Very inefficient, but parsing all results to get min/max for parameter
    parameter_min = float('inf')
    parameter_max = float('-inf')
    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            for parameter_str in parameters:
                parameter_min = min(parameter_min,
                        min(float(x) for x in results["true_{0}".format(parameter_str)]))
                parameter_max = max(parameter_max,
                        max(float(x) for x in results["true_{0}".format(parameter_str)]))
                parameter_min = min(parameter_min,
                        min(float(x) for x in results["mean_{0}".format(parameter_str)]))
                parameter_max = max(parameter_max,
                        max(float(x) for x in results["mean_{0}".format(parameter_str)]))
    axis_buffer = math.fabs(parameter_max - parameter_min) * 0.05
    axis_min = parameter_min - axis_buffer
    axis_max = parameter_max + axis_buffer

    plt.close('all')
    w = plot_width
    h = plot_height
    fig_width = (ncols * w)
    fig_height = (nrows * h)
    fig = plt.figure(figsize = (fig_width, fig_height))
    gs = gridspec.GridSpec(nrows, ncols,
            wspace = 0.0,
            hspace = 0.0)

    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            x = []
            y = []
            y_upper = []
            y_lower = []
            bad_x = []
            bad_y = []
            bad_y_upper = []
            bad_y_lower = []
            has_chain_stats = bool("psrf_{0}".format(parameters[0]) in results)
            if (not has_chain_stats) or ((not max_psrf) and (not min_ess)):
                for parameter_str in parameters:
                    x.extend(float(x) for x in results["true_{0}".format(parameter_str)])
                    y.extend(float(x) for x in results["mean_{0}".format(parameter_str)])
                    y_lower.extend(float(x) for x in results["eti_95_lower_{0}".format(parameter_str)])
                    y_upper.extend(float(x) for x in results["eti_95_upper_{0}".format(parameter_str)])
            else:
                for parameter_str in parameters:
                    for i in range(len(results["true_{0}".format(parameter_str)])):
                        true_val = float(results["true_{0}".format(parameter_str)][i])
                        mean_val = float(results["mean_{0}".format(parameter_str)][i])
                        lower_val = float(results["eti_95_lower_{0}".format(parameter_str)][i])
                        upper_val = float(results["eti_95_upper_{0}".format(parameter_str)][i])
                        psrf_val = float(results["psrf_{0}".format(parameter_str)][i])
                        ess_val = float(results["ess_{0}".format(parameter_str)][i])
                        if (psrf_val > max_psrf) or (ess_val < min_ess):
                            bad_x.append(true_val)
                            bad_y.append(mean_val)
                            bad_y_lower.append(lower_val)
                            bad_y_upper.append(upper_val)
                        x.append(true_val)
                        y.append(mean_val)
                        y_lower.append(lower_val)
                        y_upper.append(upper_val)

            assert(len(x) == len(y))
            assert(len(x) == len(y_lower))
            assert(len(x) == len(y_upper))
            assert(len(bad_x) == len(bad_y))
            assert(len(bad_x) == len(bad_y_lower))
            assert(len(bad_x) == len(bad_y_upper))
            proportion_within_ci = pycoevolity.stats.get_proportion_of_values_within_intervals(
                    x,
                    y_lower,
                    y_upper)
            rmse = pycoevolity.stats.root_mean_square_error(x, y)
            ax = plt.subplot(gs[row_index, column_index])
            line = ax.errorbar(
                    x = x,
                    y = y,
                    yerr = get_errors(y, y_lower, y_upper),
                    ecolor = '0.65',
                    elinewidth = 0.5,
                    capsize = 0.8,
                    barsabove = False,
                    marker = 'o',
                    linestyle = '',
                    markerfacecolor = 'none',
                    markeredgecolor = '0.35',
                    markeredgewidth = 0.7,
                    markersize = 2.5,
                    zorder = 100,
                    rasterized = True)
            if bad_x:
                bad_line = ax.errorbar(
                        x = bad_x,
                        y = bad_y,
                        yerr = get_errors(bad_y, bad_y_lower, bad_y_upper),
                        ecolor = highlight_color,
                        elinewidth = 0.5,
                        capsize = 0.8,
                        barsabove = False,
                        marker = 'o',
                        linestyle = '',
                        markerfacecolor = 'none',
                        markeredgecolor = highlight_color,
                        markeredgewidth = 0.7,
                        markersize = 2.5,
                        zorder = 200,
                        rasterized = True)
                for cap_line in bad_line.lines[1]:
                    cap_line.set_alpha(0.3)
                for bar_line in bad_line.lines[2]:
                    bar_line.set_alpha(0.3)
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            identity_line, = ax.plot(
                    [axis_min, axis_max],
                    [axis_min, axis_max])
            plt.setp(identity_line,
                    color = '0.7',
                    linestyle = '-',
                    linewidth = 1.0,
                    marker = '',
                    zorder = 0)
            ax.text(0.02, 0.97,
                    "\\scriptsize\\noindent$p({0:s} \\in \\textrm{{\\sffamily CI}}) = {1:.3f}$".format(
                            parameter_symbol,
                            proportion_within_ci),
                    horizontalalignment = "left",
                    verticalalignment = "top",
                    transform = ax.transAxes,
                    size = 6.0,
                    zorder = 200)
            ax.text(0.02, 0.87,
                    # "\\scriptsize\\noindent$\\textrm{{\\sffamily RMSE}} = {0:.2e}$".format(
                    "\\scriptsize\\noindent RMSE = {0:.2e}".format(
                            rmse),
                    horizontalalignment = "left",
                    verticalalignment = "top",
                    transform = ax.transAxes,
                    size = 6.0,
                    zorder = 200)
            if column_labels and (row_index == 0):
                col_header = column_labels[column_index]
                ax.text(0.5, 1.015,
                        col_header,
                        horizontalalignment = "center",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
            if row_labels and (column_index == (ncols - 1)):
                row_label = row_labels[row_index]
                ax.text(1.015, 0.5,
                        row_label,
                        horizontalalignment = "left",
                        verticalalignment = "center",
                        rotation = 270.0,
                        transform = ax.transAxes)

    # show only the outside ticks
    all_axes = fig.get_axes()
    for ax in all_axes:
        if not ax.is_last_row():
            ax.set_xticks([])
        if not ax.is_first_col():
            ax.set_yticks([])

    # show tick labels only for lower-left plot 
    all_axes = fig.get_axes()
    for ax in all_axes:
        if not ax.is_last_row():
            xtick_labels = ["" for item in ax.get_xticklabels()]
            ax.set_xticklabels(xtick_labels)
        if not ax.is_first_col():
            ytick_labels = ["" for item in ax.get_yticklabels()]
            ax.set_yticklabels(ytick_labels)

    # avoid doubled spines
    all_axes = fig.get_axes()
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
            sp.set_linewidth(2)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        else:
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['right'].set_visible(True)

    fig.text(0.5, 0.001,
            "True {0} (${1}$)".format(parameter_label, parameter_symbol),
            horizontalalignment = "center",
            verticalalignment = "bottom",
            size = x_label_size)
    if y_label is None:
        y_label = "Estimated {0} ($\\hat{{{1}}}$)".format(parameter_label, parameter_symbol)
    fig.text(0.005, 0.5,
            y_label,
            horizontalalignment = "left",
            verticalalignment = "center",
            rotation = "vertical",
            size = y_label_size)

    gs.update(left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_dir = os.path.join(project_util.PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(plot_dir,
            "{0}-scatter.pdf".format(plot_file_prefix))
    plt.savefig(plot_path, dpi=600)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))

def generate_specific_scatter_plot(
        parameters,
        results,
        parameter_label = "divergence time",
        parameter_symbol = "\\tau",
        plot_title = None,
        include_x_label = True,
        include_y_label = True,
        include_rmse = True,
        include_ci = True,
        plot_width = 3.5,
        plot_height = 3.0,
        xy_label_size = 16.0,
        title_size = 16.0,
        pad_left = 0.2,
        pad_right = 0.99,
        pad_bottom = 0.18,
        pad_top = 0.9,
        plot_file_prefix = None,
        variable_only = False):
    _LOG.info("Generating scatter plots for {0}...".format(parameter_label))

    assert(len(parameters) == len(set(parameters)))
    if not plot_file_prefix:
        plot_file_prefix = parameters[0] 

    # Very inefficient, but parsing all results to get min/max for parameter
    parameter_min = float('inf')
    parameter_max = float('-inf')
    for parameter_str in parameters:
        parameter_min = min(parameter_min,
                min(float(x) for x in results["true_{0}".format(parameter_str)]))
        parameter_max = max(parameter_max,
                max(float(x) for x in results["true_{0}".format(parameter_str)]))
        parameter_min = min(parameter_min,
                min(float(x) for x in results["mean_{0}".format(parameter_str)]))
        parameter_max = max(parameter_max,
                max(float(x) for x in results["mean_{0}".format(parameter_str)]))
    axis_buffer = math.fabs(parameter_max - parameter_min) * 0.05
    axis_min = parameter_min - axis_buffer
    axis_max = parameter_max + axis_buffer

    plt.close('all')
    fig = plt.figure(figsize = (plot_width, plot_height))
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)

    x = []
    y = []
    y_upper = []
    y_lower = []
    for parameter_str in parameters:
        x.extend(float(x) for x in results["true_{0}".format(parameter_str)])
        y.extend(float(x) for x in results["mean_{0}".format(parameter_str)])
        y_lower.extend(float(x) for x in results["eti_95_lower_{0}".format(parameter_str)])
        y_upper.extend(float(x) for x in results["eti_95_upper_{0}".format(parameter_str)])

    assert(len(x) == len(y))
    assert(len(x) == len(y_lower))
    assert(len(x) == len(y_upper))
    proportion_within_ci = pycoevolity.stats.get_proportion_of_values_within_intervals(
            x,
            y_lower,
            y_upper)
    rmse = pycoevolity.stats.root_mean_square_error(x, y)
    ax = plt.subplot(gs[0, 0])
    line = ax.errorbar(
            x = x,
            y = y,
            yerr = get_errors(y, y_lower, y_upper),
            ecolor = '0.65',
            elinewidth = 0.5,
            capsize = 0.8,
            barsabove = False,
            marker = 'o',
            linestyle = '',
            markerfacecolor = 'none',
            markeredgecolor = '0.35',
            markeredgewidth = 0.7,
            markersize = 2.5,
            zorder = 100,
            rasterized = True)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    identity_line, = ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max])
    plt.setp(identity_line,
            color = '0.7',
            linestyle = '-',
            linewidth = 1.0,
            marker = '',
            zorder = 0)
    if include_ci:
        ax.text(0.02, 0.97,
                "\\normalsize\\noindent$p({0:s} \\in \\textrm{{\\sffamily CI}}) = {1:.3f}$".format(
                        parameter_symbol,
                        proportion_within_ci),
                horizontalalignment = "left",
                verticalalignment = "top",
                transform = ax.transAxes,
                size = 8.0,
                zorder = 200)
    if include_rmse:
        ax.text(0.02, 0.87,
                "\\normalsize\\noindent RMSE = {0:.2e}".format(
                        rmse),
                horizontalalignment = "left",
                verticalalignment = "top",
                transform = ax.transAxes,
                size = 8.0,
                zorder = 200)
    if include_x_label:
        ax.set_xlabel(
                "True {0} (${1}$)".format(parameter_label, parameter_symbol),
                fontsize = xy_label_size)
    if include_y_label:
        ax.set_ylabel(
                "Estimated {0} ($\\hat{{{1}}}$)".format(parameter_label, parameter_symbol),
                fontsize = xy_label_size)
    root_shape, root_scale = get_root_gamma_parameters(root_shape_setting, root_scale_setting)
    if plot_title:
        ax.set_title(plot_title,
                fontsize = title_size)

    gs.update(
            left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_dir = os.path.join(project_util.PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(plot_dir,
            "{0}-scatter.pdf".format(plot_file_prefix))
    plt.savefig(plot_path, dpi=600)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))


def generate_histograms(
        parameters,
        results_grid,
        column_labels = None,
        row_labels = None,
        parameter_label = "Number of variable sites",
        parameter_discrete = True,
        range_key = "range",
        number_of_digits = 0,
        plot_width = 1.9,
        plot_height = 1.8,
        pad_left = 0.1,
        pad_right = 0.98,
        pad_bottom = 0.12,
        pad_top = 0.92,
        plot_file_prefix = None):
    _LOG.info("Generating histograms for {0}...".format(parameter_label))

    assert(len(parameters) == len(set(parameters)))
    if row_labels:
        assert len(row_labels) ==  len(results_grid)
    if column_labels:
        assert len(column_labels) == len(results_grid[0])

    nrows = len(results_grid)
    ncols = len(results_grid[0])

    if not plot_file_prefix:
        plot_file_prefix = parameters[0] 

    # Very inefficient, but parsing all results to get min/max for parameter
    parameter_min = float('inf')
    parameter_max = float('-inf')
    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            for parameter_str in parameters:
                parameter_min = min(parameter_min,
                        min(float(x) for x in results["{0}".format(parameter_str)]))
                parameter_max = max(parameter_max,
                        max(float(x) for x in results["{0}".format(parameter_str)]))

    axis_buffer = math.fabs(parameter_max - parameter_min) * 0.05
    axis_min = parameter_min - axis_buffer
    axis_max = parameter_max + axis_buffer

    plt.close('all')
    w = plot_width
    h = plot_height
    fig_width = (ncols * w)
    fig_height = (nrows * h)
    fig = plt.figure(figsize = (fig_width, fig_height))
    gs = gridspec.GridSpec(nrows, ncols,
            wspace = 0.0,
            hspace = 0.0)

    hist_bins = None
    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            x = []
            for parameter_str in parameters:
                if parameter_discrete:
                    x.extend(int(x) for x in results["{0}".format(parameter_str)])
                else:
                    x.extend(float(x) for x in results["{0}".format(parameter_str)])

            summary = pycoevolity.stats.get_summary(x)
            _LOG.info("0.025, 0.975 quantiles: {0:.2f}, {1:.2f}".format(
                    summary["qi_95"][0],
                    summary["qi_95"][1]))

            x_range = (parameter_min, parameter_max)
            if parameter_discrete:
                x_range = (int(parameter_min), int(parameter_max))
            ax = plt.subplot(gs[row_index, column_index])
            n, bins, patches = ax.hist(x,
                    # normed = True,
                    weights = [1.0 / float(len(x))] * len(x),
                    bins = hist_bins,
                    range = x_range,
                    cumulative = False,
                    histtype = 'bar',
                    align = 'mid',
                    orientation = 'vertical',
                    rwidth = None,
                    log = False,
                    color = None,
                    edgecolor = '0.5',
                    facecolor = '0.5',
                    fill = True,
                    hatch = None,
                    label = None,
                    linestyle = None,
                    linewidth = None,
                    zorder = 10,
                    )
            if hist_bins is None:
                hist_bins = bins
            ax.text(0.98, 0.98,
                    "\\scriptsize {mean:,.{ndigits}f} ({lower:,.{ndigits}f}--{upper:,.{ndigits}f})".format(
                            # int(round(summary["mean"])),
                            # int(round(summary[range_key][0])),
                            # int(round(summary[range_key][1]))),
                            mean = summary["mean"],
                            lower = summary[range_key][0],
                            upper = summary[range_key][1],
                            ndigits = number_of_digits),
                    horizontalalignment = "right",
                    verticalalignment = "top",
                    transform = ax.transAxes,
                    zorder = 200)

            if column_labels and (row_index == 0):
                col_header = column_labels[column_index]
                ax.text(0.5, 1.015,
                        col_header,
                        horizontalalignment = "center",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
            if row_labels and (column_index == (ncols - 1)):
                row_label = row_labels[row_index]
                ax.text(1.015, 0.5,
                        row_label,
                        horizontalalignment = "left",
                        verticalalignment = "center",
                        rotation = 270.0,
                        transform = ax.transAxes)

    # make sure y-axis is the same
    y_max = float('-inf')
    all_axes = fig.get_axes()
    for ax in all_axes:
        ymn, ymx = ax.get_ylim()
        y_max = max(y_max, ymx)
    for ax in all_axes:
        ax.set_ylim(0.0, y_max)

    # show only the outside ticks
    all_axes = fig.get_axes()
    for ax in all_axes:
        if not ax.is_last_row():
            ax.set_xticks([])
        if not ax.is_first_col():
            ax.set_yticks([])

    # show tick labels only for lower-left plot 
    # all_axes = fig.get_axes()
    # for ax in all_axes:
    #     if ax.is_last_row() and ax.is_first_col():
    #         continue
    #     xtick_labels = ["" for item in ax.get_xticklabels()]
    #     ytick_labels = ["" for item in ax.get_yticklabels()]
    #     ax.set_xticklabels(xtick_labels)
    #     ax.set_yticklabels(ytick_labels)
    all_axes = fig.get_axes()
    for ax in all_axes:
        if not ax.is_last_row():
            xtick_labels = ["" for item in ax.get_xticklabels()]
            ax.set_xticklabels(xtick_labels)
        if not ax.is_first_col():
            ytick_labels = ["" for item in ax.get_yticklabels()]
            ax.set_yticklabels(ytick_labels)

    # avoid doubled spines
    all_axes = fig.get_axes()
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
            sp.set_linewidth(2)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        else:
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['right'].set_visible(True)

    fig.text(0.5, 0.001,
            parameter_label,
            horizontalalignment = "center",
            verticalalignment = "bottom",
            size = 18.0)
    fig.text(0.005, 0.5,
            # "Density",
            "Frequency",
            horizontalalignment = "left",
            verticalalignment = "center",
            rotation = "vertical",
            size = 18.0)

    gs.update(left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_dir = os.path.join(project_util.PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(plot_dir,
            "{0}-histograms.pdf".format(plot_file_prefix))
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))


def generate_model_plots(
        results_grid,
        column_labels = None,
        row_labels = None,
        number_of_comparisons = 5,
        plot_as_histogram = False,
        histogram_correct_values = [],
        plot_width = 1.6,
        plot_height = 1.5,
        pad_left = 0.1,
        pad_right = 0.98,
        pad_bottom = 0.12,
        pad_top = 0.92,
        y_label_size = 18.0,
        y_label = None,
        number_font_size = 12.0,
        plot_file_prefix = None):
    _LOG.info("Generating model plots...")

    cmap = truncate_color_map(plt.cm.binary, 0.0, 0.65, 100)

    if row_labels:
        assert len(row_labels) ==  len(results_grid)
    if column_labels:
        assert len(column_labels) == len(results_grid[0])

    nrows = len(results_grid)
    ncols = len(results_grid[0])

    plt.close('all')
    w = plot_width
    h = plot_height
    fig_width = (ncols * w)
    fig_height = (nrows * h)
    fig = plt.figure(figsize = (fig_width, fig_height))
    gs = gridspec.GridSpec(nrows, ncols,
            wspace = 0.0,
            hspace = 0.0)

    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            true_map_nevents = []
            true_map_nevents_probs = []
            for i in range(number_of_comparisons):
                true_map_nevents.append([0 for i in range(number_of_comparisons)])
                true_map_nevents_probs.append([[] for i in range(number_of_comparisons)])
            true_nevents = tuple(int(x) for x in results["true_num_events"])
            map_nevents = tuple(int(x) for x in results["map_num_events"])
            true_nevents_cred_levels = tuple(float(x) for x in results["true_num_events_cred_level"])
            # true_model_cred_levels = tuple(float(x) for x in results["true_model_cred_level"])
            assert(len(true_nevents) == len(map_nevents))
            assert(len(true_nevents) == len(true_nevents_cred_levels))
            # assert(len(true_nevents) == len(true_model_cred_levels))

            true_nevents_probs = []
            map_nevents_probs = []
            for i in range(len(true_nevents)):
                true_nevents_probs.append(float(
                    results["num_events_{0}_p".format(true_nevents[i])][i]))
                map_nevents_probs.append(float(
                    results["num_events_{0}_p".format(map_nevents[i])][i]))
            assert(len(true_nevents) == len(true_nevents_probs))
            assert(len(true_nevents) == len(map_nevents_probs))

            mean_true_nevents_prob = sum(true_nevents_probs) / len(true_nevents_probs)
            median_true_nevents_prob = pycoevolity.stats.median(true_nevents_probs)

            nevents_within_95_cred = 0
            # model_within_95_cred = 0
            ncorrect = 0
            for i in range(len(true_nevents)):
                true_map_nevents[map_nevents[i] - 1][true_nevents[i] - 1] += 1
                true_map_nevents_probs[map_nevents[i] - 1][true_nevents[i] - 1].append(map_nevents_probs[i])
                if true_nevents_cred_levels[i] <= 0.95:
                    nevents_within_95_cred += 1
                # if true_model_cred_levels[i] <= 0.95:
                #     model_within_95_cred += 1
                if true_nevents[i] == map_nevents[i]:
                    ncorrect += 1
            p_nevents_within_95_cred = nevents_within_95_cred / float(len(true_nevents))
            # p_model_within_95_cred = model_within_95_cred / float(len(true_nevents))
            p_correct = ncorrect / float(len(true_nevents))

            _LOG.info("p(nevents within CS) = {0:.4f}".format(p_nevents_within_95_cred))
            # _LOG.info("p(model within CS) = {0:.4f}".format(p_model_within_95_cred))
            ax = plt.subplot(gs[row_index, column_index])

            if plot_as_histogram:
                total_nevent_estimates = len(map_nevents)
                nevents_indices = [float(x) for x in range(number_of_comparisons)]
                nevents_counts = [0 for x in nevents_indices]
                for k in map_nevents:
                    nevents_counts[k - 1] += 1
                nevents_freqs = [
                        (x / float(total_nevent_estimates)) for x in nevents_counts
                        ]
                assert len(nevents_indices) == len(nevents_freqs)
                bar_width = 0.9
                bar_color = "0.5"
                bars_posterior = ax.bar(
                        nevents_indices,
                        nevents_freqs,
                        bar_width,
                        color = bar_color,
                        label = "MAP")
                x_tick_labels = [str(i + 1) for i in range(number_of_comparisons)]
                plt.xticks(
                        nevents_indices,
                        x_tick_labels
                        )
                annot_x_loc = 0.02
                annot_horizontal_alignment = "left"
                if histogram_correct_values:
                    correct_val = histogram_correct_values[row_index][column_index]
                    correct_line, = ax.plot(
                            [correct_val - 1, correct_val - 1],
                            [0.0, 1.0])
                    plt.setp(correct_line,
                            color = '0.7',
                            linestyle = '--',
                            linewidth = 1.0,
                            marker = '',
                            zorder = 200)
                    if correct_val == 1:
                        annot_x_loc = 0.98
                        annot_horizontal_alignment = "right"
                ax.text(annot_x_loc, 0.99,
                        "\\scriptsize$p(\\hat{{k}} = k) = {0:.3f}$".format(
                                p_correct),
                        horizontalalignment = annot_horizontal_alignment,
                        verticalalignment = "top",
                        transform = ax.transAxes)
                ax.text(annot_x_loc, 0.93,
                        "\\scriptsize$\\widetilde{{p(k|\\mathbf{{D}})}} = {0:.3f}$".format(
                                median_true_nevents_prob),
                        horizontalalignment = annot_horizontal_alignment,
                        verticalalignment = "top",
                        transform = ax.transAxes)
                ax.text(annot_x_loc, 0.87,
                        "\\scriptsize$p(k \\in \\textrm{{\\sffamily CS}}) = {0:.3f}$".format(
                                p_nevents_within_95_cred),
                        horizontalalignment = annot_horizontal_alignment,
                        verticalalignment = "top",
                        transform = ax.transAxes)
            else:
                ax.imshow(true_map_nevents,
                        origin = 'lower',
                        cmap = cmap,
                        interpolation = 'none',
                        aspect = 'auto'
                        # extent = [0.5, 3.5, 0.5, 3.5]
                        )
                for i, row_list in enumerate(true_map_nevents):
                    for j, num_events in enumerate(row_list):
                        ax.text(j, i,
                                str(num_events),
                                horizontalalignment = "center",
                                verticalalignment = "center",
                                size = number_font_size)
                ax.text(0.99, 0.02,
                        "\\scriptsize$p(k \\in \\textrm{{\\sffamily CS}}) = {0:.3f}$".format(
                                p_nevents_within_95_cred),
                        horizontalalignment = "right",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
                ax.text(0.02, 0.98,
                        "\\scriptsize$p(\\hat{{k}} = k) = {0:.3f}$".format(
                                p_correct),
                        horizontalalignment = "left",
                        verticalalignment = "top",
                        transform = ax.transAxes)
                ax.text(0.99, 0.98,
                        "\\scriptsize$\\widetilde{{p(k|\\mathbf{{D}})}} = {0:.3f}$".format(
                                median_true_nevents_prob),
                        horizontalalignment = "right",
                        verticalalignment = "top",
                        transform = ax.transAxes)
            if column_labels and (row_index == 0):
                col_header = column_labels[column_index]
                ax.text(0.5, 1.015,
                        col_header,
                        horizontalalignment = "center",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
            if row_labels and (column_index == (ncols - 1)):
                row_label = row_labels[row_index]
                ax.text(1.015, 0.5,
                        row_label,
                        horizontalalignment = "left",
                        verticalalignment = "center",
                        rotation = 270.0,
                        transform = ax.transAxes)

    all_axes = fig.get_axes()
    safe_to_hardcode_y_tick_labels = False
    if plot_as_histogram:
        for ax in all_axes:
            ax.set_ylim(0.0, 1.0)
            safe_to_hardcode_y_tick_labels = True

    # show only the outside ticks
    for ax in all_axes:
        if not ax.is_last_row():
            ax.set_xticks([])
        if not ax.is_first_col():
            ax.set_yticks([])

    # show tick labels only for lower-left plot 
    all_axes = fig.get_axes()
    for ax in all_axes:
        # Make sure ticks correspond only with number of events
        if not plot_as_histogram:
            ax.xaxis.set_ticks(range(number_of_comparisons))
            ax.yaxis.set_ticks(range(number_of_comparisons))
        if ax.is_last_row():
            if not plot_as_histogram:
                xtick_labels = [item for item in ax.get_xticklabels()]
                for i in range(len(xtick_labels)):
                    xtick_labels[i].set_text(str(i + 1))
                ax.set_xticklabels(xtick_labels)
        else:
            xtick_labels = ["" for item in ax.get_xticklabels()]
            ax.set_xticklabels(xtick_labels)

        if ax.is_first_col():
            if not plot_as_histogram:
                ytick_labels = [item for item in ax.get_yticklabels()]
                for i in range(len(ytick_labels)):
                    ytick_labels[i].set_text(str(i + 1))
                ax.set_yticklabels(ytick_labels)
            else:
                if not ax.is_last_row() and safe_to_hardcode_y_tick_labels:
                    ytick_labels = [item for item in ax.get_yticklabels()]
                    if len(ytick_labels) == 6:
                        ytick_labels = ["", "0.2", "0.4", "0.6", "0.8", "1.0"]
                        ax.set_yticklabels(ytick_labels)
        else:
            ytick_labels = ["" for item in ax.get_yticklabels()]
            ax.set_yticklabels(ytick_labels)

    # avoid doubled spines
    all_axes = fig.get_axes()
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
            sp.set_linewidth(2)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        else:
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)
        else:
            ax.spines['right'].set_visible(True)

    if plot_as_histogram:
        fig.text(0.5, 0.001,
                "Estimated number of events ($\\hat{{k}}$)",
                horizontalalignment = "center",
                verticalalignment = "bottom",
                size = 18.0)
    else:
        fig.text(0.5, 0.001,
                "True number of events ($k$)",
                horizontalalignment = "center",
                verticalalignment = "bottom",
                size = 18.0)
    if y_label is None:
        if plot_as_histogram:
            y_label = "Frequency"
        else:
            y_label = "Estimated number of events ($\\hat{{k}}$)"
    fig.text(0.005, 0.5,
            y_label,
            horizontalalignment = "left",
            verticalalignment = "center",
            rotation = "vertical",
            size = 18.0)

    gs.update(left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_dir = os.path.join(project_util.PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    if plot_file_prefix:
        plot_path = os.path.join(plot_dir,
                "{0}-nevents.pdf".format(plot_file_prefix))
    else:
        plot_path = os.path.join(plot_dir,
                "nevents.pdf")
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))

def generate_specific_model_plots(
        results,
        number_of_comparisons = 5,
        plot_title = None,
        include_x_label = True,
        include_y_label = True,
        include_median = True,
        include_cs = True,
        include_prop_correct = True,
        plot_width = 3.5,
        plot_height = 3.0,
        xy_label_size = 16.0,
        title_size = 16.0,
        pad_left = 0.2,
        pad_right = 0.99,
        pad_bottom = 0.18,
        pad_top = 0.9,
        lower_annotation_y = 0.02,
        upper_annotation_y = 0.92,
        plot_file_prefix = None,
        variable_only = False):
    _LOG.info("Generating model plots...")

    cmap = truncate_color_map(plt.cm.binary, 0.0, 0.65, 100)

    plt.close('all')
    fig = plt.figure(figsize = (plot_width, plot_height))
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)

    true_map_nevents = []
    true_map_nevents_probs = []
    for i in range(number_of_comparisons):
        true_map_nevents.append([0 for i in range(number_of_comparisons)])
        true_map_nevents_probs.append([[] for i in range(number_of_comparisons)])
    true_nevents = tuple(int(x) for x in results["true_num_events"])
    map_nevents = tuple(int(x) for x in results["map_num_events"])
    true_nevents_cred_levels = tuple(float(x) for x in results["true_num_events_cred_level"])
    # true_model_cred_levels = tuple(float(x) for x in results["true_model_cred_level"])
    assert(len(true_nevents) == len(map_nevents))
    assert(len(true_nevents) == len(true_nevents_cred_levels))
    # assert(len(true_nevents) == len(true_model_cred_levels))

    true_nevents_probs = []
    map_nevents_probs = []
    for i in range(len(true_nevents)):
        true_nevents_probs.append(float(
            results["num_events_{0}_p".format(true_nevents[i])][i]))
        map_nevents_probs.append(float(
            results["num_events_{0}_p".format(map_nevents[i])][i]))
    assert(len(true_nevents) == len(true_nevents_probs))
    assert(len(true_nevents) == len(map_nevents_probs))

    mean_true_nevents_prob = sum(true_nevents_probs) / len(true_nevents_probs)
    median_true_nevents_prob = pycoevolity.stats.median(true_nevents_probs)

    nevents_within_95_cred = 0
    model_within_95_cred = 0
    ncorrect = 0
    for i in range(len(true_nevents)):
        true_map_nevents[map_nevents[i] - 1][true_nevents[i] - 1] += 1
        true_map_nevents_probs[map_nevents[i] - 1][true_nevents[i] - 1].append(map_nevents_probs[i])
        if true_nevents_cred_levels[i] <= 0.95:
            nevents_within_95_cred += 1
        # if true_model_cred_levels[i] <= 0.95:
        #     model_within_95_cred += 1
        if true_nevents[i] == map_nevents[i]:
            ncorrect += 1
    p_nevents_within_95_cred = nevents_within_95_cred / float(len(true_nevents))
    # p_model_within_95_cred = model_within_95_cred / float(len(true_nevents))
    p_correct = ncorrect / float(len(true_nevents))

    _LOG.info("p(nevents within CS) = {0:.4f}".format(p_nevents_within_95_cred))
    # _LOG.info("p(model within CS) = {0:.4f}".format(p_model_within_95_cred))
    ax = plt.subplot(gs[0, 0])

    ax.imshow(true_map_nevents,
            origin = 'lower',
            cmap = cmap,
            interpolation = 'none',
            aspect = 'auto'
            )
    for i, row_list in enumerate(true_map_nevents):
        for j, num_events in enumerate(row_list):
            ax.text(j, i,
                    str(num_events),
                    horizontalalignment = "center",
                    verticalalignment = "center")
    if include_cs:
        ax.text(0.98, lower_annotation_y,
                "$p(k \\in \\textrm{{\\sffamily CS}}) = {0:.3f}$".format(
                        p_nevents_within_95_cred),
                horizontalalignment = "right",
                verticalalignment = "bottom",
                transform = ax.transAxes)
    if include_prop_correct:
        ax.text(0.02, upper_annotation_y,
                "$p(\\hat{{k}} = k) = {0:.3f}$".format(
                        p_correct),
                horizontalalignment = "left",
                verticalalignment = "bottom",
                transform = ax.transAxes)
    if include_median:
        ax.text(0.98, upper_annotation_y,
                "$\\widetilde{{p(k|\\mathbf{{D}})}} = {0:.3f}$".format(
                        median_true_nevents_prob),
                horizontalalignment = "right",
                verticalalignment = "bottom",
                transform = ax.transAxes)
    if include_x_label:
        ax.set_xlabel("True \\# of events ($k$)",
                # labelpad = 8.0,
                fontsize = xy_label_size)
    if include_y_label:
        ax.set_ylabel("Estimated \\# of events ($\\hat{{k}}$)",
                labelpad = 8.0,
                fontsize = xy_label_size)
    root_shape, root_scale = get_root_gamma_parameters(root_shape_setting, root_scale_setting)
    if plot_title:
        ax.set_title(plot_title,
                fontsize = title_size)

    # Make sure ticks correspond only with number of events
    ax.xaxis.set_ticks(range(number_of_comparisons))
    ax.yaxis.set_ticks(range(number_of_comparisons))
    xtick_labels = [item for item in ax.get_xticklabels()]
    for i in range(len(xtick_labels)):
        xtick_labels[i].set_text(str(i + 1))
    ytick_labels = [item for item in ax.get_yticklabels()]
    for i in range(len(ytick_labels)):
        ytick_labels[i].set_text(str(i + 1))
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)

    gs.update(
            left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_dir = os.path.join(project_util.PLOT_DIR)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(plot_dir,
            "{0}-nevents.pdf".format(plot_file_prefix))
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))


def parse_results(paths):
    return pycoevolity.parsing.get_dict_from_spreadsheets(
            paths,
            sep = "\t",
            offset = 0)


def main_cli(argv = sys.argv):
    plt.style.use('tableau-colorblind10')
    try:
        os.makedirs(project_util.PLOT_DIR)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    number_of_comparisons = 5
    brooks_gelman_1998_recommended_psrf = 1.2
    min_ess = 200
    highlight_color = "red"

    eco_color = "C0"
    abc_color = "C1"

    independent_sim_names = (
            "fixed-independent-pairs-05-sites-00500-locus-500",
            "fixed-independent-pairs-05-sites-01000-locus-500",
            "fixed-independent-pairs-05-sites-02500-locus-500",
            "fixed-independent-pairs-05-sites-10000-locus-500",
            )
    simultaneous_sim_names = (
            "fixed-simultaneous-pairs-05-sites-00500-locus-500",
            "fixed-simultaneous-pairs-05-sites-01000-locus-500",
            "fixed-simultaneous-pairs-05-sites-02500-locus-500",
            "fixed-simultaneous-pairs-05-sites-10000-locus-500",
            )
    free_sim_names = (
            "pairs-05-sites-00500-locus-500",
            "pairs-05-sites-01000-locus-500",
            "pairs-05-sites-02500-locus-500",
            "pairs-05-sites-10000-locus-500",
            )
    independent_results_grid = [
            [
                parse_results(glob.glob(os.path.join(project_util.SIM_DIR,
                        sim_name,
                        "batch*",
                        "simcoevolity-results.tsv.gz"))
                        ) for sim_name in independent_sim_names
            ],
            [
                parse_results(glob.glob(os.path.join(project_util.PYMSBAYES_DIR,
                        "results",
                        "results-{0}.tsv.gz".format(sim_name)))
                        ) for sim_name in independent_sim_names
            ]
    ]
    simultaneous_results_grid = [
            [
                parse_results(glob.glob(os.path.join(project_util.SIM_DIR,
                        sim_name,
                        "batch*",
                        "simcoevolity-results.tsv.gz"))
                        ) for sim_name in simultaneous_sim_names 
            ],
            [
                parse_results(glob.glob(os.path.join(project_util.PYMSBAYES_DIR,
                        "results",
                        "results-{0}.tsv.gz".format(sim_name)))
                        ) for sim_name in simultaneous_sim_names
            ]
    ]
    free_results_grid = [
            [
                parse_results(glob.glob(os.path.join(project_util.SIM_DIR,
                        sim_name,
                        "batch*",
                        "simcoevolity-results.tsv.gz"))
                        ) for sim_name in free_sim_names
            ],
            [
                parse_results(glob.glob(os.path.join(project_util.PYMSBAYES_DIR,
                        "results",
                        "results-{0}.tsv.gz".format(sim_name)))
                        ) for sim_name in free_sim_names
            ]
    ]
    results = {
            "free" : free_results_grid,
            "independent": independent_results_grid,
            "simultaneous": simultaneous_results_grid,
            }
    column_labels = [
            "1 locus",
            "2 loci",
            "5 loci",
            "20 loci",
    ]
    row_labels = [
            "ecoevolity",
            "dpp-msbayes",
    ]

    for sim_label, results_grid in results.items():
        for results_grid_row in results_grid:
            for results_dict in results_grid_row:
                append_sum_of_mean_errors_column(results_dict, "root_height")
                append_sum_of_mean_errors_column(results_dict, "pop_size_root")
        parameter_symbol = "t"
        generate_scatter_plots(
                parameters = [
                        "root_height_c1sp1",
                        "root_height_c2sp1",
                        "root_height_c3sp1",
                        "root_height_c4sp1",
                        "root_height_c5sp1",
                        ],
                results_grid = results_grid,
                column_labels = column_labels,
                row_labels = row_labels,
                parameter_label = "divergence time",
                parameter_symbol = parameter_symbol,
                max_psrf = brooks_gelman_1998_recommended_psrf,
                min_ess = min_ess,
                highlight_color = highlight_color,
                plot_width = 2.0,
                plot_height = 2.0,
                pad_left = 0.1,
                pad_right = 0.975,
                pad_bottom = 0.15,
                pad_top = 0.95,
                y_label_size = 18.0,
                y_label = "Estimated time ($\\hat{{{0}}}$)".format(parameter_symbol),
                plot_file_prefix = sim_label + "-div-time")
        parameter_symbol = "N_e\\mu"
        generate_scatter_plots(
                parameters = [
                        "pop_size_root_c1sp1",
                        "pop_size_root_c2sp1",
                        "pop_size_root_c3sp1",
                        "pop_size_root_c4sp1",
                        "pop_size_root_c5sp1",
                        ],
                results_grid = results_grid,
                column_labels = column_labels,
                row_labels = row_labels,
                parameter_label = "ancestral population size",
                parameter_symbol = parameter_symbol,
                max_psrf = brooks_gelman_1998_recommended_psrf,
                min_ess = min_ess,
                highlight_color = highlight_color,
                plot_width = 2.0,
                plot_height = 2.0,
                pad_left = 0.11,
                pad_right = 0.975,
                pad_bottom = 0.15,
                pad_top = 0.95,
                y_label_size = 18.0,
                y_label = "Estimated size ($\\hat{{{0}}}$)".format(parameter_symbol),
                plot_file_prefix = sim_label + "-root-pop-size")
        generate_scatter_plots(
                parameters = [
                        "pop_size_c1sp1",
                        "pop_size_c2sp1",
                        "pop_size_c3sp1",
                        "pop_size_c4sp1",
                        "pop_size_c5sp1",
                        "pop_size_c1sp2",
                        "pop_size_c2sp2",
                        "pop_size_c3sp2",
                        "pop_size_c4sp2",
                        "pop_size_c5sp2",
                        ],
                results_grid = results_grid,
                column_labels = column_labels,
                row_labels = row_labels,
                parameter_label = "descendant population size",
                parameter_symbol = parameter_symbol,
                max_psrf = brooks_gelman_1998_recommended_psrf,
                min_ess = min_ess,
                highlight_color = highlight_color,
                plot_width = 2.0,
                plot_height = 2.0,
                pad_left = 0.11,
                pad_right = 0.975,
                pad_bottom = 0.15,
                pad_top = 0.95,
                y_label_size = 18.0,
                y_label = "Estimated size ($\\hat{{{0}}}$)".format(parameter_symbol),
                plot_file_prefix = sim_label + "-leaf-pop-size")

        if sim_label == "independent":
            correct_values = [ [number_of_comparisons for col in row] for row in results_grid ]
            generate_model_plots(
                    results_grid = results_grid,
                    column_labels = column_labels,
                    row_labels = row_labels,
                    number_of_comparisons = number_of_comparisons,
                    plot_as_histogram = True,
                    histogram_correct_values = correct_values,
                    plot_width = 2.5,
                    plot_height = 2.7,
                    pad_left = 0.065,
                    pad_right = 0.975,
                    pad_bottom = 0.12,
                    pad_top = 0.96,
                    y_label_size = 18.0,
                    y_label = "Frequency",
                    number_font_size = 6.0,
                    plot_file_prefix = sim_label)
        elif sim_label == "simultaneous":
            correct_values = [ [1 for col in row] for row in results_grid ]
            generate_model_plots(
                    results_grid = results_grid,
                    column_labels = column_labels,
                    row_labels = row_labels,
                    number_of_comparisons = number_of_comparisons,
                    plot_as_histogram = True,
                    histogram_correct_values = correct_values,
                    plot_width = 2.5,
                    plot_height = 2.7,
                    pad_left = 0.065,
                    pad_right = 0.975,
                    pad_bottom = 0.12,
                    pad_top = 0.96,
                    y_label_size = 18.0,
                    y_label = "Frequency",
                    number_font_size = 6.0,
                    plot_file_prefix = sim_label)
        else:
            generate_model_plots(
                    results_grid = results_grid,
                    column_labels = column_labels,
                    row_labels = row_labels,
                    number_of_comparisons = number_of_comparisons,
                    plot_as_histogram = False,
                    plot_width = 2.5,
                    plot_height = 2.7,
                    pad_left = 0.065,
                    pad_right = 0.975,
                    pad_bottom = 0.12,
                    pad_top = 0.96,
                    y_label_size = 18.0,
                    y_label = "Estimated number ($\\hat{{k}}$)",
                    number_font_size = 6.0,
                    plot_file_prefix = sim_label)

        generate_violin_plots(
                parameters = ["mean_model_distance"],
                results_grid = results_grid,
                comparison_labels = ["ecoevolity", "dpp-msbayes"],
                comparison_colors = [eco_color, abc_color],
                column_labels = column_labels,
                plot_width = 2.0,
                plot_height = 2.4,
                pad_left = 0.06,
                pad_right = 0.98,
                pad_bottom = 0.12,
                pad_top = 0.92,
                y_label = "Mean model distance",
                y_label_size = 14.0,
                x_tick_rotation = None,
                force_shared_y_range = True,
                force_shared_spines = True,
                show_means = True,
                show_sample_sizes = True,
                show_mwu = True,
                plot_file_prefix = sim_label,
                show_legend = False)

        generate_violin_plots(
                parameters = ["sum_of_mean_errors_root_height"],
                results_grid = results_grid,
                comparison_labels = ["ecoevolity", "dpp-msbayes"],
                comparison_colors = [eco_color, abc_color],
                column_labels = column_labels,
                plot_width = 2.0,
                plot_height = 2.4,
                pad_left = 0.08,
                pad_right = 0.98,
                pad_bottom = 0.12,
                pad_top = 0.92,
                y_label = "Divergence time error",
                y_label_size = 14.0,
                x_tick_rotation = None,
                force_shared_y_range = True,
                force_shared_spines = True,
                show_means = True,
                mean_decimals = 4,
                show_sample_sizes = True,
                show_mwu = True,
                plot_file_prefix = f"{sim_label}-div-time-error",
                show_legend = False)

        generate_violin_plots(
                parameters = ["sum_of_mean_errors_pop_size_root"],
                results_grid = results_grid,
                comparison_labels = ["ecoevolity", "dpp-msbayes"],
                comparison_colors = [eco_color, abc_color],
                column_labels = column_labels,
                plot_width = 2.0,
                plot_height = 2.4,
                pad_left = 0.1,
                pad_right = 0.98,
                pad_bottom = 0.12,
                pad_top = 0.92,
                y_label = "Ancestral pop size error",
                y_label_size = 14.0,
                x_tick_rotation = None,
                force_shared_y_range = True,
                force_shared_spines = True,
                show_means = True,
                mean_decimals = 5,
                show_sample_sizes = True,
                show_mwu = True,
                plot_file_prefix = f"{sim_label}-ancestral-pop-size-error",
                show_legend = False)

        # Create violin plots without y-axis labels
        generate_violin_plots(
                parameters = ["mean_model_distance"],
                results_grid = results_grid,
                comparison_labels = ["ecoevolity", "dpp-msbayes"],
                comparison_colors = [eco_color, abc_color],
                column_labels = column_labels,
                plot_width = 2.0,
                plot_height = 2.4,
                pad_left = 0.06,
                pad_right = 0.98,
                pad_bottom = 0.12,
                pad_top = 0.92,
                y_label = None,
                x_tick_rotation = None,
                force_shared_y_range = True,
                force_shared_spines = True,
                show_means = True,
                show_sample_sizes = True,
                show_mwu = True,
                plot_file_prefix = f"{sim_label}-no-y-label",
                show_legend = False)
        generate_violin_plots(
                parameters = ["sum_of_mean_errors_root_height"],
                results_grid = results_grid,
                comparison_labels = ["ecoevolity", "dpp-msbayes"],
                comparison_colors = [eco_color, abc_color],
                column_labels = column_labels,
                plot_width = 2.0,
                plot_height = 2.4,
                pad_left = 0.06,
                pad_right = 0.98,
                pad_bottom = 0.12,
                pad_top = 0.92,
                y_label = None,
                y_label_size = 14.0,
                x_tick_rotation = None,
                force_shared_y_range = True,
                force_shared_spines = True,
                show_means = True,
                mean_decimals = 4,
                show_sample_sizes = True,
                show_mwu = True,
                plot_file_prefix = f"{sim_label}-div-time-error-no-y-label",
                show_legend = False)
        generate_violin_plots(
                parameters = ["sum_of_mean_errors_pop_size_root"],
                results_grid = results_grid,
                comparison_labels = ["ecoevolity", "dpp-msbayes"],
                comparison_colors = [eco_color, abc_color],
                column_labels = column_labels,
                plot_width = 2.0,
                plot_height = 2.4,
                pad_left = 0.06,
                pad_right = 0.98,
                pad_bottom = 0.12,
                pad_top = 0.92,
                y_label = None,
                x_tick_rotation = None,
                force_shared_y_range = True,
                force_shared_spines = True,
                show_means = True,
                mean_decimals = 5,
                show_sample_sizes = True,
                show_mwu = True,
                plot_file_prefix = f"{sim_label}-ancestral-pop-size-error-no-y-label",
                show_legend = False)

        generate_histograms(
                parameters = [
                        "n_var_sites_c1",
                        "n_var_sites_c2",
                        "n_var_sites_c3",
                        ],
                results_grid = results_grid[:1],
                column_labels = column_labels,
                row_labels = None,
                parameter_label = "Number of variable sites",
                parameter_discrete = True,
                range_key = "range",
                number_of_digits = 0,
                plot_width = 2.0,
                plot_height = 2.3,
                pad_left = 0.09,
                pad_right = 0.995,
                pad_bottom = 0.22,
                pad_top = 0.91,
                plot_file_prefix = sim_label + "-number-of-variable-sites")
        generate_histograms(
                parameters = [
                        "ess_sum_ln_likelihood",
                        ],
                results_grid = results_grid[:1],
                column_labels = column_labels,
                row_labels = None,
                parameter_label = "Effective sample size of log likelihood",
                parameter_discrete = False,
                range_key = "range",
                number_of_digits = 0,
                plot_width = 2.0,
                plot_height = 2.3,
                pad_left = 0.09,
                pad_right = 0.995,
                pad_bottom = 0.22,
                pad_top = 0.91,
                plot_file_prefix = sim_label + "-ess-ln-likelihood")
        generate_histograms(
                parameters = [
                        "ess_sum_root_height_c1sp1",
                        "ess_sum_root_height_c2sp1",
                        "ess_sum_root_height_c3sp1",
                        ],
                results_grid = results_grid[:1],
                column_labels = column_labels,
                row_labels = None,
                parameter_label = "Effective sample size of divergence time",
                parameter_discrete = False,
                range_key = "range",
                number_of_digits = 0,
                plot_width = 2.0,
                plot_height = 2.3,
                pad_left = 0.09,
                pad_right = 0.995,
                pad_bottom = 0.22,
                pad_top = 0.91,
                plot_file_prefix = sim_label + "-ess-div-time")
        generate_histograms(
                parameters = [
                        "ess_sum_pop_size_root_c1sp1",
                        "ess_sum_pop_size_root_c2sp1",
                        "ess_sum_pop_size_root_c3sp1",
                        ],
                results_grid = results_grid[:1],
                column_labels = column_labels,
                row_labels = None,
                parameter_label = "Effective sample size of ancestral population size",
                parameter_discrete = False,
                range_key = "range",
                number_of_digits = 0,
                plot_width = 2.0,
                plot_height = 2.3,
                pad_left = 0.09,
                pad_right = 0.995,
                pad_bottom = 0.22,
                pad_top = 0.91,
                plot_file_prefix = sim_label + "-ess-root-pop-size")
        generate_histograms(
                parameters = [
                        "psrf_ln_likelihood",
                        ],
                results_grid = results_grid[:1],
                column_labels = column_labels,
                row_labels = None,
                parameter_label = "PSRF of log likelihood",
                parameter_discrete = False,
                range_key = "range",
                number_of_digits = 3,
                plot_width = 2.0,
                plot_height = 2.3,
                pad_left = 0.09,
                pad_right = 0.995,
                pad_bottom = 0.22,
                pad_top = 0.91,
                plot_file_prefix = sim_label + "-psrf-ln-likelihood")
        generate_histograms(
                parameters = [
                        "psrf_root_height_c1sp1",
                        "psrf_root_height_c2sp1",
                        "psrf_root_height_c3sp1",
                        ],
                results_grid = results_grid[:1],
                column_labels = column_labels,
                row_labels = None,
                parameter_label = "PSRF of divergence time",
                parameter_discrete = False,
                range_key = "range",
                number_of_digits = 3,
                plot_width = 2.0,
                plot_height = 2.3,
                pad_left = 0.09,
                pad_right = 0.995,
                pad_bottom = 0.22,
                pad_top = 0.91,
                plot_file_prefix = sim_label + "-psrf-div-time")

        # plot_ess_versus_error(
        #         parameters = [
        #                 "root_height_c1sp1",
        #                 "root_height_c2sp1",
        #                 "root_height_c3sp1",
        #                 ],
        #         results_grid = results_grid[:1],
        #         column_labels = column_labels,
        #         row_labels = None,
        #         parameter_label = "divergence time",
        #         plot_file_prefix = sim_label + "-div-time")

    # plot_nevents_estimated_vs_true_probs(
    #         nevents = 1,
    #         sim_dir = "03pairs-dpp-root-0100-100k",
    #         nbins = 5,
    #         plot_file_prefix = "100k-sites")


if __name__ == "__main__":
    main_cli()
