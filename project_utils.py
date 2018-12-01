#!python3
# project_utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.tsa.stattools import grangercausalitytests as granger
from brian2 import *
from brian2tools import *
from scipy.stats.stats import pearsonr
import datetime
import time
import os


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    corr, p_value = pearsonr(S.i, S.j)
    figure(figsize=(14, 4))
    subplot(131)
    plot(zeros(Ns), arange(Ns), 'ok', ms=3)
    plot(ones(Nt), arange(Nt), 'ok', ms=3)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k', lw=0.25)
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(132)
    max_weight = np.max(np.abs(np.array(S.w_syn).flatten()))
    scatter(S.i, S.j, s=((S.w_syn / max_weight) * 2.5) ** 2, c='k')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    title(" corr=" + '%.2f' % corr + ", p-value=" + '%.2f' % p_value)
    subplot(133)
    plt.hist(S.w_syn, 10, color='k', edgecolor='w')


def plot_correlations(nc, src, target, s, nme):
    a, b = src, target
    all_vals = np.concatenate((a.flatten(), b.flatten()))
    lims = [np.min(all_vals), np.max(all_vals)]
    lim_diff = lims[1] - lims[0]
    lim_prct = 0.07
    lim_delta = lim_prct * lim_diff
    lims = [lims[0] - lim_delta, lims[1] + lim_delta]
    fig, axs = plt.subplots(nc, nc, figsize=(10, 10))
    plt.tight_layout(2.5, 0.02, 0.02)
    fig.suptitle(nme)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, i_row in enumerate(reversed(range(nc))):
        for j in range(nc):
            ax = axs[i_row, j]
            R = np.corrcoef(a[j].flatten(), b[i].flatten())[1, 0]
            scatter_kwargs = dict(x=a[j], y=b[i], s=3, facecolors='none',
                                  edgecolors=(0, 0, 0, 0.2), marker='o',
                                  linewidths=0.5)
            text_kwargs = dict(x=lims[0] + 5, y=lims[1] - 5, s=' %.2f' % R,
                               color='k', verticalalignment='top',
                               horizontalalignment='left')
            if True in [i_val == j for i_val, j_val in zip(s.i, s.j)
                        if j_val == i]:
                ax.set_facecolor('k')
                scatter_kwargs['edgecolors'] = (1, 1, 1, 0.2)
                text_kwargs['color'] = 'w'
            ax.scatter(**scatter_kwargs)
            ax.text(**text_kwargs)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            if i_row == (nc - 1) and j == 0:
                ax.set_xlabel('Source Neuron')
                ax.set_ylabel('Target Neuron')
            else:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)


def multipage(filename=None, figs=None, dpi=200, fmt='pdf'):
    if filename is None:
        filename = 'all_figures_%s' % time.strftime('%y%m%d-%H%M')

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]

    file_path = os.path.join('figures', filename)
    if fmt == 'pdf':
        with PdfPages(file_path) as pp:
            file_path += '.pdf'
            for fig in figs:
                fig.savefig(pp, format='pdf', dpi=dpi)
    else:

        for i, fig in enumerate(figs):
            fig.savefig('%s_%d.%s' % (file_path, i, fmt), format=fmt, dpi=dpi)

# Pick the lag that is more significant (higher p-value)
def grangertests(v1, v2, maxlag=3):
    g = granger([*zip(*[v1, v2]/mV)], maxlag, verbose=True)
    print(g)
    return g
