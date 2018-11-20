#!python3
# project_utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from brian2 import *
from brian2tools import *
import datetime
import time
import os


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
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
    subplot(133)
    plt.hist(S.w_syn, 10, color='k', edgecolor='w')


def multipage(filename=None, figs=None, dpi=200):
    if filename is None:
        filename = 'all_figures_%s' % time.strftime('%y%m%d-%H%M')
    file_path = os.path.join('figures', filename + '.pdf')
    with PdfPages(file_path) as pp:
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf', dpi=dpi)
