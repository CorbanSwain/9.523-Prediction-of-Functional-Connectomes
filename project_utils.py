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
from PIL import Image
import pprint as pp
import multiprocessing
import pickle
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches
import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors


class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def save_obj(obj, pth):
    with open(pth + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(pth):
    extension = '.pkl'
    pth = pth + ('' if pth.endswith(extension) else extension)
    with open(pth, 'rb') as f:
        return pickle.load(f)


def touchdir(pth):
    try:
        os.mkdir(pth)
    except FileExistsError:
        pass

def despine_all(ax):
    despine(ax, **{pos: True for pos in ('left', 'right', 'top', 'bottom')})

def despine(ax, **kwargs):
    [ax.spines[k].set_visible(not v) for k, v in kwargs.items()]

def add_peaks(v, spike_monitor, v_peak):
    v_with_peaks = v
    for i_neuron, spike_times in spike_monitor.all_values()['t'].items():
        for spike_time in spike_times:
            i_time = int(spike_time / defaultclock.dt)
            v_with_peaks[i_neuron, i_time] = v_peak
    return v_with_peaks


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
    max_weight = np.max(np.abs(np.array(S.w_syn_sign).flatten()))
    scatter(S.i, S.j, s=((S.w_syn_sign / max_weight) * 2.5) ** 2, c='k')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    title(" corr=" + '%.2f' % corr + ", p-value=" + '%.2f' % p_value)
    subplot(133)
    plt.hist(S.w_syn, 10, color='k', edgecolor='w')


def plot_correlations(nc, src, target, s, compare_test='correlation', connectivity=None):
    a, b = src, target
    all_vals = np.concatenate((a.flatten(), b.flatten()))
    lims = [np.min(all_vals), np.max(all_vals)]
    lim_diff = lims[1] - lims[0]
    lim_prct = 0.07
    lim_delta = lim_prct * lim_diff
    lims = [lims[0] - lim_delta, lims[1] + lim_delta]
    fig = plt.figure(figsize=(10, 10))
    gs_factor = 3
    gs = plt.GridSpec(nc * gs_factor + 1, nc * gs_factor + 1, figure=fig,
                      wspace=0, hspace=0)

    normed_map2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1, vmax=1),
                                        cmap=mpl.cm.RdBu)
    normed_map1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1, vmax=1),
                                        cmap=mpl.cm.RdBu)
    for i in range(nc + 1):
        for j in range(nc + 1):
            if i == nc * gs_factor / 2 or j == nc * gs_factor / 2:
                ax = plt.subplot(gs[i, j])
                ax.set_axis_off()
    match_count = 0
    n = 0
    for i, i_row in enumerate(reversed(range(nc))):
        for j in range(nc):
            n += 1
            print('%s Calculator Cycle # %d / %d' % (compare_test, n, nc ** 2))
            if i_row >= nc / 2:
                ax_row_idx = 1
            else:
                ax_row_idx = 0
            if j >= nc / 2:
                ax_col_idx = 1
            else:
                ax_col_idx = 0

            ax = plt.subplot(gs[(i_row * gs_factor + ax_row_idx):((i_row + 1) * gs_factor + ax_row_idx),
                             (j * gs_factor + ax_col_idx):((j + 1) * gs_factor + ax_col_idx)])
            try:
                w = float(s.w[np.logical_and(s.i == j, s.j == i)]) / 3
            except TypeError:
                w = 0
            if compare_test == 'correlation':
                R = np.corrcoef(a[j].flatten(), b[i].flatten())[1, 0]
                metric = (R, R)
                text2_args = ['\\textbf{R = %.2f}' % metric[1]]
                if (metric[0] ** 2 > 0.75 and w > 0.1) or (metric[0] ** 2 < 0.75 and w < 0.1):
                    match_count += 1
            elif compare_test == 'granger':
                gg = np.stack((b[i], a[j]), 1).T / mV
                G = granger(gg.T, maxlag=20, verbose=False)
                all_ps = []
                for k_gr, v_gr in G.items():
                    all_ps.append(v_gr[0]['ssr_ftest'][0])
                all_ps = np.array(all_ps)
                metric = (np.max(all_ps) / 300, np.max(all_ps))
                text2_args = ['\\textbf{F = %.1f}' % metric[1]]
                if (metric[0] > 0.035 and np.abs(w) > 0.1) or (metric[0] < 0.035 and np.abs(w) < 0.1):
                    match_count += 1
            scatter_kwargs = dict(x=a[j], y=b[i], s=3, facecolors='none',
                                  edgecolors=(0, 0, 0, 0.2), marker='o',
                                  linewidths=0.5)
            text1_args = ['\\textbf{w = %.2f}' % w]
            text1_kwargs = dict(loc='lower right',
                                frameon=False,
                                pad=0.09,
                                prop=dict(color='k',
                                          fontsize='xx-small'
                                          ),
                                )

            text2_kwargs = dict(text1_kwargs)
            text2_kwargs['loc'] = 'upper left'

            pt1 = mpl.patches.Polygon(np.array([[1, 1], [0, 0], [1, 0]]),
                                      transform=ax.transAxes,
                                      facecolor=normed_map1.to_rgba(w),
                                      edgecolor='none',
                                      alpha=0.6,
                                      zorder=0.5
                                      )
            pt2 = mpl.patches.Polygon(np.array([[1, 1], [0, 0], [0, 1]]),
                                      transform=ax.transAxes,
                                      facecolor=normed_map2.to_rgba(metric[0]),
                                      edgecolor='none',
                                      alpha=0.6,
                                      zorder=0.4
                                      )
            ax.add_artist(pt1)
            ax.add_artist(pt2)
            if compare_test == 'correlation':
                ax.scatter(**scatter_kwargs)
            ax.add_artist(AnchoredText(*text1_args, **text1_kwargs))
            ax.add_artist(AnchoredText(*text2_args, **text2_kwargs))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    ax = fig.add_subplot(gs[:, :])
    ax.patch.set_alpha(0)
    despine_all(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Source Neuron', fontsize='large', labelpad=30)
    ax.set_ylabel('Target Neuron', fontsize='large', labelpad=30)

    gap_size = 1 / (gs_factor * nc + 1)
    low_middle = 0.5 - gap_size / 2
    high_middle = 0.5 + gap_size / 2
    thickness = -0.03
    cset1 = ('xkcd:golden yellow', 'k')
    cset2 = ('k', 'w')

    if connectivity in ('e_to_e', 'e_to_i', 'all'):
        color = cset1
    else:
        color = cset2
    pt = mpl.patches.Rectangle((0, -0.01),
                               width=low_middle,
                               height=thickness,
                               transform=ax.transAxes,
                               facecolor=color[0],
                               edgecolor='k',
                               clip_path=None,
                               clip_on=False)
    ax.text(low_middle/2, -0.01 + thickness / 2, '$E$', color=color[1], fontsize='large', verticalalignment='center',
            horizontalalignment='center')
    ax.add_artist(pt)

    if connectivity in ('i_to_e', 'i_to_i', 'all'):
        color = cset1
    else:
        color = cset2
    pt = mpl.patches.Rectangle((high_middle, -0.01),
                               width=low_middle,
                               height=thickness,
                               transform=ax.transAxes,
                               facecolor=color[0],
                               edgecolor='k',
                               clip_path=None,
                               clip_on=False)
    ax.text(low_middle/2 + high_middle, -0.01 + thickness / 2, '$I$', color=color[1], fontsize='large', verticalalignment='center',
            horizontalalignment='center')
    ax.add_artist(pt)

    if connectivity in ('i_to_e', 'e_to_e', 'all'):
        color = cset1
    else:
        color = cset2
    pt = mpl.patches.Rectangle((-0.01, 0),
                               width=thickness,
                               height=low_middle,
                               transform=ax.transAxes,
                               facecolor=color[0],
                               edgecolor='k',
                               clip_path=None,
                               clip_on=False)
    ax.text(-0.01 + thickness / 2, low_middle/2, '$E$', color=color[1], fontsize='large', verticalalignment='center',
            horizontalalignment='center')
    ax.add_artist(pt)

    if connectivity in ('i_to_i', 'e_to_i', 'all'):
        color = cset1
    else:
        color = cset2
    pt = mpl.patches.Rectangle((-0.01, high_middle),
                               width=thickness,
                               height=low_middle,
                               transform=ax.transAxes,
                               facecolor=color[0],
                               edgecolor='k',
                               clip_path=None,
                               clip_on=False)
    ax.text(-0.01 + thickness / 2, low_middle/2 + high_middle,  '$I$', color=color[1], fontsize='large', verticalalignment='center',
            horizontalalignment='center')
    ax.add_artist(pt)
    return match_count

def multipage(filename=None, figs=None, dpi=200, fmt='pdf'):
    if filename is None:
        filename = 'all_figures'
    filename = filename + '_' + time.strftime('%y%m%d-%H%M')

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]

    file_path = os.path.join('figures', filename)
    if fmt == 'pdf':
        file_path += '.pdf'
        with PdfPages(file_path) as pp:
            for fig in figs:
                fig.savefig(pp, format='pdf', dpi=dpi)
    else:
        for i, fig in enumerate(figs):
            fig.savefig('%s_%d.%s' % (file_path, i, fmt), format=fmt, dpi=dpi)


def plot_traces(t, v, ax):
    max_delta = np.max(v.flatten()) - np.min(v.flatten())
    x = t.T
    space_idx = np.arange(0, len(v)).astype(float)
    space_idx[space_idx >= (len(v) / 2)] += 1.5
    delta_factor = 1.2
    y = (v + (max_delta * delta_factor) * np.reshape(space_idx, (-1, 1))).T
    p = ax.plot(x / ms, y / mV, 'k-')
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    return p, (max_delta * delta_factor, space_idx)


def make_training_image_pair(t, v, s, directory='training_data', idx=0, px_dim=500, do_shuffle=False):
    touchdir(directory)
    fig_sz = 5
    x = t.T / ms
    n_neurons = len(t)
    gap = 190
    baseline = -155
    if do_shuffle:
        np.random.shuffle(v)
    y = (v / mV + gap * np.reshape(np.arange(0, n_neurons), (-1, 1))).T
    fig = plt.figure(figsize=(fig_sz, fig_sz))
    gs = plt.GridSpec(1, 1, left=0, right=1, bottom=0, top=1, figure=fig)
    ax = plt.subplot(gs[0, 0])
    ax.plot(x, y, 'w-')
    ax.set_xlim(x[0, 0], x[-1, 0])
    ax.set_ylim(baseline, baseline + gap * n_neurons)
    ax.axis('off')
    pth = os.path.join(directory, 'trace_%d.png' % idx)
    fig.patch.set_facecolor('k')
    fig.savefig(pth, dpi=px_dim/fig_sz, pad_inches=0, facecolor='k')
    plt.close(fig)
    img = Image.open(pth).convert('L')
    os.remove(pth)
    pth = os.path.splitext(pth)[0] + '.jpg'
    img.save(pth)

    # Connection plot (OUTPUT for ML)
    # x = s.i
    # y = s.j
    # sz = 30 / (n_neurons / 10)
    # c = ['k' if x == -1 else 'w' for x in s.w_syn_sign]
    # fig = plt.figure(figsize=(fig_sz, fig_sz))
    # gs = plt.GridSpec(1, 1, left=0, right=1, bottom=0, top=1, figure=fig)
    # ax = plt.subplot(gs[0, 0])
    # face_c = (0.5, 0.5, 0.5)
    # ax.scatter(x, y, s=sz ** 2, c=c, marker='s')
    # ax.set_xlim(-0.5, n_neurons + 0.5)
    # ax.set_ylim(-0.5, n_neurons + 0.5)
    # ax.axis('off')
    # pth = os.path.join(directory, 'connect_%d.png' % idx)
    # fig.patch.set_facecolor(face_c)
    # fig.savefig(pth, dpi=n_neurons / fig_sz, pad_inches=0, facecolor=face_c)
    # # img = Image.open(pth).convert('L')
    # # os.remove(pth)
    # # pth = os.path.splitext(pth)[0] + '.jpg'
    # # img.save(pth)
    #
    # out_arr = np.ones((n_neurons, n_neurons)) * 0.5
    # out_arr[s.j, s.i] = [0 if x == -1 else 1 for x in s.w_syn_sign]
    # out_arr = (out_arr * (2 ** 8)).astype(np.uint8)
    # img = Image.fromarray(out_arr)
    # pth = os.path.join(directory, 'connect_%d.jpg' % idx)
    # img.save(pth)

    # plt.show()


def misc():
    traces = []
    for st_m, sp_m in zip(state_monitors, spike_monitors):
        V = st_m.vm
        for ky, vl in sp_m.all_values()['t'].items():
            for t in vl:
                V[ky, int(t / defaultclock.dt)] = Vpeak
        traces.append(V)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for ax, m, nme, tr in zip(axs, state_monitors, ng_names, traces):
        x = np.array([m.t for _ in tr]).T / (1 * ms)
        max_delta = np.max(tr.flatten()) - np.min(tr.flatten())
        y = (tr + (max_delta * 1.1) * np.reshape(np.arange(0, len(tr)),
                                                 (-1, 1))).T
        ax.plot(x, y / mV, 'k-')
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.set_title(nme)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for ax, m, nme in zip(axs, state_monitors, ng_names):
        x = np.array([m.t for _ in m.I]).T / (1 * ms)
        max_delta = np.max(m.I.flatten()) - np.min(m.I.flatten())
        y = (m.I + (max_delta * 1.1) * np.reshape(np.arange(0, len(m.I)),
                                                  (-1, 1))).T
        ax.plot(x, y / nA, 'k-')
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.set_title(nme)

    # plot_correlations(NC, src=(me_v / mV), target=(me_v / mV),
    #                   nme='E -> E', s=s_ee)
    # plot_correlations(NC, src=(mi_v / mV), target=(mi_v / mV),
    #                   nme='I -> I', s=s_ii)
    # plot_correlations(NC, src=(mi_v / mV), target=(me_v / mV),
    #                   nme='I -> E', s=s_ei)
    # plot_correlations(NC, src=(me_v / mV), target=(mi_v / mV),
    #                   nme='E -> I', s=s_ie)

    # fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    # plt.tight_layout()
    # for i in range(5):
    #     for j in range(5):
    #         axs[i, j].scatter(me_v[i], mi_v[j], c='k', marker='o')
    #         corr, p_value = pearsonr(me_v[i], mi_v[j])
    #         axs[i, j].set_title("r=" + '%.2f' % corr)
    #         grangertests(me_v[i], mi_v[j])
    #
    # fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    # plt.tight_layout()
    # for i in range(5):
    #     for j in range(5):
    #         axs[i, j].scatter(me_v[i], me_v[j], c='k', marker='o')
    #         corr, p_value = pearsonr(me_v[i], mi_v[j])
    #         axs[i, j].set_title("r=" + '%.2f' % corr)
    #         grangertests(me_v[i], mi_v[j])

# Pick the lag that is more significant (higher p-value)
def grangertests(v1, v2, maxlag=3):
    g = granger([*zip(*[v1, v2]/mV)], maxlag, verbose=True)
    print(g)
    return g


# parmap implementation for functions with non-trivial execution time
def funmap(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))
    return


def parmap(f, X, nprocs=(multiprocessing.cpu_count())):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()
    proc = [multiprocessing.Process(target=funmap, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(X))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]
