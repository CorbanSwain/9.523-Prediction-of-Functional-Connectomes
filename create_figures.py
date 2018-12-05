#!python3
# create_figures.py

from cortical_column_model import *


def figure_1():
    print('F1 - Beginning unconnected simulation.')
    theta_0_expr = 'int(t > (1000.0 * ms)) * (pi * (-1.0 / 2.0 + (t - 1000.0 * ms) / (4000.0 * ms)))'
    i_switch_expr = 'int(abs(t - (750.0 * ms)) > (250.0 * ms))'
    (t, v, I, theta_0), _ = run_cortical_model(duration=5000*ms,
                                               num_columns=5,
                                               connection_probability=0,
                                               theta_0_expr=theta_0_expr,
                                               i_switch_expr=i_switch_expr)
    print('F1 - Making plots.')
    fig = plt.figure(figsize=(6.5, 4.5))
    gs = GridSpec(5, 1, hspace=0.4)
    ax1 = plt.subplot(gs[:-1, :])
    trace_plot, (trace_delta, space_idx) = plot_traces(t, v, ax1)
    trace_delta = trace_delta / mV
    n_neurons = len(space_idx)
    rect_excit = patches.Rectangle((5100, -80),
                                   width=100,
                                   height=(trace_delta * n_neurons / 2) + 0,
                                   facecolor='k', edgecolor=None)

    rect_inhib = patches.Rectangle((5100, -80 + trace_delta * space_idx[int(n_neurons / 2)]),
                                   width=100,
                                   height=(trace_delta * n_neurons / 2) + 0,
                                   facecolor='k', edgecolor=None)
    pc = collections.PatchCollection([rect_excit, rect_inhib])

    ax1.add_patch(rect_excit)
    ax1.add_patch(rect_inhib)

    ax1.text(5260, -80 + (trace_delta * n_neurons / 2) / 2, '$E$',
             verticalalignment='center', fontweight='bold')
    ax1.text(5260,
             -80 + (trace_delta * n_neurons / 2) / 2 + trace_delta * space_idx[int(n_neurons / 2)],
             '$I$', verticalalignment='center', fontweight='bold')
    ax1.set_xlim(left=-100, right=5300)
    ax1.set_yticks([-70, 20])
    ax1.set_yticklabels(['$-70$ mV', '$20$ mV'])
    ax1.set_xticks([])
    ax1.tick_params(axis='y', which='major', pad=2)
    despine(ax1, right=True, bottom=True, top=True)
    ax1.spines['left'].set_bounds(-70, 20)

    ax2 = plt.subplot(gs[-1, :])
    ax2.hlines(0, -500, 5000, color='k', linestyle='--', linewidth=0.5)
    x = t[1, :] / ms
    y = theta_0[1, :]
    y[np.logical_and(x <= 1000, x > 500)] = np.nan
    ax2.plot(x, y, 'r-', linewidth=3)
    ax2.set_yticks([-pi/2, 0, pi/2])
    ax2.set_yticklabels([r'$-\nicefrac{\pi}{2}$',
                         r'$0$',
                         r'$+\nicefrac{\pi}{2}$'])
    ax2.set_xticks([0, 5000])
    ax2.set_xticklabels(['$0$ s', '$5$ s'])
    ax2.set_xlim(ax1.get_xlim())
    ax2.tick_params(axis='y', which='major', pad=2)
    ax2.set_ylabel(r'$\theta_0$')
    ax2.set_xlabel(r'$t$')
    despine(ax2, right=True, top=True)
    ax2.spines['left'].set_bounds(-np.pi/2, np.pi/2)
    ax2.spines['bottom'].set_bounds(0, 5000)


def figure_2():
    print('F2 - Beginning simulation.')
    (t, v, I, theta_0), _ = run_cortical_model(duration=1000*ms,
                                               connection_probability=0.5)
    print('F2 - Making plots.')


if __name__ == "__main__":
    matplotlib.rcdefaults()
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('text.latex', preamble=r'''
    \usepackage{sansmathfonts}
    \usepackage{helvet}
    \renewcommand{\rmdefault}{\sfdefault}
    \usepackage{units}''')

    figure_1()
    figure_2()