#!python3
# create_figures.py

from cortical_column_model import *
import matplotlib as mpl
import matplotlib.tight_layout


def make_assment_plot(t, v, I, theta_0, ax1=None, ax2=None):
    t_tot = t[0, -1] / ms
    if ax1:
        trace_plot, (trace_delta, space_idx) = plot_traces(t, v, ax1)
        trace_delta = trace_delta / mV
        n_neurons = len(space_idx)
        rect_excit = patches.Rectangle((t_tot * 1.02, -80),
                                       width=100,
                                       height=(trace_delta * n_neurons / 2) + 0,
                                       facecolor='k', edgecolor=None)

        rect_inhib = patches.Rectangle((t_tot * 1.02, -80 + trace_delta * space_idx[int(n_neurons / 2)]),
                                       width=100,
                                       height=(trace_delta * n_neurons / 2) + 0,
                                       facecolor='k', edgecolor=None)
        pc = collections.PatchCollection([rect_excit, rect_inhib])

        ax1.add_patch(rect_excit)
        ax1.add_patch(rect_inhib)

        ax1.text(t_tot * 1.02 + 160, -80 + (trace_delta * n_neurons / 2) / 2, '$E$',
                 verticalalignment='center', fontweight='bold')
        ax1.text(t_tot * 1.02 + 160,
                 -80 + (trace_delta * n_neurons / 2) / 2 + trace_delta * space_idx[int(n_neurons / 2)],
                 '$I$', verticalalignment='center', fontweight='bold')
        ax1.set_xlim(left=-100, right=t_tot * 1.06)
        ax1.set_yticks([-70, 20])
        ax1.set_yticklabels(['$-70$ mV', '$20$ mV'])
        ax1.set_xticks([])
        ax1.tick_params(axis='y', which='major', pad=2)
        despine(ax1, right=True, bottom=True, top=True)
        ax1.spines['left'].set_bounds(-70, 20)

    if ax2:
        ax2.hlines(0, -500, t_tot, color='k', linestyle='--', linewidth=0.5)
        x = t[1, :] / ms
        y = theta_0[1, :]
        y[np.logical_and(x <= 1000, x > 500)] = np.nan
        ax2.plot(x, y, 'r-', linewidth=3)
        ax2.set_yticks([-pi / 2, 0, pi / 2])
        ax2.set_yticklabels([r'$-\nicefrac{\pi}{2}$',
                             r'$0$',
                             r'$+\nicefrac{\pi}{2}$'])
        ax2.set_xticks([0, t_tot])
        ax2.set_xticklabels(['$0$ s', '$5$ s'])
        ax2.set_xlim(ax1.get_xlim())
        ax2.tick_params(axis='y', which='major', pad=2)
        ax2.set_ylabel(r'$\theta_0$')
        ax2.set_xlabel(r'$t$')
        despine(ax2, right=True, top=True)
        ax2.spines['left'].set_bounds(-np.pi / 2, np.pi / 2)
        ax2.spines['bottom'].set_bounds(0, t_tot)

def figure_1():
    print('F1 - Beginning unconnected simulation.')
    theta_0_expr = 'int(t > (1000.0 * ms)) * (pi * (-1.0 / 2.0 + (t - 1000.0 * ms) / (1000.0 * ms)))'
    i_switch_expr = 'int(abs(t - (750.0 * ms)) > (250.0 * ms))'
    (t, v, I, theta_0), _ = run_cortical_model(duration=2000*ms,
                                               num_columns=5,
                                               connection_probability=0,
                                               theta_0_expr=theta_0_expr,
                                               i_switch_expr=i_switch_expr)
    print('F1 - Making plots.')
    fig = plt.figure(figsize=(6.5, 4.5))
    gs = GridSpec(5, 1, hspace=0.4)
    ax1 = plt.subplot(gs[:-1, :])
    ax2 = plt.subplot(gs[-1, :])
    make_assment_plot(t, v, I, theta_0, ax1=ax1, ax2=ax2)

    return fig


def figure_2():
    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(9, 2, hspace=0.4)

    print('F2 - Beginning simulation.')
    theta_0_expr = 'int(t > (1000.0 * ms)) * (pi * (-1.0 / 2.0 + (t - 1000.0 * ms) / (1000.0 * ms)))'
    i_switch_expr = 'int(abs(t - (750.0 * ms)) > (250.0 * ms))'
    model_kwargs = dict(
        duration=2000 * ms,
        num_columns=5,
        do_conditionally_connect=False,
        theta_0_expr=theta_0_expr,
        i_switch_expr=i_switch_expr,
        theta_noise_sigma=0,
        C_alpha_0=0.9 * nA
    )

    model_kwargs['synapses_allowed'] = 'e_to_e'
    (t, v, I, theta_0), _ = run_cortical_model(**model_kwargs)
    ax1 = plt.subplot(gs[:4, 0])
    make_assment_plot(t, v, I, theta_0, ax1=ax1)

    model_kwargs['synapses_allowed'] = 'e_to_i'
    (t, v, I, theta_0), _ = run_cortical_model(**model_kwargs)
    ax1 = plt.subplot(gs[4:-1, 0])
    ax2 = plt.subplot(gs[-1, 0])
    make_assment_plot(t, v, I, theta_0, ax1=ax1, ax2=ax2)

    model_kwargs['synapses_allowed'] = 'i_to_e'
    (t, v, I, theta_0), _ = run_cortical_model(**model_kwargs)
    ax1 = plt.subplot(gs[:4, 1])
    make_assment_plot(t, v, I, theta_0, ax1=ax1)

    model_kwargs['synapses_allowed'] = 'i_to_i'
    (t, v, I, theta_0), _ = run_cortical_model(**model_kwargs)
    ax1 = plt.subplot(gs[4:-1, 1])
    ax2 = plt.subplot(gs[-1, 1])
    make_assment_plot(t, v, I, theta_0, ax1=ax1, ax2=ax2)
    return fig


def figure_4():
    path_fmt = lambda s: os.path.join('training_output',
                                      'train_181206-0217_%s' % s)


    x = np.zeros((21, 4))
    y_acc = np.zeros((21, 4))
    y_cx = np.zeros((21, 4))
    label_acc_t = np.zeros((6, ))
    label_acc_st = np.zeros((6, ))
    label_miss_t = np.zeros((6, 6))
    label_miss_st = np.zeros((6, 6))
    n_t = np.zeros((6, ))
    n_st = np.zeros((6, ))
    for i, experiment_name in enumerate(('traces', 'negative_control',
                                     'shuffled_traces',
                                     'shuffled_negative_control')):
        path = path_fmt(experiment_name)
        cache = load_obj(os.path.join(path, 'training_output_cache'))
        x[:, i] = np.array(cache['accuracy_course'][0])
        y_acc[:, i] = np.array(cache['accuracy_course'][3])
        y_cx[:, i] = np.array(cache['accuracy_course'][4])

        tad = cache['test_accuracy_data']
        labels = tad['tested_labels']
        if i in (0, 2):
            for _, correct, predicted in tad['all_image_tests']:
                i_c = labels.index(correct)
                i_p = labels.index(predicted)
                if i == 0:
                    n_t[i_c] += 1
                elif i == 2:
                    n_st[i_c] += 1

                if i_c == i_p:
                    if i == 0:
                        label_acc_t[i_c] += 1
                    elif i == 2:
                        label_acc_st[i_c] += 1
                else:
                   if i == 0:
                       label_miss_t[i_c, i_p] += 1
                   elif i == 2:
                       label_miss_st[i_c, i_p] += 1

    fig = plt.figure(4, figsize=(10, 4.5))
    gs = plt.GridSpec(4, 2, hspace=0.3)
    ax = plt.subplot(gs[0:2, 0])
    ax.hlines([80, 1 / 6 * 100], -5, 100, color='k', linestyle='--', linewidth=0.5)
    ax.plot(x, y_acc * 100)
    ax.set_ylabel('Validation Accuracy, \%')
    despine(ax, right=True, top=True, bottom=True)
    ax.set_xlim(-5, 100)
    ax.set_ylim(0, 101)
    ax.set_xticks([])
    ax.set_yticks([0, (1/6*100), 80, 100])
    ax.set_yticklabels([r'$0$',
                        r'$P_{chance}$',
                        r'$80$',
                        r'$100$'])
    ax.spines['left'].set_bounds(0, 100)

    ax = plt.subplot(gs[2:4, 0])
    ax.plot(x, y_cx)
    ax.set_ylabel('Cross Entropy Error')
    ax.set_xlabel('Training Step')
    despine(ax, right=True, top=True)
    ax.set_xlim(-5, 100)
    ax.set_ylim(0, 14)
    ax.set_yticks([0, 14])
    ax.set_xticks([0, 100])
    # ax.spines['left'].set_bounds(10, 100)
    ax.spines['bottom'].set_bounds(0, 100)
    fig.align_labels()

    ax = plt.subplot(gs[0:2, 1])
    ind = np.arange(6)
    width = 0.3
    ax.hlines([1 / 6 * 100], -0.5 + width / 2, 5.5 + width / 2, color='k', linestyle='--', linewidth=0.5)
    ax.bar(ind, label_acc_t / n_t * 100, width, bottom=0)
    ax.bar(ind + width, label_acc_st / n_st * 100, width, color='C2', bottom=0)
    despine(ax, right=True, top=True)
    ax.set_ylabel('Test Accuracy, \%')
    ax.set_xlim(-0.5 + width / 2, 5.5 + width / 2)
    ax.set_ylim(0, 100)
    ax.set_xticks(ind + width / 2)
    ax.set_yticks([0, 100])
    ax.set_yticklabels([r'$0$',
                        r'$100$'])
    ax.set_xticklabels([
        r'all',
        r'$E\rightarrow E$',
        r'$E\rightarrow I$',
        r'$I\rightarrow E$',
        r'$I\rightarrow I$',
        r'none'
    ])
    ax.spines['left'].set_bounds(0, 100)


    ax = plt.subplot(gs[2, 1])
    width = 1/2
    ind = np.arange(6)
    tot = np.sum(label_miss_t, 1)
    bars = []
    pp.pprint(label_miss_t)
    for i in range(6):
        semi_tot = 0 if i == 0 else np.sum(label_miss_t[:, 0:i], 1)
        b = ax.bar(ind, label_miss_t[:, i], width, bottom=semi_tot)
        bars.append(b)

    for i_ind in ind:
        ypos = tot[i_ind] * 1.1 if tot[i_ind] < 100 else tot[i_ind] * 0.85
        ypos = 1 if ypos == 0 else ypos
        vert_align = 'bottom' if tot[i_ind] < 100 else 'top'
        ax.text(i_ind, ypos, '%d' % tot[i_ind], verticalalignment=vert_align,
                horizontalalignment='center')


    ax.set_xticks(ind)
    ax.set_yscale('log')
    ax.set_yticks([1, 10, 100, 430])
    ax.set_yticklabels([1, 10, 100, 430])
    ax.spines['left'].set_bounds(1, 430)
    ax.set_facecolor('none')
    ax.set_ylim(1, 2000)
    despine(ax, right=True, top=True)
    ax.set_xticklabels([
        'all',
        '$E\\rightarrow E$',
        '$E\\rightarrow I$',
        '$I\\rightarrow E$',
        '$I\\rightarrow I$',
        'none'
    ])
    ax.legend([bars[i] for i in (1, 2, 4, 5)], [
        '$E\\rightarrow E$',
        r'$E\rightarrow I$',
        r'$I\rightarrow I$',
        r'none'
    ], ncol=2, fontsize='x-small', frameon=False, loc=1, labelspacing=0.5,
              columnspacing=0.5, markerscale=0.75)

    ax = plt.subplot(gs[3, 1])
    width = 1 / 2
    ind = np.arange(6)
    tot = np.sum(label_miss_st, 1)
    bars = []
    pp.pprint(label_miss_st)
    for i in range(6):
        semi_tot = 0 if i == 0 else np.sum(label_miss_st[:, 0:i], 1)
        b = ax.bar(ind, label_miss_st[:, i], width, bottom=semi_tot)
        bars.append(b)

    for i_ind in ind:
        ypos = tot[i_ind] * 1.1 if tot[i_ind] < 100 else tot[i_ind] * 0.85
        ypos = 1 if ypos == 0 else ypos
        vert_align = 'bottom' if tot[i_ind] < 100 else 'top'
        ax.text(i_ind, ypos, '%d' % tot[i_ind], verticalalignment=vert_align,
                horizontalalignment='center')

    ax.set_ylabel('\# mislabeled')
    ax.set_xlabel('correct label')
    ax.set_xticks(ind)
    ax.set_yscale('log')
    ax.set_yticks([1, 10, 100, 430])
    ax.set_yticklabels([1, 10, 100, 430])
    ax.spines['left'].set_bounds(1, 430)
    ax.set_facecolor('none')
    ax.set_ylim(1, 2000)
    despine(ax, right=True, top=True)
    ax.set_xticklabels([
        'all',
        '$E\\rightarrow E$',
        '$E\\rightarrow I$',
        '$I\\rightarrow E$',
        '$I\\rightarrow I$',
        'none'
    ])

    return fig


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
    figure_4()
    multipage('final_figures', dpi=300, fmt='png')
    plt.show()