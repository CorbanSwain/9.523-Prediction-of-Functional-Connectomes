#!python3
# create_figures.py

from cortical_column_model import *
import matplotlib as mpl
from matplotlib import patches
import matplotlib.tight_layout


def make_assessment_plot(t, v, I, theta_0, connection=None,
                         colortheta=True,
                         ax1=None, ax2=None,
                         strip_xaxis=False, strip_yaxis=False):
    def strip_y(ax):
        despine(ax, left=True)
        ax.set_yticks([])
        ax.set_ylabel('')

    def strip_x(ax):
        despine(ax, bottom=True)
        ax.set_xticks([])
        ax.set_xlabel('')

    t_tot = t[0, -1] / ms
    if ax1:
        trace_plot, (trace_delta, space_idx) = plot_traces(t, v, ax1)
        trace_delta = trace_delta / mV
        n_neurons = len(space_idx)

        box_width = 100
        box_height = trace_delta * n_neurons / 2
        box_left = t_tot * 1.02
        box_right = box_left + box_width
        box_xcenter = box_left + box_width / 2
        box1_bottom = -80
        box1_top = box1_bottom + box_height
        box1_ycenter = box1_bottom + box_height / 2
        box2_bottom = box1_bottom + trace_delta * space_idx[int(n_neurons / 2)]
        box2_top = box2_bottom + box_height
        box2_ycenter = box2_bottom + box_height / 2

        rect_excit = patches.Rectangle((box_left, box1_bottom),
                                       width=box_width,
                                       height=box_height,
                                       facecolor='k', edgecolor=None)
        rect_inhib = patches.Rectangle((box_left, box2_bottom),
                                       width=box_width,
                                       height=box_height,
                                       facecolor='k', edgecolor=None)
        ax1.add_patch(rect_excit)
        ax1.add_patch(rect_inhib)

        ax1.text(box_xcenter, box1_ycenter, '$E$',
                 verticalalignment='center', color='w',
                 horizontalalignment='center', fontsize='large')
        ax1.text(box_xcenter, box2_ycenter,
                 '$I$', verticalalignment='center', color='w',
                 horizontalalignment='center', fontsize='large')

        if connection:
            connection = [True, ] * 4 if connection is True else connection
            shrink = 5
            annotate_kwargs = dict(textcoords='data', xycoords='data', annotation_clip=False)
            arrow_kwargs = dict(arrowstyle='simple,tail_width=0.3,head_width=0.75,head_length=0.75',
                                color='r',
                                patchA=None,
                                patchB=None,
                                linewidth=1,
                                capstyle='butt',
                                joinstyle='miter',
                                shrinkA=shrink, shrinkB=shrink
                                )
            if connection[0]:
                ax1.annotate('',
                             xytext=(box_right, box1_ycenter + 100),
                             xy=(box_right, box1_ycenter - 100),
                             arrowprops=dict(connectionstyle='arc3,rad=-1.8',
                                             **arrow_kwargs
                                             ),
                             **annotate_kwargs)
            if connection[1]:
                ax1.annotate('',
                             xytext=(box_xcenter, box1_top),
                             xy=(box_xcenter, box2_bottom),
                             arrowprops=dict(connectionstyle='arc3,rad=-0.15',
                                             **arrow_kwargs),
                             **annotate_kwargs)
            if connection[2]:
                ax1.annotate('',
                             xytext=(box_right, box2_bottom + 75),
                             xy=(box_right, box1_top - 75),
                             arrowprops=dict(connectionstyle='arc3,rad=-0.5',
                                             **arrow_kwargs
                                             ),
                             **annotate_kwargs)
            if connection[3]:
                ax1.annotate('',
                             xytext=(box_right, box2_ycenter - 100),
                             xy=(box_right, box2_ycenter + 100),
                             arrowprops=dict(connectionstyle='arc3,rad=1.8',
                                             **arrow_kwargs
                                             ),
                             **annotate_kwargs)

        ax1.set_xlim(left=-50, right=box_right)
        ax1.set_yticks([-70, 20])
        ax1.set_yticklabels(['$-70$ mV', '$20$ mV'])
        ax1.set_xticks([])
        ax1.tick_params(axis='y', which='major', pad=2)
        despine(ax1, right=True, bottom=True, top=True)
        ax1.spines['left'].set_bounds(-70, 20)
        if strip_yaxis:
            strip_y(ax1)

    if ax2:
        ax2.hlines(0, -500, t_tot, color='k', linestyle='--', linewidth=0.5)
        x = t[1, :] / ms
        y = theta_0[1, :]
        y[np.logical_and(x <= 1000, x > 500)] = np.nan
        if colortheta:
            color = 'r'
        else:
            color = 'k'
        ax2.plot(x, y, '-', linewidth=3, c=color)
        ax2.set_yticks([-pi / 2, 0, pi / 2])
        ax2.set_yticklabels([r'$-\nicefrac{\pi}{2}$',
                             r'$0$',
                             r'$+\nicefrac{\pi}{2}$'])
        ax2.set_xticks([0, t_tot])
        ax2.set_xticklabels(['$0$ s', '$%.0f$ s' % (t_tot / 1e3)])
        ax2.set_xlim(ax1.get_xlim())
        ax2.tick_params(axis='y', which='major', pad=2)
        ax2.set_ylabel(r'$\theta_0$')
        ax2.set_xlabel(r'time', labelpad=-1)
        despine(ax2, right=True, top=True)
        ax2.spines['left'].set_bounds(-np.pi / 2, np.pi / 2)
        ax2.spines['bottom'].set_bounds(0, t_tot)
        if strip_yaxis:
            strip_y(ax2)
        if strip_xaxis:
            strip_x(ax2)


def run_static_input_model(synapses_allowed='none'):
    assessment_model_cache_path = os.path.join('caches', 'run_static_input_data')
    try:
        cached_data = load_obj(assessment_model_cache_path)
        cached_result = cached_data[synapses_allowed]
    except (FileNotFoundError, EOFError):
        cached_data = dict()
        cached_result = None
    except KeyError:
        cached_result = None

    def run_model():
        print('Beginning static simulation (%s).' % synapses_allowed)
        output = run_cortical_model(duration=1000 * ms,
                                    num_columns=5,
                                    do_conditionally_connect=False,
                                    theta_noise_sigma=0,
                                    C_alpha_0=0.9 * nA,
                                    J_0_alpha_E=1.8,
                                    J_2_alpha_E=1.2,
                                    J_0_alpha_I=-1.8,
                                    J_2_alpha_I=-1.2,
                                    synapses_allowed=synapses_allowed)
        s = output[1]
        s_new = Objdict()
        s_new.i = np.array(s.i)
        s_new.j = np.array(s.j)
        s_new.w = np.array(s.w_syn)
        output = (output[0], s_new)
        cached_data[synapses_allowed] = output
        save_obj(cached_data, assessment_model_cache_path)
        return output

    return cached_result or run_model()


def run_assessment_model(synapses_allowed='none'):
    assesment_model_cache_path = os.path.join('caches', 'run_assessment_data')
    try:
        cached_data = load_obj(assesment_model_cache_path)
        cached_result = cached_data[synapses_allowed]
    except FileNotFoundError:
        cached_data = dict()
        cached_result = None
    except KeyError:
        cached_result = None

    def run_model():
        print('Beginning assesment simulation (%s).' % synapses_allowed)
        theta_0_expr = 'int(t > (1000.0 * ms)) * (pi * (-1.0 / 2.0 + (t - 1000.0 * ms) / (1000.0 * ms)))'
        i_switch_expr = 'int(abs(t - (750.0 * ms)) > (250.0 * ms))'
        output, _ = run_cortical_model(duration=2000 * ms,
                                       num_columns=5,
                                       theta_0_expr=theta_0_expr,
                                       i_switch_expr=i_switch_expr,
                                       do_conditionally_connect=False,
                                       theta_noise_sigma=0,
                                       C_alpha_0=0.9 * nA,
                                       synapses_allowed=synapses_allowed)
        cached_data[synapses_allowed] = output
        save_obj(cached_data, assesment_model_cache_path)
        return output

    return cached_result or run_model()


def name_to_connection(nme):
    i_allowed = synapses_allowed_list.index(nme)
    if i_allowed <= 3:
        connection = [bool(i == i_allowed) for i in range(4)]
    elif i_allowed == 4:
        connection = True
    elif i_allowed == 5:
        connection = False
    else:
        raise ValueError('Unexpected name, not in synapses_allowed_list.')

    return connection


def figure_1(synapses_allowed):
    print('F1 - Making plots (%s).' % synapses_allowed)
    fig = plt.figure(figsize=(6.5, 4.5))
    gs = GridSpec(5, 1, hspace=0.4)
    ax1 = plt.subplot(gs[:-1, :])
    ax2 = plt.subplot(gs[-1, :])
    make_assessment_plot(*run_assessment_model(synapses_allowed),
                         connection=name_to_connection(synapses_allowed),
                         ax1=ax1, ax2=ax2)
    return fig


def figure_2():
    print('F2 - Making Plots.')
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(9, 3, hspace=0.4, bottom=0.06, left=0.05, right=.95, top=0.95)

    nme = 'e_to_e'
    ax1 = plt.subplot(gs[:4, 0])
    make_assessment_plot(*run_assessment_model(nme),
                         connection=name_to_connection(nme),
                         ax1=ax1, strip_xaxis=True, colortheta=False)

    nme = 'e_to_i'
    ax1 = plt.subplot(gs[4:-1, 0])
    ax2 = plt.subplot(gs[-1, 0])
    make_assessment_plot(*run_assessment_model(nme),
                         connection=name_to_connection(nme),
                         ax1=ax1, ax2=ax2, colortheta=False)

    nme = 'i_to_e'
    ax1 = plt.subplot(gs[:4, 1])
    make_assessment_plot(*run_assessment_model(nme),
                         connection=name_to_connection(nme),
                         ax1=ax1, colortheta=False, strip_xaxis=True, strip_yaxis=True)

    nme = 'i_to_i'
    ax1 = plt.subplot(gs[4:-1, 1])
    ax2 = plt.subplot(gs[-1, 1])
    make_assessment_plot(*run_assessment_model(nme),
                         connection=name_to_connection(nme),
                         ax1=ax1, ax2=ax2, colortheta=False, strip_yaxis=True)

    nme = 'all'
    ax1 = plt.subplot(gs[:4, 2])
    make_assessment_plot(*run_assessment_model(nme),
                         connection=name_to_connection(nme),
                         ax1=ax1, colortheta=False, strip_xaxis=True, strip_yaxis=True)

    nme = 'none'
    ax1 = plt.subplot(gs[4:-1, 2])
    ax2 = plt.subplot(gs[-1, 2])
    make_assessment_plot(*run_assessment_model(nme),
                         connection=name_to_connection(nme),
                         ax1=ax1, ax2=ax2, colortheta=False, strip_yaxis=True)
    return fig


def figure_3():
    match_counts = np.zeros((6, 2))

    lst = [synapses_allowed_list[i] for i in (5, 0, 1, 2, 3, 4)]
    for i, arch in enumerate(lst):
        (_, v, _, _), s = run_static_input_model(arch)
        print('F3 - Making Plots')
        match_counts[i, 0] = plot_correlations(10, v, v, s, compare_test='correlation', connectivity=arch)
        match_counts[i, 1] = plot_correlations(10, v, v, s, compare_test='granger', connectivity=arch)

    print(match_counts)
    save_obj(match_counts, os.path.join('caches', 'match_counts'))

def figure_4():
    print('F4 - Gathering training outputs.')
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

    print('F4 - Making plots.')
    fig = plt.figure(4, figsize=(14, 4.5))
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

    [figure_1(sa) for sa in synapses_allowed_list]
    figure_2()
    figure_3()
    figure_4()
    print('Saving All Figures: ')
    multipage('final_figures', dpi=300, fmt='png')
    multipage('final_figures', dpi=300, fmt='eps')
    multipage('final_figures', dpi=300, fmt='pdf')
    plt.show()