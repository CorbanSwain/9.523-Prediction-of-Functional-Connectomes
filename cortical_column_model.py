#!python3
# cortical_column_model.py

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from brian2tools import *
from project_utils import *
import matplotlib
import matplotlib.patches as patches
from matplotlib.markers import MarkerStyle
from scipy.stats.stats import pearsonr
import matplotlib.collections as collections

synapses_allowed_list = ['e_to_e', 'e_to_i', 'i_to_e', 'i_to_i', 'all']

def run_cortical_model(duration=500 * ms,
                       num_columns=5,
                       theta_in=0,
                       connection_probability=0.5,
                       theta_noise_sigma=np.pi / 2,
                       neuron_noise=3 * mV,
                       synapse_factor_E=1,
                       synapse_factor_I=1,
                       theta_0_expr='theta_in',
                       i_switch_expr='1',
                       b=0.000850 * nA,
                       epsilon=0.4,
                       J_0_alpha_E=1.8,
                       J_2_alpha_E=1.2,
                       J_0_alpha_I=-1.5,
                       J_2_alpha_I=-1,
                       max_synapse_magnitude=85 * mV,
                       synapses_allowed='all',
                       C_alpha_0=1 * nA,
                       synapse_weight_expr='w_syn_sign',
                       do_conditionally_connect=True):

    start_scope()

    # Neuron Parameters
    C = 281 * pF
    gL = 30 * nS
    taum = C / gL
    EL = -70.6 * mV
    VT = -50.4 * mV
    DeltaT = 2 * mV
    Vcut = VT + 5 * DeltaT
    Vpeak = 20 * mV
    delay = taum * .1
    sigma = neuron_noise

    # Electrophysiological behaviour
    # > Regular Spiking
    tauw, a, b, Vr = 144 * ms, 4 * nS, b, -70.6 * mV
    # > Bursting
    # tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
    # > Fast Spiking
    # tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking

    # Network & Synapse Parameters
    connection_p_max = connection_probability

    # Computed Parameters
    num_neurons = num_columns * 2
    column_thetas = [np.pi * ((- 1 / 2) + (i / (num_columns - 1)))
                     for i in range(num_columns)]
    J_max = np.max([abs(J_0_alpha_E + J_2_alpha_E), abs(J_0_alpha_I + J_2_alpha_I)])

    # Defining Neuron Dynamics
    spiking_neuron_eqns = '''
    dvm/dt = (gL * (EL - vm) + gL * DeltaT * exp((vm - VT)/DeltaT) + I - w) / C + (sigma * xi * taum ** -0.5) : volt
    dw/dt = (a * (vm - EL) - w) / tauw : amp
    I = I_switch * C_alpha * (1.0 - epsilon + epsilon * cos(2.0 * (theta - theta_0))) : amp
    I_switch = %s : 1
    C_alpha : amp
    theta : 1
    theta_0 = %s : 1
    theta_noise : 1
    synapse_magnitude : volt
    J_0 : 1
    J_2 : 1
    e_neuron : 1 
    i_neuron : 1
    ''' % (i_switch_expr, theta_0_expr)
    neurons = NeuronGroup(N=num_neurons,
                          model=spiking_neuron_eqns,
                          threshold='vm > Vcut',
                          reset='vm = Vr; w += b',
                          method='euler')
    neurons.vm = EL #sets initial voltage

    # Defining functions to set theta- and E/I-varying neuron parameters
    def make_neuron_val_array(fun):
        return [fun(i) for i in range(num_neurons)]

    def get_neuron_val(idx, fun_excit, fun_inhib):
        return fun_excit(idx) if idx < num_columns else fun_inhib(idx - num_columns)

    def get_c_alpha(idx):
        return get_neuron_val(idx,
                              fun_excit=lambda _: C_alpha_0,
                              fun_inhib=lambda _: C_alpha_0)

    def get_j_0(idx):
        return get_neuron_val(idx,
                              fun_excit=lambda _: J_0_alpha_E,
                              fun_inhib=lambda _: J_0_alpha_I)

    def get_j_2(idx):
        return get_neuron_val(idx,
                              fun_excit=lambda _: J_2_alpha_E,
                              fun_inhib=lambda _: J_2_alpha_I)

    def get_theta(idx):
        def theta_fun(i_column):
            return column_thetas[i_column]
        return get_neuron_val(idx,
                              fun_excit=theta_fun,
                              fun_inhib=theta_fun)

    def get_theta_noise(_):
        return np.random.randn() * theta_noise_sigma

    def get_synapse_magnitude(idx):
        def make_mag_fun(factor):
            return lambda _: max_synapse_magnitude * factor
        return get_neuron_val(idx,
                              fun_excit=make_mag_fun(synapse_factor_E),
                              fun_inhib=make_mag_fun(synapse_factor_I))

    def get_e_neuron(idx):
        return get_neuron_val(idx, lambda _: 1, lambda _: 0)

    def get_i_neuron(idx):
        return get_neuron_val(idx, lambda _: 0, lambda _: 1)

    # Setting variable parameters
    (neurons.C_alpha,
     neurons.J_0,
     neurons.J_2,
     neurons.theta,
     neurons.theta_noise,
     neurons.synapse_magnitude,
     neurons.e_neuron,
     neurons.i_neuron) \
        = [make_neuron_val_array(fun)
           for fun in (get_c_alpha,
                       get_j_0,
                       get_j_2,
                       get_theta,
                       get_theta_noise,
                       get_synapse_magnitude,
                       get_e_neuron,
                       get_i_neuron)]

    if synapses_allowed == 'all':
        connect_cond_str = '1'
    elif synapses_allowed == 'e_to_e':
        connect_cond_str = 'e_neuron_pre * e_neuron_post'
    elif synapses_allowed == 'e_to_i':
        connect_cond_str = 'e_neuron_pre * i_neuron_post'
    elif synapses_allowed == 'i_to_e':
        connect_cond_str = 'i_neuron_pre * e_neuron_post'
    elif synapses_allowed == 'i_to_i':
        connect_cond_str = 'i_neuron_pre * e_neuron_post'

    if not do_conditionally_connect:
        connection_p_max = 1

    if connection_p_max > 0:
        # Defining synapse dynamics and connections
        synapse_model = '''
        delta_theta = theta_post - theta_pre : 1
        J_2_term = J_2_pre * cos(2.0 * (delta_theta + theta_noise_pre)) : 1
        w_syn = J_0_pre + J_2_term : 1 
        w_syn_sign = sign(w_syn) : 1
        w_syn_applied = %s : 1
        connect_cond = %s : 1
        ''' % (synapse_weight_expr, connect_cond_str)
        inter_synapse_on_pre = '''
        vm += w_syn_applied * synapse_magnitude_pre / (connection_p_max * num_neurons)
        '''
        synapses = Synapses(neurons, neurons,
                            model=synapse_model,
                            on_pre=inter_synapse_on_pre,
                            delay=delay)

        connect_kwargs = dict()
        if do_conditionally_connect:
            connect_kwargs['p'] = 'connect_cond * connection_p_max * abs(w_syn) / J_max'
        else:
            connect_kwargs['p'] = 'connect_cond'
        synapses.connect(**connect_kwargs)
    else:
        synapses = None

    # Setting up monitors to record during the simulated activity
    spike_monitor = SpikeMonitor(neurons)
    state_monitor = StateMonitor(neurons, ('vm', 'I', 'theta_0'), record=True)

    # Performing simulation
    run(duration)

    # Preparing outputs and returning
    t_repeated = np.array([state_monitor.t for _ in state_monitor.vm]) * second
    v_with_peaks = add_peaks(state_monitor.vm, spike_monitor, Vpeak)
    return (t_repeated, v_with_peaks, state_monitor.I, state_monitor.theta_0), synapses


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


def generate_learning_set(num_simulations=1000, do_shuffle_traces=False, directory='training',
                          do_shuffle_labels=False):

    def rand_in_range(low, high):
        return low + (high - low) * np.random.rand()

    random_vars = [
        ('connection_probability', lambda _: rand_in_range(0.4, 1)),
        ('theta_noise_sigma', lambda _: rand_in_range(0, 0.5)),
        ('neuron_noise', lambda _: rand_in_range(2, 4) * mV),
        ('b', lambda _: rand_in_range(500E-6, 1E-3) * nA),
        ('epsilon', lambda _: rand_in_range(0.1, 0.5)),
        ('J_0_alpha_E', lambda _: rand_in_range(1, 2)),
        ('J_0_alpha_I', lambda _: rand_in_range(1, 2)),
        ('J_2_alpha_E', lambda dct: rand_in_range(0.75, dct['J_0_alpha_E'])),
        ('J_2_alpha_I', lambda dct: rand_in_range(0.75, dct['J_0_alpha_I'])),
        ('max_synapse_magnitude', lambda _: rand_in_range(85 - 3, 85 + 3) * mV),
        ('theta_in', lambda _: rand_in_range(-np.pi/2, pi/2)),
        ('synapses_allowed',
         lambda _: synapses_allowed_list[np.random.randint(0, len(synapses_allowed_list))]),
        ('C_alpha_0', lambda _: rand_in_range(0.9, 1.2) * nA)
    ]

    def lazy_setdefault(dct, ky, valfun, *args):
        if ky not in dct:
            dct[ky] = valfun(*args)

    kwargs_init = dict(
        duration=500 * ms,
        num_columns=5,
        connection_probability=0.5,
        theta_noise_sigma=np.pi / 2,
        neuron_noise=3 * mV,
        b=0.000850 * nA,
        # epsilon=0.4,
        J_0_alpha_E=1.8,
        J_2_alpha_E=1.2,
        J_0_alpha_I=-1.5,
        J_2_alpha_I=-1,
        max_synapse_magnitude=85 * mV,
        do_conditionally_connect=False,
        synapse_weight_expr='w_syn',
        # C_alpha_0=1 * nA,
    )

    sub_dir = directory
    directory = os.path.join('training', sub_dir)
    if os.path.exists(directory):
        directory = directory + '_' + time.strftime('%y%m%d-%H%M')

    touchdir(directory)
    t0 = datetime.datetime.now()
    for i_simulation in range(num_simulations):
        kwargs = dict(kwargs_init)
        for ky, randfun in random_vars:
            lazy_setdefault(kwargs, ky, randfun, kwargs)
        ti = datetime.datetime.now()
        dt = ti - t0
        rate = i_simulation / (dt.total_seconds() / 60)
        time_left = (rate * (num_simulations - i_simulation)) / 60
        print('Sim: %5d / %5d | %4.1f run/min | %4.2f hr remain | save to -> %-20s'
              % (i_simulation, num_simulations, rate, time_left, sub_dir))
        (t, v_traces, _, _), synapses = run_cortical_model(**kwargs)
        if do_shuffle_labels:
            given_label = synapses_allowed_list[np.random.randint(0, len(synapses_allowed_list))]
        else:
            given_label = kwargs['synapses_allowed']
        full_dir = os.path.join(directory,given_label)
        make_training_image_pair(t,
                                 v_traces,
                                 synapses,
                                 idx=i_simulation,
                                 px_dim=224,
                                 directory=full_dir,
                                 do_shuffle=do_shuffle_traces)


if __name__ == "__main__":
    matplotlib.rcdefaults()
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('text.latex', preamble=r'''    
    \usepackage{sansmathfonts}
    \usepackage{helvet}
    \renewcommand{\rmdefault}{\sfdefault}
    \usepackage{units}''')

    # figure_1()
    # figure_2()

    num_sims = 3500
    runs = [
        (num_sims, False, 'traces', False),
        (num_sims, True, 'shuffled_traces', False),
        (num_sims, False, 'negative_control', True),
        (num_sims, True, 'shuffled_negative_control', True)
    ]
    [generate_learning_set(*args) for args in runs]
    multipage('drafting_figure_2')
    plt.show()
