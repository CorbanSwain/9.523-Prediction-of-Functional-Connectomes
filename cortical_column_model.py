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

def run_cortical_model(duration=500 * ms,
                       num_columns=5,
                       theta_in=0,
                       connection_probability=0.5,
                       theta_noise_sigma=np.pi / 2,
                       neuron_noise=3 * mV,
                       synapse_factor_E=1,
                       synapse_factor_I=1,
                       theta_0_expr='theta_in',
                       i_switch_expr='1'):

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
    tauw, a, b, Vr = 144 * ms, 4 * nS, 0.0805 * nA, -70.6 * mV
    # > Bursting
    # tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
    # > Fast Spiking
    # tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking

    # Network & Synapse Parameters
    C_alpha_0 = 1 * nA
    epsilon = 0.4
    J_0_alpha_E = 1.8
    J_2_alpha_E = 1.2
    J_0_alpha_I = -1.5
    J_2_alpha_I = -1
    max_synapse_magnitude = 85 * mV
    connection_p_max = connection_probability

    # Computed Parameters
    num_neurons = num_columns * 2
    column_thetas = [np.pi * ((- 1 / 2) + (i / (num_columns - 1)))
                     for i in range(num_columns)]
    J_max = np.max([abs(J_0_alpha_E + J_2_alpha_E), abs(J_0_alpha_I + J_2_alpha_I)])

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
    ''' % (i_switch_expr, theta_0_expr)
    neurons = NeuronGroup(N=num_neurons,
                          model=spiking_neuron_eqns,
                          threshold='vm > Vcut',
                          reset='vm = Vr; w += b',
                          method='euler')
    # Set Initial Voltage
    neurons.vm = EL

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

    (neurons.C_alpha,
     neurons.J_0,
     neurons.J_2,
     neurons.theta,
     neurons.theta_noise,
     neurons.synapse_magnitude) = [make_neuron_val_array(fun)
                                   for fun in (get_c_alpha,
                                               get_j_0,
                                               get_j_2,
                                               get_theta,
                                               get_theta_noise,
                                               get_synapse_magnitude)]

    if connection_p_max > 0:
        synapse_model = '''
        delta_theta = theta_post - theta_pre : 1
        J_2_term = J_2_pre * cos(2.0 * (delta_theta + theta_noise_pre)) : 1
        w_syn = J_0_pre + J_2_term : 1 
        w_syn_sign = sign(w_syn) : 1
        '''
        inter_synapse_on_pre = '''
        vm += w_syn_sign * synapse_magnitude_pre / (connection_p_max * num_neurons)
        '''
        inter_synapse_kwargs = dict(
            model=synapse_model,
            on_pre=inter_synapse_on_pre,
            delay=delay)
        synapses = Synapses(neurons, neurons, **inter_synapse_kwargs)
        synapses.connect(p='connection_p_max * abs(w_syn) / J_max')
    else:
        synapses = None

    spike_monitor = SpikeMonitor(neurons)
    state_monitor = StateMonitor(neurons, ('vm', 'I', 'theta_0'), record=True)
    run(duration)
    t_repeated = np.array([state_monitor.t for _ in state_monitor.vm]) * second
    v_with_peaks = add_peaks(state_monitor.vm, spike_monitor, Vpeak)
    return (t_repeated, v_with_peaks, state_monitor.I, state_monitor.theta_0), synapses


def figure_1():
    print('F1 - Beginning Unconnected Simulation')
    theta_0_expr = 'int(t > (1000.0 * ms)) * (pi * (-1.0 / 2.0 + (t - 1000.0 * ms) / (4000.0 * ms)))'
    i_switch_expr = 'int(abs(t - (750.0 * ms)) > (250.0 * ms))'
    (t, v, I, theta_0), _ = run_cortical_model(duration=5000*ms,
                                               num_columns=5,
                                               connection_probability=0,
                                               theta_0_expr=theta_0_expr,
                                               i_switch_expr=i_switch_expr)
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
    despine(ax2, right=True, top=True)
    ax2.spines['left'].set_bounds(-np.pi/2, np.pi/2)
    ax2.spines['bottom'].set_bounds(0, 5000)


def figure_2():
    (t, v, I, theta_0), _ = run_cortical_model(duration=1000*ms,
                                               connection_probability=0.5)


if __name__ == "__main__":
    matplotlib.rcdefaults()
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', **{'family': 'sans-serif',
                             'sans-serif': ['Helvetica']})
    matplotlib.rc('text.latex', preamble=r'''    
    \usepackage{sansmathfonts}
    \usepackage{helvet}
    \renewcommand{\rmdefault}{\sfdefault}
    \usepackage{units}''')

    figure_1()
    multipage('drafting_figure_1', fmt='eps')
    plt.show()
