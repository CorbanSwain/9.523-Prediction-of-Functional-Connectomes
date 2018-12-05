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

synapses_allowed_list = ['e_to_e', 'e_to_i', 'i_to_e', 'i_to_i', 'all', 'none']

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
    elif synapses_allowed == 'none':
        connect_cond_str = '0'

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
    return (t_repeated, v_with_peaks, state_monitor.I, state_monitor.theta_0), \
           synapses


