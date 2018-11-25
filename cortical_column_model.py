#!python3
# cortical_column_model.py

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from brian2tools import *
from project_utils import *
from scipy.stats.stats import pearsonr

def cortical_model():
    start_scope()

    duration = 1000 * ms
    base_firing_rate = 200 * Hz
    num_columns = 10
    column_thetas = [np.pi * ((- 1 / 2) + (i / num_columns - 1))
                     for i in range(num_columns)]
    delay = 2 * ms
    threshold = 1
    tau = 3 * ms
    theta_0 = pi * 0
    C = 1.5
    epsilon = 0.3

    J_0_alpha_E = 1.5
    J_2_alpha_E = 1
    J_0_alpha_I = -2
    J_2_alpha_I = -1

    N_0 = PoissonGroup(1, rates=base_firing_rate)

    spiking_neuron_kwargs = dict(
        threshold='m > threshold',
        reset='m = 0',
        refractory=1 * ms)
    spiking_neuron_eqns = '''
    dm/dt = (1 / tau) * -m:  1
    theta : 1
    J_0 : 1
    J_2 : 1
    '''
    column_neuron_args = [num_columns, spiking_neuron_eqns]
    N_E = NeuronGroup(*column_neuron_args, **spiking_neuron_kwargs)
    N_I = NeuronGroup(*column_neuron_args, **spiking_neuron_kwargs)
    neuron_groups = (N_0, N_E, N_I)
    ng_names = ('Input Neuron', 'Excitatory Neuron', 'Inhibitory Neuron')

    N_E.theta = N_I.theta = column_thetas
    N_E.J_0 = J_0_alpha_E
    N_E.J_2 = J_2_alpha_E
    N_I.J_0 = J_0_alpha_I
    N_I.J_2 = J_2_alpha_I

    synapse_model = '''
    w : 1
    '''
    input_synapse_on_pre = '''
    w = C * (1.0 - epsilon + epsilon * cos(2.0 * (theta_post - theta_0)))
    m_post += w
    '''
    input_synapse_kwargs = dict(
        model=synapse_model,
        on_pre=input_synapse_on_pre,
        delay=delay)
    S_E0 = Synapses(N_0, N_E, **input_synapse_kwargs)
    S_I0 = Synapses(N_0, N_I, **input_synapse_kwargs)

    inter_synapse_on_pre = '''
    w = (1.0 / num_columns) * (J_0_pre + J_2_pre * cos(2.0 * (theta_post - theta_pre)))
    m_post += w                      
    '''
    inter_synapse_kwargs = dict(
        model=synapse_model,
        on_pre=inter_synapse_on_pre,
        delay=delay)
    S_EE = Synapses(N_E, N_E, **inter_synapse_kwargs)
    S_II = Synapses(N_I, N_I, **inter_synapse_kwargs)
    S_EI = Synapses(N_I, N_E, **inter_synapse_kwargs)
    S_IE = Synapses(N_E, N_I, **inter_synapse_kwargs)
    synps = (S_E0, S_I0, S_EE, S_II, S_EI, S_IE)
    [s.connect(p=1) for s in synps]

    M_0 = SpikeMonitor(N_0)
    M_E = SpikeMonitor(N_E)
    M_I = SpikeMonitor(N_I)

    state_E = StateMonitor(N_E, ('m', ), record=True)
    state_I = StateMonitor(N_I, ('m', ), record=True)

    run(duration)

    spike_monitors = [M_0, M_E, M_I]
    state_monitors = [state_E, state_I]


    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m, nme in zip(axs, spike_monitors, ng_names):
        brian_plot(m, axes=ax)
        ax.set_title(nme)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for ax, m, nme in zip(axs, state_monitors, ng_names[1:]):
        # ax.plot(m.t, m.m[0])
        x = np.array([m.t for _ in m.m]).T / (1 * ms)
        y = (m.m + 3.5 * np.reshape(np.arange(0, len(m.m)), (-1, 1))).T
        ax.plot(x, y, 'k-')
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.set_title(nme)


def cortical_model_2():
    start_scope()

    duration = 2000 * ms
    num_columns = 5
    column_thetas = [np.pi * ((- 1 / 2) + (i / (num_columns - 1)))
                     for i in range(num_columns)]

    # theta_0 = pi * 0.4
    C_alpha_0 = 0.8 * nA
    epsilon = 0.5
    J_0_alpha_E = 1.5
    J_2_alpha_E = 0.9
    J_0_alpha_I = -1.5
    J_2_alpha_I = -1
    w_factor = 25 * mV
    connection_p = 0.5

    # Parameters
    C = 281 * pF
    gL = 30 * nS
    taum = C / gL
    EL = -70.6 * mV
    VT = -50.4 * mV
    DeltaT = 2 * mV
    Vcut = VT + 5 * DeltaT

    sigma = 1.5 * mV
    delay = taum * 0.1

    # Pick an electrophysiological behaviour
    tauw, a, b, Vr = 144 * ms, 4 * nS, 0.0805 * nA, -70.6 * mV  # Regular spiking (as in the paper)
    # tauw,a,b,Vr=20*ms,4*nS,0.5*nA,VT+5*mV # Bursting
    # tauw,a,b,Vr=144*ms,2*C/(144*ms),0*nA,-70.6*mV # Fast spiking

    spiking_neuron_eqns = '''
    dvm/dt = (gL * (EL - vm) + gL * DeltaT * exp((vm - VT)/DeltaT) + I - w) / C + (sigma * xi * taum ** -0.5) : volt
    dw/dt = (a * (vm - EL) - w) / tauw : amp
    I = C_t * C_alpha * (1.0 - epsilon + epsilon * cos(2.0 * (theta - theta_0))) : amp
    C_t = (1 - sign(t - (2000 * ms))) / 2 : 1
    C_alpha : amp
    theta : 1
    theta_0 = clip((t / (1 * second)) * pi - pi, -pi / 2, pi / 2) : 1
    J_0 : 1
    J_2 : 1
    '''

    spiking_neuron_kwargs = dict(
        N=num_columns,
        model=spiking_neuron_eqns,
        threshold='vm > Vcut',
        reset='vm = Vr; w += b',
        method='euler')
    N_E = NeuronGroup(**spiking_neuron_kwargs, name='Excitatory')
    N_I = NeuronGroup(**spiking_neuron_kwargs, name='Inhibitory')

    N_E.theta = N_I.theta = column_thetas
    N_E.vm = N_I.vm = Vr
    N_E.C_alpha = C_alpha_0
    N_I.C_alpha = C_alpha_0
    N_E.J_0 = J_0_alpha_E
    N_E.J_2 = J_2_alpha_E
    N_I.J_0 = J_0_alpha_I
    N_I.J_2 = J_2_alpha_I

    synapse_model = 'w_syn : 1'
    inter_synapse_on_pre = '''
    vm += w_syn * w_factor
    '''
    inter_synapse_kwargs = dict(
        model=synapse_model,
        on_pre=inter_synapse_on_pre,
        delay=delay)
    S_EE = Synapses(N_E, N_E, **inter_synapse_kwargs)
    S_II = Synapses(N_I, N_I, **inter_synapse_kwargs)
    S_EI = Synapses(N_I, N_E, **inter_synapse_kwargs)
    S_IE = Synapses(N_E, N_I, **inter_synapse_kwargs)
    synps = (S_EE, S_II, S_EI, S_IE)
    [s.connect(p=connection_p) for s in (S_EE, S_II)]
    [s.connect(p=connection_p) for s in (S_EI, S_IE)]
    for s in synps:
        s.w_syn = '(J_0_pre + J_2_pre * cos(2.0 * (theta_post - theta_pre))) / (num_columns * connection_p)'
    synp_names = ('Exct <- Exct', 'Inhb <- Inhb',
                  'Exct <- Inhb', 'Inhb <- Exct')
    for synp, nme in zip(synps, synp_names):
        visualise_connectivity(synp)
        title(nme)

    M_E = SpikeMonitor(N_E)
    M_I = SpikeMonitor(N_I)
    state_E = StateMonitor(N_E, ('vm', 'theta'), record=True)
    state_I = StateMonitor(N_I, ('vm', 'theta'), record=True)

    run(duration)

    spike_monitors = [M_E, M_I]
    state_monitors = [state_E, state_I]
    ng_names = ('Excitatory Neurons', 'Inhibitory_Neurons')

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    for ax, m, nme in zip(axs, spike_monitors, ng_names):
        brian_plot(m, axes=ax)
        ax.set_title(nme)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for ax, m, nme in zip(axs, state_monitors, ng_names):
        x = np.array([m.t for _ in m.vm]).T / (1 * ms)
        max_delta = np.max(m.vm[0].flatten()) - np.min(m.vm[0].flatten())
        y = (m.vm + (max_delta * 0.9) * np.reshape(np.arange(0, len(m.vm)),
                                                   (-1, 1))).T
        ax.plot(x, y, 'k-')
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.set_title(nme)

    return [(m.vm, m.t) for m in state_monitors]


if __name__ == "__main__":
    output = cortical_model_2()

    # [print(x) for m in output for x in m]

    me_v, me_t = output[0]
    mi_v, mi_t = output[1]

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    plt.tight_layout()
    for i in range(5):
        for j in range(5):
            axs[i, j].scatter(me_v[i], mi_v[j], c='k', marker='o')
            corr, p_value = pearsonr(me_v[i], mi_v[j])
            axs[i, j].set_title("r=" + '%.2f' % corr)
            grangertests(me_v[i], mi_v[j])

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    plt.tight_layout()
    for i in range(5):
        for j in range(5):
            axs[i, j].scatter(me_v[i], me_v[j], c='k', marker='o')
            corr, p_value = pearsonr(me_v[i], mi_v[j])
            axs[i, j].set_title("r=" + '%.2f' % corr)
            grangertests(me_v[i], mi_v[j])


    multipage()
    plt.show()


