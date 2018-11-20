#!python3
# brian_tests.py

from brian2 import *
from brian2tools import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def model(N=20, R_max=150, f=10, w=0.1, p=0.5, tau=5, tau_t=50, delta_t=0.1):
    # Get parameters
    R_max = R_max * Hz
    f = f * Hz
    tau = tau * ms
    tau_t = tau_t * ms
    duration = 200 * ms

    # Simulation code
    G = PoissonGroup(N, rates='R_max*0.5*(1+sin(2*pi*f*t))')
    eqs = '''
    dV/dt = -V/tau : 1
    dVt/dt = (1-Vt)/tau_t : 1
    '''
    H = NeuronGroup(1, eqs, threshold='V>Vt', reset='V=0; Vt += delta_t',
                    method='linear')
    H.Vt = 1
    S = Synapses(G, H, on_pre='V += w')
    S.connect(p=p)
    # Run it
    MG = SpikeMonitor(G)
    MH = StateMonitor(H, ('V', 'Vt'), record=True)
    run(duration)
    return N, S, MG, MH


def initialize_plot(N, S, MG, MH):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plt.subplots_adjust(right=0.5, left=0.05)
    brian_plot(MG, axes=axs[0])
    axs[0].set_title('Source neurons (Poisson)')

    p1 = axs[1].plot(zeros(N), arange(N), 'ob')
    p2 = axs[1].plot([0, 1], [S.i, ones(len(S.i)) * N / 2.], '-k')
    p3 = axs[1].plot([1], [N / 2.], 'og')
    axs[1].set_xlim(-0.1, 1.1)
    axs[1].set_ylim(-1, N)
    axs[1].axis('off')
    axs[1].set_title('Synapses')

    p4 = axs[2].plot(MH.t, MH.V[0], label='V')
    p5 = axs[2].plot(MH.t, MH.Vt[0], label='Vt')
    axs[2].legend(loc='upper left')
    axs[2].set_title('Target neuron')
    plt.show()
    return axs, (p1, p2, p3, p4, p5)


def add_sliders():
    n_slider = (20, 5, 100, 5, "Number of source neurons", int)
    r_max_slider = (300, 0, 500, 10, "Source neuron max firing rate (Hz)",
                    float)
    f_slider = (10, 1, 50, 1, "Source neuron frequency (Hz)", float)
    p_slider = (0.5, 0, 1, 0.01, "Synapse probability", float)
    w_slider = (0.3, 0, 1, 0.01, "Synapse weight", float)
    tau_slider = (5, 1, 50, 1, "Target neuron membrane time constant (ms)",
                  float)
    tau_t_slider = (30, 5, 500, 5, "Target neuron adaptation constant (ms)",
                    float)
    delta_t_slider = (1.0, 0, 20, 0.1, "Target neuron adaptation strength",
                      float)

    slider_params = [n_slider,
                     r_max_slider,
                     f_slider,
                     p_slider,
                     w_slider,
                     tau_slider,
                     tau_t_slider,
                     delta_t_slider]

    def make_position(i):
        ypos = 0.9 - (i * 0.05)
        return [0.55, ypos, 0.4, 0.03]

    sliders = []
    for i, sp in enumerate(slider_params):
        ax = plt.axes(make_position(i))
        s = Slider(ax, )
        sliders.append(s)

def update(axs, plots, N, S, MG, MH):
    p1, p2, p3, p4, p5 = plots

    brian_plot(MG, axes=axs[0])
    p1.set_xdata(zeros(N))
    p1.set_ydata(arange(N))
    p2.set_ydata([S.i, ones(len(S.i)) * N / 2.])
    p3.set_ydata([N / 2.])
    axs[1].set_ylim(-1, N)
    p4.set_xdata(MH.t)
    p4.set_ydata(MH.V[0])
    p5.set_xdata(MH.t)
    p5.set_ydata(MH.Vt)
    plt.show()

def test_1():
    start_scope()

    eqs = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    '''
    G = NeuronGroup(2, eqs, threshold='v>1', reset='v = 0', method='exact')
    G.I = [2, 0]
    G.tau = [10, 100] * ms

    # Comment these two lines out to see what happens without Synapses
    S = Synapses(G, G, on_pre='v_post += 0.2')
    S.connect(i=0, j=1)

    M = StateMonitor(G, 'v', record=True)

    run(100 * ms)

    plot(M.t / ms, M.v[0], label='Neuron 0')
    plot(M.t / ms, M.v[1], label='Neuron 1')
    xlabel('Time (ms)')
    ylabel('v')
    legend();

def test_2():
    G = PoissonGroup(1, rates=(500 * Hz))
    M = SpikeMonitor(G)
    run(500 * ms)
    fig, ax = plt.subplots(1, 1)
    brian_plot(M, axes=ax)

if __name__ == "__main__":

    # sim_output = model(20, 300, 10, 0.5, 0.3, 5, 30, 1.0)
    # plot_output = initialize_plot(*sim_output)

    # test_1()
    test_2()
    plt.show()

