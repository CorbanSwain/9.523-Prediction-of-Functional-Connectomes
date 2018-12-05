#!python3
# data_generator.py

from cortical_column_model import *


def generate_learning_set(num_simulations=1000, do_shuffle_traces=False, directory='training_data',
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
    directory = os.path.join('training_data', sub_dir)
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
        if dt.total_seconds() < 1:
            rate = 0
            time_left = 0
        else:
            rate = i_simulation / (dt.total_seconds() / 60)
            time_left = ((num_simulations - i_simulation)) / rate / 60

        print('Sim: %5d / %5d | %4.1f run/min | %6.2f hr remain | save to -> %-20s'
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
    num_sims = 6000
    runs = [
        (num_sims, False, 'traces', False),
        (num_sims, True, 'shuffled_traces', False),
        (num_sims, False, 'negative_control', True),
        (num_sims, True, 'shuffled_negative_control', True)
    ]
    [generate_learning_set(*args) for args in runs]

    # windows pc command =
    # C:\Users\CorbanSwain\AppData\Local\Programs\Python\Python36\python.exe data_generator.py
