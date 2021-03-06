#!python3
# mobile_net_training.py

import os
import sys
from tffp2.scripts import retrain
import tensorflow as tf
import time
import subprocess
import pprint as pp
import logging
import pickle

this_dir = os.path.dirname(os.path.realpath(__file__))


def save_obj(obj, pth):
    with open(pth + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(pth):
    extension = '.pkl'
    pth = pth + '' if pth.endswith(extension) else extension
    with open(pth + '.pkl', 'rb') as f:
        return pickle.load(f)


def touchdir(pth):
    try:
        os.mkdir(pth)
    except FileExistsError:
        pass


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


def start_tensorboard(pth):
    import tensorboard as tb
    import tensorboard.main
    import tensorboard.program
    from threading import Thread

    tb.program.FLAGS.logdir = pth
    thread = Thread(target=tb.main.main, args=tuple())
    thread.start()
    return thread


def run_training(image_dir, run_dir, log_dir, bottleneck_dir, model_dir):
    image_size = 224
    architecture = "mobilenet_1.0_%d" % image_size

    def sub_path(pth):
        return os.path.join(run_dir, pth)

    run_type = os.path.split(image_dir)[1]

    flags = Objdict()
    flags.bottleneck_dir = bottleneck_dir
    flags.how_many_training_steps = 100
    flags.model_dir = model_dir
    flags.summaries_dir = \
        os.path.join(log_dir, '_'.join((architecture,
                                        'N'+str(flags.how_many_training_steps),
                                        run_type)))
    flags.output_graph = sub_path('retrained_graph.pb')
    flags.output_labels = sub_path('retrained_labels.txt')
    flags.architecture = architecture
    flags.image_dir = image_dir
    flags.learning_rate = 0.005
    flags.testing_percentage = 20
    flags.validation_percentage = 10
    flags.test_batch_size = -1
    flags.validation_batch_size = -1
    flags.train_batch_size = -1
    flags.eval_step_interval = 5

    return retrain.run_with_args(**flags)


def train_on_all_imagesets():
    output_parent_dir = os.path.join(this_dir, 'training_output')
    run_dir = os.path.join(output_parent_dir,
                           time.strftime('train_%y%m%d-%H%M'))
    model_dir = os.path.join(output_parent_dir, 'models')
    log_dir = os.path.join(output_parent_dir, 'logs')
    touchdir(log_dir)
    bottleneck_dir = os.path.join(output_parent_dir, 'bottleneck_cache')
    touchdir(bottleneck_dir)

    tb_thread = start_tensorboard(log_dir)

    base_image_dir = os.path.join('training_data', 'USE_ME_181204-1755')

    for i_type in ('traces', 'negative_control', 'shuffled_traces',
                   'shuffled_negative_control'):
        print('Beginning training for %s data set.' % i_type)
        image_dir = os.path.join(base_image_dir, i_type)
        labeled_run_dir = run_dir + '_' + i_type
        touchdir(labeled_run_dir)
        output = run_training(image_dir,
                              labeled_run_dir,
                              log_dir,
                              bottleneck_dir,
                              model_dir)
        pp.pprint(output)
        save_obj(output, os.path.join(labeled_run_dir, 'training_output_cache'))

    print('Completed Training')
    tb_thread.join()


if __name__ == "__main__":
    train_on_all_imagesets()
