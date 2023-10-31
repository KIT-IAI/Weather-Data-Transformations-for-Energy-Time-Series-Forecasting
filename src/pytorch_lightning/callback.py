import time
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytorch_lightning import Callback


class TerminalCallback(Callback):
    """ Terminal callback for terminal logging and metric saving. """

    def __init__(self, display=None, output_every_epoch=1, id=None):
        """ Initialize with keys to listen to and output frequency. """
        self.display = display
        self.data = {}
        self.durations = list()
        self.output_every_epoch = output_every_epoch
        self.id = id
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        """ Save starting time on epoch start. """
        self.epoch_start_time = time.time()

    def on_epoch_end(self, trainer, pl_module):
        """ Calculate time, log/save metric on epoch end. """
        epoch = pl_module.current_epoch + 1
        logs = trainer.callback_metrics
        for key in logs.keys():
            if key not in self.data:
                self.data[key] = list()
            self.data[key].append((epoch, logs[key].cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
        """ Print training process information """
        epoch = pl_module.current_epoch
        max_epochs = trainer.max_epochs
        if self.epoch_start_time is None:
            duration = 0
        else:
            duration = time.time() - self.epoch_start_time
        self.durations.append(duration)

        if self.output_every_epoch > 0 and (epoch - 1) % self.output_every_epoch == 0:
            if self.id is None:
                output = f'Epoch {epoch:4}/{max_epochs}  '
            else:
                output = f'[{self.id:06}] Epoch {epoch:4}/{max_epochs}  '

            # set all keys if display is None
            if self.display is None:
                keys_to_display = self.data.keys()
            else:
                keys_to_display = self.display

            for key in keys_to_display:
                data = self.data[key][-1]
                _, metric = data
                output += '  ' + key + f' {metric:.4f}'
            output += f'   Duration {duration:.2f} s'
            print(output)

    def on_train_end(self, trainer, pl_module):
        """ Calculate best metric values and their epoch. """
        metric = {}
        for key in self.data.keys():
            epochs = [epoch for epoch, _ in self.data[key]]
            values = [value for _, value in self.data[key]]
            min_idx = np.argmin(values, axis=0)
            metric[f'opt/epoch/{key}'] = epochs[min_idx]
            metric[f'opt/{key}'] = values[min_idx]
        self.opt_metric = metric.copy()
        metric['avg_duration'] = np.mean(self.durations)
        pprint.pprint(metric)

    def plot_metric(self, keys, filename='plot.png'):
        """ Save line plot of given metric key. """
        values = list()
        for key in keys:
            x, y = list(zip(*self.data[key]))
            plt.plot(x[::2], y[0::2], label=key)
        plt.legend()
        plt.savefig(filename)
        plt.close()