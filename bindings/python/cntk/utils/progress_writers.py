# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
from __future__ import print_function
import time


class BaseProgressWriter(object):
    '''Base class for all classes that record the training progress. Should never be used directly.'''

    def __init__(self, frequency=None, first=0):
        '''
        Constructor.

        Args:
            frequency (`int` or `None`, default `None`):  determines how often the actual writing will occur.
              A value of 0 means a geometric schedule (1,2,4,8...).
              A value > 0 means a arithmetic schedule (3,9,12... when `freq` is 3)
              A value of None means no per-minibatch writes.
            first (`int`, default 0): Only start writing after the minibatch number is greater or equal to ``first``.
        '''
        if (frequency is not None) and (frequency < 0):
            raise ValueError('frequency must be a positive integer')

        from sys import maxsize
        if frequency is None:
            frequency = maxsize

        self.freq = frequency
        self.first = first
        self.loss_since_start = 0
        self.metric_since_start = 0
        self.samples_since_start = 0
        self.updates_since_start = 0
        self.loss_since_last = 0
        self.metric_since_last = 0
        self.samples_since_last = 0
        self.total_updates = 0
        self.epochs = 0
        self.epoch_start_time = 0

    def update_with_trainer(self, trainer, with_metric=False):
        '''
        Updates the accumulators using the loss, the minibatch_size and optionally the metric
        using the information from the ``trainer``.

        Args:
            trainer (:class:`cntk.trainer.Trainer`): trainer from which information is gathered
            with_metric (`bool`): whether to update the metric accumulators
        '''
        if trainer.previous_minibatch_sample_count == 0:
            return
        self._update_progress(
            trainer.previous_minibatch_loss_average,
            trainer.previous_minibatch_sample_count,
            trainer.previous_minibatch_evaluation_average if with_metric else None)

    def summarize_progress(self, with_metric=False):
        '''
        Write a summary of progress since the last call to this function.

        Args:
            with_metric (`bool`): if `False` it only prints the loss, otherwise it prints both the loss and the metric
        '''
        self.epochs += 1
        epoch_end_time = time.time()
        avg_loss, avg_metric, samples = (0, 0, 0)
        if self.samples_since_start != 0:
            avg_loss, avg_metric, samples = self._reset_start()

        time_delta = epoch_end_time - self.epoch_start_time
        ret = self._write_progress_summary(avg_loss, avg_metric if with_metric else None, samples, time_delta)
        self.epoch_start_time = epoch_end_time
        return ret

    def close(self, *args, **kwargs):
        '''
        Make sure that any outstanding records are immediately persisted, then close any open files.
        Any subsequent attempt to use the object will cause a RuntimeError.
        '''
        self._close(*args, **kwargs)

    def _avg_loss_since_start(self):
        return self.loss_since_start / self.samples_since_start

    def _avg_metric_since_start(self):
        return self.metric_since_start / self.samples_since_start

    def _avg_loss_since_last(self):
        return self.loss_since_last / self.samples_since_last

    def _avg_metric_since_last(self):
        return self.metric_since_last / self.samples_since_last

    def _reset_start(self):
        ret = self._avg_loss_since_start(), self._avg_metric_since_start(), self.samples_since_start
        self.loss_since_start = 0
        self.metric_since_start = 0
        self.samples_since_start = 0
        self.updates_since_start = 0
        return ret

    def _reset_last(self):
        ret = self._avg_loss_since_last(), self._avg_metric_since_last(), self.samples_since_last
        self.loss_since_last = 0
        self.metric_since_last = 0
        self.samples_since_last = 0
        return ret

    def _update_progress(self, loss, minibatch_size, metric=None):
        self.samples_since_start += minibatch_size
        self.samples_since_last += minibatch_size
        self.loss_since_start += loss * minibatch_size
        self.loss_since_last += loss * minibatch_size
        self.updates_since_start += 1
        self.total_updates += 1

        if metric is not None:
            self.metric_since_start += metric * minibatch_size
            self.metric_since_last += metric * minibatch_size

        if self.epoch_start_time == 0:
            self.epoch_start_time = time.time()

        if self.freq == 0 and (self.updates_since_start + 1) & self.updates_since_start == 0:
            avg_loss, avg_metric, samples = self._reset_last()
            self._write_progress_update(0, avg_loss, avg_metric if metric is not None else None, samples)
        elif self.freq > 0 and (self.updates_since_start % self.freq == 0 or self.updates_since_start <= self.first):
            avg_loss, avg_metric, samples = self._reset_last()

            if self.updates_since_start <= self.first:  # updates for individual MBs
                first_mb = self.updates_since_start
            else:
                first_mb = max(self.updates_since_start - self.freq + 1, self.first + 1)

            self._write_progress_update(first_mb, avg_loss, avg_metric if metric is not None else None, samples)

    def _write_progress_update(self, first_mb, avg_loss, avg_metric, samples):
        # To be overriden in derived classes.
        raise NotImplementedError('Attempting to use an abstract BaseProgressWriter class')

    def _write_progress_summary(self, avg_loss, avg_metric, samples, time_delta):
        # To be overriden in derived classes.
        raise NotImplementedError('Attempting to use an abstract BaseProgressWriter class')

    def _close(self, *args, **kwargs):
        # To be overriden in derived classes.
        raise NotImplementedError('Attempting to use an abstract BaseProgressWriter class')


def _warn_deprecated(message):
    from warnings import warn
    warn('DEPRECATED: ' + message, DeprecationWarning, stacklevel=2)


# TODO: Let's switch to import logging in the future instead of print. [ebarsoum]
class ProgressPrinter(BaseProgressWriter):
    '''
    Allows tracking various training time statistics (e.g. loss and metric)
    and output them as training progresses.

    It provides the number of samples, average loss and average metric
    since the last output or since the start of accumulation.

    Args:
        freq (int or None, default None):  determines how often
         printing will occur. The value of 0 means an geometric
         schedule (1,2,4,...). A value > 0 means a arithmetic schedule
         (a log print for minibatch number: ``freq``, a log print for minibatch number: 2*``freq``, a log print for minibatch number: 3*``freq``,...), and a value of None means no per-minibatch log.
        first (int, default 0): Only start logging after the minibatch number is greater or equal to ``first``.
        tag (string, default EmptyString): prepend minibatch log lines with your own string
        log_to_file (string or None, default None): if None, output log data to stdout.  If a string is passed, the string is path to a file for log data.
        gen_heartbeat (bool, default False): If True output a progress message every 10 seconds or so to stdout.
        num_epochs (int, default 300): The total number of epochs to be trained.  Used for some metadata.  This parameter is optional.
    '''

    def __init__(self, freq=None, first=0, tag='', log_to_file=None, rank=None, gen_heartbeat=False, num_epochs=300):
        '''
        Constructor. The optional ``freq`` parameter determines how often
        printing will occur. The value of 0 means an geometric
        schedule (1,2,4,...). A value > 0 means a arithmetic schedule
        (freq, 2*freq, 3*freq,...), and a value of None means no per-minibatch log.
        set log_to_file if you want the output to go file instead of stdout.
        set rank to distributed.rank if you are using distibuted parallelism -- each rank's log will go to seperate file.
        '''
        super(ProgressPrinter, self).__init__(freq, first)

        self.tag = '' if not tag else "[{}] ".format(tag)
        self.progress_timer_time = 0
        self.log_to_file = log_to_file
        self.rank = rank
        self.gen_heartbeat = gen_heartbeat
        self.num_epochs = num_epochs

        self.logfilename = None
        if self.log_to_file is not None:
            self.logfilename = self.log_to_file

            if self.rank != None:
                self.logfilename = self.logfilename + 'rank' + str(self.rank)

            # print to stdout
            print("Redirecting log to file " + self.logfilename)

            with open(self.logfilename, "w") as logfile:
                logfile.write(self.logfilename + "\n")

            self._logprint('CNTKCommandTrainInfo: train : ' + str(num_epochs))
            self._logprint('CNTKCommandTrainInfo: CNTKNoMoreCommands_Total : ' + str(num_epochs))
            self._logprint('CNTKCommandTrainBegin: train')

        if freq == 0:
            self._logprint(' average      since    average      since      examples')
            self._logprint('    loss       last     metric       last              ')
            self._logprint(' ------------------------------------------------------')

    def avg_loss_since_start(self):
        '''DEPRECATED.'''
        _warn_deprecated('The method was deprecated.')
        return self._avg_loss_since_start()

    def avg_metric_since_start(self):
        '''DEPRECATED.'''
        _warn_deprecated('The method was deprecated.')
        return self._avg_metric_since_start()

    def avg_loss_since_last(self):
        '''DEPRECATED.'''
        _warn_deprecated('The method was deprecated.')
        return self._avg_loss_since_last()

    def avg_metric_since_last(self):
        '''DEPRECATED.'''
        _warn_deprecated('The method was deprecated.')
        return self._avg_metric_since_last()

    def update(self, loss, minibatch_size, metric=None):
        '''
        DEPRECATED. Use :func:`cntk.utils.ProgressPrinter.update()` instead.

        Updates the accumulators using the loss, the minibatch_size and the optional metric.

        Args:
            loss (`float`): the value with which to update the loss accumulators
            minibatch_size (`int`): the value with which to update the samples accumulator
            metric (`float` or `None`): if `None` do not update the metric
             accumulators, otherwise update with the given value
        '''
        _warn_deprecated('Use ProgressPrinter.update_with_trainer() instead.')
        self._update_progress(loss, minibatch_size, metric)

    def epoch_summary(self, with_metric=False):
        '''
        DEPRECATED. Use :func:`cntk.utils.ProgressPrinter.summarize_progress()` instead.

        If on an arithmetic schedule print an epoch summary using the 'start' accumulators.
        If on a geometric schedule does nothing.

        Args:
            with_metric (`bool`): if `False` it only prints the loss, otherwise it prints both the loss and the metric
        '''
        _warn_deprecated('Use ProgressPrinter.summarize_progress() instead.')
        return self.summarize_progress(with_metric)

    def end_progress_print(self, msg=""):
        '''
        DEPRECATED. Use :func:`cntk.utils.ProgressPrinter.close()` instead.

        Prints the given message signifying the end of training.

        Args:
            msg (`string`): message to print.
        '''
        _warn_deprecated('Use ProgressPrinter.close() instead.')
        self.close(msg)

    def _update_progress(self, loss, minibatch_size, metric=None):
        # Override for BaseProgressWriter._update_progress.
        super(ProgressPrinter, self)._update_progress(loss, minibatch_size, metric)
        self._generate_progress_heartbeat()

    def _write_progress_update(self, first_mb, avg_loss, avg_metric, samples):
        # Override for BaseProgressWriter._write_progress_update.
        if self.freq == 0:
            if avg_metric is not None:
                self._logprint(' {:8.3g}   {:8.3g}   {:8.3g}   {:8.3g}    {:10d}'.format(
                    self.avg_loss_since_start(), avg_loss,
                    self.avg_metric_since_start(), avg_metric,
                    self.samples_since_start))
            else:
                self._logprint(' {:8.3g}   {:8.3g}   {:8s}   {:8s}    {:10d}'.format(
                    self.avg_loss_since_start(), avg_loss,
                    '', '', self.samples_since_start))
        else:
            if avg_metric is not None:
                self._logprint(' Minibatch[{:4d}-{:4d}]: loss = {:0.6f} * {:d}, metric = {:0.1f}% * {:d};'.format(
                    first_mb, self.updates_since_start, avg_loss, samples, avg_metric * 100.0, samples))
            else:
                self._logprint(' Minibatch[{:4d}-{:4d}]: loss = {:0.6f} * {:d};'.format(
                    first_mb, self.updates_since_start, avg_loss, samples))

    def _write_progress_summary(self, avg_loss, avg_metric, samples, time_delta):
        # Override for BaseProgressWriter._write_progress_summary.
        # Only log epoch summary when on arithmetic schedule.
        if self.freq == 0:
            return

        speed = samples / time_delta if time_delta > 0 else 0

        if avg_metric is not None:
            self._logprint("Finished Epoch[{} of {}]: {}loss = {:0.6f} * {}, metric = {:0.1f}% * {} {:0.3f}s ({:5.1f} samples per second);".format(
                self.epochs, self.num_epochs, self.tag, avg_loss, samples, avg_metric * 100.0, samples, time_delta, speed))
        else:
            self._logprint("Finished Epoch[{} of {}]: {}loss = {:0.6f} * {} {:0.3f}s ({:5.1f} samples per second);".format(
                self.epochs, self.num_epochs, self.tag, avg_loss, samples, time_delta, speed))

        return avg_loss, avg_metric, samples

    def _close(self, msg=""):
        # Override for BaseProgressWriter._close.
        if self.log_to_file is not None:
            # TODO: This is different from the previous implementation of end_progress_print() which always printed
            # the below line, even to console.
            self._logprint('CNTKCommandTrainEnd: train')
            if msg != "":
                self._logprint(msg)

    def _logprint(self, logline):
        if self.log_to_file is None:
            # to stdout.  if distributed, all ranks merge output into stdout
            print(logline)
        else:
            # to named file.  if distributed, one file per rank
            with open(self.logfilename, "a") as logfile:
                logfile.write(logline + "\n")

    def _generate_progress_heartbeat(self):
        timer_delta = time.time() - self.progress_timer_time

        # print progress no sooner than 10s apart
        if timer_delta > 10 and self.gen_heartbeat:
            # print to stdout
            print("PROGRESS: 0.00%")
            self.progress_timer_time = time.time()


class TensorBoardProgressWriter(BaseProgressWriter):
    '''
    Allows tracking various training time statistics (e.g. loss and metric) and write them as TensorBoard event files.
    The generated files can be opened in TensorBoard to visualize the progress.
    '''

    def __init__(self, freq=None, log_dir='.', model=None):
        '''
        Constructor.

        Args:
            freq (`int` or `None`, default None): frequency at which progress is logged.
              For example, the value of 2 will cause the progress to be logged every second time when
              `:func:cntk.util.TensorBoardFileWriter.update_with_trainer` is invoked.
              None indicates that progress is logged only when
              `:func:cntk.util.TensorBoardFileWriter.summarize_progress` is invoked.
              Must be a positive integer otherwise.
            log_dir (`string`, default '.'): directory where to create the TensorBoard event file.
            model (:class:`cntk.ops.Function` or `None`, default `None`): model graph to plot.
        '''
        if (freq is not None) and (freq <= 0):
            raise ValueError('frequency cannot be a negative number')

        super(TensorBoardProgressWriter, self).__init__(frequency=freq)

        from cntk import cntk_py
        self.writer = cntk_py.TensorBoardFileWriter(log_dir, model)

    def write_value(self, name, value, step):
        '''
        Record value of a scalar variable at the given time step.

        Args:
            name (`string`): name of a variable.
            value (`float`): value of the variable.
            step (`int`): time step at which the value is recorded.
        '''
        if self.writer is None:
            raise RuntimeError('Attempting to use a closed TensorBoardProgressWriter')

        self.writer.write_value(str(name), float(value), int(step))

    def flush(self):
        '''Make sure that any outstanding records are immediately persisted.'''
        if self.writer is None:
            raise RuntimeError('Attempting to use a closed TensorBoardProgressWriter')

        self.writer.flush()

    def _write_progress_update(self, first_mb, avg_loss, avg_metric, samples):
        # Override for BaseProgressWriter._write_progress_update.
        self.write_value('minibatch/avg_loss', avg_loss, self.total_updates)
        if avg_metric is not None:
            self.write_value('minibatch/avg_metric', avg_metric * 100.0, self.total_updates)

    def _write_progress_summary(self, avg_loss, avg_metric, samples, time_delta):
        # Override for BaseProgressWriter._write_progress_summary.
        self.write_value('summary/avg_loss', avg_loss, self.epochs)
        if avg_metric is not None:
            self.write_value('summary/avg_metric', avg_metric * 100.0, self.epochs)

    def _close(self, *args, **kwargs):
        # Override for BaseProgressWriter._close.
        if self.writer is None:
            raise RuntimeError('Attempting to use a closed TensorBoardProgressWriter')

        self.writer.close()
        self.writer = None


class ProgressWriters(object):
    '''Allows using multiple progress writers as if they were one writer.'''

    def __init__(self, *writers):
        '''
        Constructor.

        Args:
            *writers (:class:`cntk.utils.BaseProgressTracker`): list of progress writers to use.
        '''
        # Use frequency=1 to make sure underlying writers are invoked at each step.
        if not writers:
            raise ValueError("No progress writers specified")

        for writer in writers:
            if not isinstance(writer, BaseProgressWriter):
                raise ValueError("All ProgressTracker arguments must be derived from BaseProgressWriter")

        self.writers = writers

    def update_with_trainer(self, trainer, with_metric=False):
        '''
        Update all writers with the loss, the minibatch_size and optionally the metric from the ``trainer``.

        Args:
            trainer (:class:`cntk.trainer.Trainer`): trainer from which information is gathered.
            with_metric (`bool`): whether to update the metric accumulators.
        '''
        if self.writers is None:
            raise RuntimeError('Attempting to use a closed ProgressTracker')

        for writer in self.writers:
            writer.update_with_trainer(trainer, with_metric)

    def summarize_progress(self, with_metric=False):
        '''
        Invoke all writers to summarize progress so far.

        Args:
            with_metric (`bool`): whether the progress summary emitted by each writer should include the metric.
        '''
        if self.writers is None:
            raise RuntimeError('Attempting to use a closed ProgressTracker')

        for writer in self.writers:
            writer.summarize_progress(with_metric)

    def close(self, *args, **kwargs):
        '''
        Close all writers. Any subsequent attempt to use the object will cause a RuntimeError.
        '''
        if self.writers is None:
            raise RuntimeError('Attempting to use a closed ProgressTracker')

        for writer in self.writers:
            writer.close(*args, **kwargs)

        self.writers = None


# print the total number of parameters to log
def log_number_of_parameters(model, trace_level=0):
    parameters = model.parameters
    from functools import reduce
    from operator import add, mul
    total_parameters = reduce(add, [reduce(mul, p1.shape) for p1 in parameters], 0)
    # BUGBUG: If model has uninferred dimensions, we should catch that and fail here
    print("Training {} parameters in {} parameter tensors.".format(total_parameters, len(parameters)))
    if trace_level > 0:
        print()
        for p in parameters:
            print("\t{}".format(p.shape))