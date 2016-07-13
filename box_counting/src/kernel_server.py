#!/usr/bin/env python
import os
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import random
import cherrypy
from cherrypy import expose, HTTPError
from threading import RLock
from collections import defaultdict
from random import shuffle
import numpy as np

from kernelio import get_kernel_manager

DSETS = 'converted_datasets'
IDX_SUFFIX = 'idx'

PORT = 2114
DEFAULT_TASK_EXPIRE = 600 # Seconds
TEMPLATE = """
<html>
<head>
  <META HTTP-EQUIV="REFRESH" CONTENT="60">
  <title>%s</title>
  <style type="text/css">
    table.status {
      border-width: 0px;
      border-spacing: 0px;
      border-style: none;
      border-color: black;
      border-collapse: collapse;
      background-color: white;
      margin-left: auto;
      margin-right: auto;
    }
    table.status td {
        border-width: 1px;
        padding: 1px;
        border-style: solid;
        border-color: black;
        text-align: center;
    }
    table.summary {
      border-width: 0px;
      border-spacing: 0px;
      border-style: none;
      border-color: none;
      border-collapse: collapse;
      background-color: white;
      margin-left: auto;
      margin-right: auto;
    }
    table.summary td {
        border-width: 0px;
        padding: 3px;
        border-style: none;
        border-color: black;
        text-align: center;
        width: 50px;
    }
    td.tech { width: 50px; }
    td.done {
      background-color: green;
    }
    td.pending {
      background-color: yellow;
    }
    td.failed {
      background-color: red;
    }
    td.na {
      background-color: gray;
    }
  </style>
</head>
<body>
<h1>Time Remaining: %s</h1>
%s
</body>
</html>
"""

class UnfinishedException(Exception): pass

def plaintext(f):
    f._cp_config = {'response.headers.Content-Type': 'text/plain'}
    return f

class ExperimentServer(object):

    def __init__(self, tasks, render,
                 task_expire=DEFAULT_TASK_EXPIRE):
        self.status_lock = RLock()
        self.tasks = tasks
        self.render = render
        self.task_expire = task_expire

        self.unfinished = set(self.tasks.items())

    def clean(self):
        with self.status_lock:
            self.unfinished = filter(lambda x: (not x[1].finished),
                                     self.unfinished)
            for key, task in self.unfinished:
                if (task.in_progress and
                    task.staleness() > self.task_expire):
                    task.quit()

    @expose
    def index(self):
        with self.status_lock:
            self.clean()
            return self.render(self.tasks)

    @plaintext
    @expose
    def request(self):
        with self.status_lock:
            self.clean()
            # Select a job to perform
            unfinished = list(self.unfinished)
            shuffle(unfinished)
            candidates = sorted(unfinished, key=lambda x: x[1].priority())
            if len(candidates) == 0:
                raise HTTPError(404)
            key, task = candidates.pop(0)
            task.ping()

        arguments = {'key': key}
        return yaml.dump(arguments, Dumper=Dumper)

    @plaintext
    @expose
    def update(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.ping()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def quit(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.quit()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def fail(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.fail()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def submit(self, key_yaml=None, sub_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
            submission = yaml.load(sub_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.store_results(submission)
                task.finish()
        return "OK"

def time_remaining_estimate(tasks, alpha=0.1):
    to_go = float(len([task for task in tasks if not task.finished]))
    finish_times = sorted([task.finish_time for task in tasks if task.finished])
    ewma = 0.0
    for interarrival in np.diff(finish_times):
        ewma = alpha*interarrival + (1.0 - alpha)*ewma

    if ewma == 0:
        return '???'

    remaining = to_go * ewma
    if remaining >= 604800:
        return '%.1f weeks' % (remaining/604800)
    elif remaining >= 86400:
        return '%.1f days' % (remaining/86400)
    elif remaining >= 3600:
        return '%.1f hours' % (remaining/3600)
    elif remaining >= 60:
        return '%.1f minutes' % (remaining/60)
    else:
        return '%.1f seconds' % remaining

def render(tasks):
    experiment_title = 'Status: box kernel'

    time_est = time_remaining_estimate(tasks.values())

    reindexed = defaultdict(list)
    for k, v in tasks.items():
        reindexed[k[:-2]].append(v)

    tasks = reindexed
    experiment_ids = sorted(tasks.keys())

    table = '<table class="status">'
    # Experiment header row
    table += '<tr><td style="border:0" rowspan="1">Kernel</td>'
    table += '<td class="tech">Progress</td>'
    table += '</tr>\n'

    # Data rows
    for experiment_id in sorted(experiment_ids):
        table += ('<tr><td class="data">%s</td>' % str(experiment_id))
        if experiment_id in tasks:
            table += ('<td style="padding: 0px;">%s</td>' % render_task_summary(tasks[experiment_id]))
        else:
            table += '<td class="na"></td>'
        table += '</tr>\n'

    table += '</table>'
    return (TEMPLATE % (experiment_title, time_est, table))

def render_task_summary(tasks):
    n = float(len(tasks))
    failed = 0
    finished = 0
    in_progress = 0
    waiting = 0
    for task in tasks:
        if task.finished:
            finished += 1
        elif task.failed:
            failed += 1
        elif task.in_progress:
            in_progress += 1
        else:
            waiting += 1

    if n == finished:
        table = '<table class="summary"><tr>'
        table += ('<td class="done" title="Finished">D</td>')
        table += ('<td class="done" title="Finished">O</td>')
        table += ('<td class="done" title="Finished">N</td>')
        table += ('<td class="done" title="Finished">E</td>')
        table += '</tr></table>'
    else:
        table = '<table class="summary"><tr>'
        table += ('<td title="Waiting">%.2f%%</td>' % (100*waiting/n))
        table += ('<td class="failed" title="Failed">%.2f%%</td>' % (100*failed/n))
        table += ('<td class="pending" title="In Progress">%.2f%%</td>' % (100*in_progress/n))
        table += ('<td class="done" title="Finished">%.2f%%</td>' % (100*finished/n))
        table += '</tr></table>'
    return table

class Task(object):

    def __init__(self, dataset, ktype, epsilon, delta, seed, i, j):
        self.dataset = dataset
        self.ktype = ktype
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.i = i
        self.j = j

        self.priority_adjustment = 0

        self.grounded = False

        self.last_checkin = None
        self.finished = False
        self.failed = False
        self.in_progress = False

        self.finish_time = None

    def ground(self, kerneldir):
        self.kernelfile = os.path.join(kerneldir, '%s_%s.db' % (self.dataset, self.ktype))

        self.kernel_manager = get_kernel_manager(self.kernelfile)
        if self.kernel_manager.is_finished(self.dataset, self.ktype,
            self.epsilon, self.delta, self.seed, self.i, self.j):
            self.finish()

        self.grounded = True

    def value(self):
        return self.kernel_manager.get_kernel(self.dataset, self.ktype,
                 self.epsilon, self.delta,
                 self.seed, self.i, self.j)

    def runtime(self):
        return self.kernel_manager.get_time(self.dataset, self.ktype,
                 self.epsilon, self.delta,
                 self.seed, self.i, self.j)

    def store_results(self, submission):
        """Write results to disk."""
        if not self.grounded:
            raise Exception('Task not grounded!')

        self.kernel_manager.store_kernel(submission,
            self.dataset, self.ktype, self.epsilon, self.delta,
            self.seed, self.i, self.j)

    def ping(self):
        if not self.finished:
            self.in_progress = True
            self.last_checkin = time.time()

    def quit(self):
        if not self.finished:
            self.in_progress = False
            self.last_checkin = None

    def fail(self):
        if not self.finished:
            self.failed = True
            self.in_progress = False

    def staleness(self):
        return time.time() - self.last_checkin

    def priority(self):
        return (10000*int(self.in_progress) + 1000*int(self.failed)
                + self.priority_adjustment)

    def finish(self):
        self.finished = True
        self.in_progress = False
        self.failed = False
        self.finish_time = time.time()

def start_experiment(configuration_file, kernel_file):
    task_dict = load_config(configuration_file, kernel_file)

    server = ExperimentServer(task_dict, render)
    cherrypy.config.update({'server.socket_port': PORT,
                            'server.socket_host': '0.0.0.0'})
    cherrypy.quickstart(server)

def get_dset_size(dataset):
    fname = os.path.join(DSETS, '%s.%s' % (dataset, IDX_SUFFIX))
    with open(fname, 'r') as f:
        n = len([line for line in f])
    return n

def load_config(configuration_file, kerneldir):
    tasks = {}

    print 'Loading configuration...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)

    for experiment in configuration['experiments']:
        dataset = experiment['dataset']
        ktype = experiment['ktype']
        epsilon = experiment['epsilon']
        delta = experiment['delta']
        seed = experiment['seed']
        priority = experiment.get('priority', 0)

        n = get_dset_size(dataset)

        for i in range(n):
            for j in range(i, n):
                key = (dataset, ktype, epsilon, delta, seed, i, j)
                task = Task(*key)
                task.priority_adjustment = priority
                task.ground(kerneldir)
                tasks[key] = task

    return tasks

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile kerneldir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    start_experiment(*args, **options)
