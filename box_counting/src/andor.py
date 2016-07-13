#!/usr/bin/env python
import os
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from kernel_server import get_dset_size, Task
from progress import ProgressMonitor

def compute_andor(configuration_file, kerneldir):
    print 'Loading configuration...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)

    kernels = dict()
    for experiment in configuration['experiments']:
        dataset = experiment['dataset']
        epsilon = experiment['epsilon']
        delta = experiment['delta']
        seed = experiment['seed']

        n = get_dset_size(dataset)
        mantissa = np.zeros((n, n))
        exponent = np.zeros((n, n))
        time = np.zeros((n, n))

        prog = ProgressMonitor(total=(n*(n+1)/2), msg='%s,andor,%f,%f,%d' % (dataset, epsilon, delta, seed))
        alldone = True
        for i in range(n):
            for j in range(i, n):
                prog.increment()
                andorkey = (dataset, 'andor', epsilon, delta, seed, i, j)
                andortask = Task(*andorkey)
                andortask.ground(kerneldir)
                if andortask.finished:
                    continue

                andkey = (dataset, 'and', epsilon, delta, seed, i, j)
                andtask = Task(*andkey)
                andtask.ground(kerneldir)
                if not andtask.finished:
                    alldone = False
                    continue

                orkey = (dataset, 'or', epsilon, delta, seed, i, j)
                ortask = Task(*orkey)
                ortask.ground(kerneldir)
                if not ortask.finished:
                    alldone = False
                    continue

                andtime = andtask.runtime()
                ortime = ortask.runtime()
                andman, andexp = andtask.value()
                orman, orexp = ortask.value()
                submission = {
                    'mantissa' : (andman / orman),
                    'exponent' : (andexp - orexp),
                    'time'     : (andtime + ortime),
                }
                andortask.store_results(submission)

        if not alldone:
            print 'Unfinished: %s, %f, %f, %d' % (dataset, epsilon, delta, seed)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile kerneldir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    compute_andor(*args, **options)
