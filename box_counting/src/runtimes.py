#!/usr/bin/env python
import os
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from kernelio import get_kernel_manager

def compute_runtime(configuration_file, kerneldir, outputfile):
    print 'Loading configuration...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)

    if os.path.exists(outputfile):
        with open(outputfile, 'r+') as f:
            existing_lines = [line.strip() for line in f]
    else:
        existing_lines = []

    kernels = dict()
    for experiment in configuration['experiments']:
        dataset = experiment['dataset']
        ktype = experiment['ktype']
        epsilon = experiment['epsilon']
        delta = experiment['delta']
        seed = experiment['seed']

        line = '%s,%s,%f,%f,%d' % (dataset, ktype, epsilon, delta, seed)
        if any(l.startswith(line) for l in existing_lines):
            continue

        kernelfile = os.path.join(kerneldir, '%s_%s.db' % (dataset, ktype))
        kernel_manager = get_kernel_manager(kernelfile)
        time = kernel_manager.get_total_time(dataset, ktype, epsilon, delta, seed)
        line += (',%f\n' % time)
        print line,
        with open(outputfile, 'a+') as f:
            f.write(line)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile kerneldir outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    compute_runtime(*args, **options)
