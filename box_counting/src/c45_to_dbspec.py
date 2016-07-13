#!/usr/bin/env python
import os
import numpy as np
from collections import defaultdict

from mldata import parse_c45 as load_c45

OUTPUTDIR = 'converted_datasets'

def main(dataset):
    dset = load_c45(dataset, rootdir='scaled_datasets')
    X = np.array(dset.to_float(mapper=lambda x: x[2:-1]), dtype=int)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    fname = os.path.join(OUTPUTDIR, '%s.spec' % dataset)
    with open(fname, 'w+') as f:
        f.write('2\n')
        f.write('0 1\n')
        f.write('0\n')
        f.write('%d\n' % (len(dset.schema) - 3))
        for mn, mx in zip(mins, maxs):
            f.write('%d\t%d\n' % (mn, mx))

    indices = defaultdict(list)
    labels = defaultdict(list)
    for i, ex in enumerate(dset):
        k = ex[0]
        labels[k].append(ex[-1])
        indices[k].append(i)

    fname = os.path.join(OUTPUTDIR, '%s.db' % dataset)
    with open(fname, 'w+') as f:
        for k in sorted(indices.keys()):
            l = int(any(labels[k]))
            n = len(indices[k])
            vals = []
            for i in indices[k]:
                for v in X[i, :]:
                    vals.append(str(v))
            vals = ' '.join(vals)
            f.write('%d\t %d\t %s\n' % (l, n, vals))

    fname = os.path.join(OUTPUTDIR, '%s.idx' % dataset)
    with open(fname, 'w+') as f:
        for k in sorted(indices.keys()):
            f.write('%s\n' % k)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog dataset")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 1:
        parser.print_help()
        exit()
    main(*args, **options)

