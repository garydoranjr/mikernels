#!/usr/bin/env python
import os
import numpy as np
import pylab as pl
from collections import defaultdict

DATA_DIR = 'data'
NAT = 'data/natural_scene.data'
NAT_NAMES = 'data/natural_scene.names'
CLASSES = [
'desert',
'mountains',
'sea',
'sunset',
'trees',
]
NAT_N = len(CLASSES)

def main():
    with open(NAT, 'r') as f:
        data = [line.strip().split(',') for line in f]

    with open(NAT_NAMES, 'r') as f:
        names_file = list(f)
    names_file = ''.join(names_file[:-NAT_N])

    labels = np.array([d[-NAT_N:] for d in data], dtype=int).astype(bool)

    for i in range(1, NAT_N + 1):
        for j in range(1, NAT_N + 1):
            if i == j: continue
            ci = CLASSES[-i]
            cj = CLASSES[-j]
            basename = ('%s_no_%s' % (ci, cj))
            namesfilename = os.path.join(DATA_DIR, basename + '.names')
            datafilename  = os.path.join(DATA_DIR, basename + '.data')
            datalines = []
            pos = 0
            for di, li in zip(data, labels):
                datalines.append(','.join(di[:-NAT_N]))
                label = int(li[-i] & (li[-j] == 0))
                if label > 0: pos += 1
                datalines[-1] = ('%s,%d\n' % (datalines[-1], label))
            datastr = ''.join(datalines)
            with open(namesfilename, 'w+') as f: f.write(names_file)
            with open(datafilename, 'w+') as f: f.write(datastr)
    exit()

    labels = np.array(dict([(int(d[0]), d[-NAT_N:]) for d in data]).values(), dtype=int).astype(bool)

    counts = np.array(
    [[np.sum(labels[:, i] & labels[:, j]) for i in range(NAT_N)]
        for j in range(NAT_N)])
    counts2 = np.array(
    [[np.sum(labels[:, i] & (labels[:, j] == 0)) for i in range(NAT_N)]
        for j in range(NAT_N)])
    counts3 = np.array(
    [[np.sum(labels[:, i] == 0) for i in range(NAT_N)]
        for j in range(NAT_N)])

    pos = counts2
    neg = (counts + counts3)
    print pos
    print neg
    print np.sort(pos.astype(float) / neg)

if __name__ == '__main__':
    main()
