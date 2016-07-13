#!/usr/bin/env python
from collections import defaultdict
import numpy as np
import pylab as pl
from scipy.stats import pearsonr
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

def main(nskstatfile, twolevelstatfile, outputfile=None):
    nskstats = {}
    with open(nskstatfile, 'r+') as f:
        for line in f:
            dset, alg, bk, ik, norm, stat = line.strip().split(',')
            if alg != 'svm': continue
            if bk != 'nsk': continue
            if ik != 'rbf': continue
            if norm != 'averaging': continue
            nskstats[dset] = float(stat)

    twolevelstats = {}
    with open(twolevelstatfile, 'r+') as f:
        for line in f:
            dset, alg, bk, ik, ik2, stat = line.strip().split(',')
            if alg != 'svm': continue
            if bk != 'twolevel': continue
            if ik != 'rbf': continue
            if ik2 != 'rbf': continue
            twolevelstats[dset] = float(stat)

    dsets = sorted(set(nskstats.keys()) & set(twolevelstats.keys()))
    print len(dsets)

    x = np.array([nskstats[d] for d in dsets])
    y = np.array([twolevelstats[d] for d in dsets])
    A = np.vstack([x, np.ones(x.shape)]).T
    m, c = np.linalg.lstsq(A, y)[0]
    r, pval = pearsonr(x, y)
    print 'm: %f' % m
    print 'c: %f' % c
    print 'R: %f' % r
    print 'PVAL: %f' % pval

    fig = pl.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'ko')

    xx = np.array([0, 120])
    ax.plot(xx, (m*xx + c), ':', color='darkred', lw=2,
        label=('Least Squares (R = %.2f)' % r))

    ax.plot(xx, xx, 'k-', lw=2)

    ax.set_xlabel('NSK Training Time')
    ax.set_ylabel('Level-2 Training Time')

    if outputfile is None:
        pl.show()
    else:
        pdf = PdfPages(outputfile)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog nsk-statfile twolevel-statfile [outputfile]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 2 or len(args) > 3:
        parser.print_help()
        exit()
    main(*args, **options)
