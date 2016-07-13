#!/usr/bin/env python
from collections import defaultdict
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

BINS = 10

def safe_avg(l):
    if len(l) == 0: return 0
    else: return np.average(l)

def main(statfile, outputfile=None):
    maxstats = defaultdict(lambda: -1)
    stats = []
    with open(statfile, 'r+') as f:
        for line in f:
            dset, _, _, _, _, _, params, stat = line.strip().split(',')
            stat = float(stat)
            C, gamma, gamma2 = map(float,
                map(lambda s: s.split(':')[1], params.split('|')))
            maxstats[dset] = max(maxstats[dset], stat)
            stats.append((dset, np.log10(gamma), np.log10(gamma2), stat))

    stats = [(g, g2, s/maxstats[d]) for d, g, g2, s in stats]
    X, Y, Z = zip(*stats)
    bins = np.linspace(-5, 1, BINS)
    dX = np.digitize(np.asarray(X), bins)
    dY = np.digitize(np.asarray(Y), bins)

    binned = defaultdict(list)
    for dx, dy, z in zip(dX, dY, Z):
        binned[dx, dy].append(z)

    img = np.array([[safe_avg(binned[r, c]) for c in range(BINS)]
                    for r in range(BINS)])

    fig = pl.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='jet')

    if outputfile is None:
        pl.show()
    else:
        pdf = PdfPages(outputfile)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog statfile [outputfile]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 1 or len(args) > 2:
        parser.print_help()
        exit()
    main(*args, **options)
