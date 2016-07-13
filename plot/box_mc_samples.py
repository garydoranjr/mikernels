#!/usr/bin/env python
import numpy as np
import pylab as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

SPACING = 0.001

MAX_EPS = 0.25
MAX_DELTA = 0.1

N_LEVELS = 6
MAX_S = 1e10

def main(n, outputfile=None, title='Samples'):
    n = int(n)
    eps = np.arange(0+SPACING, MAX_EPS, SPACING)
    delta = np.arange(0+SPACING, MAX_DELTA, SPACING)
    E, D = np.meshgrid(eps, delta)
    S = np.ceil(8*(1+E)*(n**2)*np.log(2/D)/(E**2))
    best_case = np.min(S)
    print '%1.2e' % np.ceil(16*(n**2)*np.log(2))
    levels = np.exp(np.linspace(np.log(1.5*best_case), np.log(MAX_S), N_LEVELS))

    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    cont = ax.contour(E, D, S, levels, colors='k')
    pl.clabel(cont, inline=True, fmt='%.1e', fontsize=15)
    ax.set_xlabel(r'$\epsilon$', fontsize=20)
    ax.set_ylabel(r'$\delta$', fontsize=20)
    ax.set_title(title)

    if outputfile is None:
        pl.show()
    else:
        pdf = PdfPages(outputfile)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog bag-size [outputfile title]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 1 or len(args) > 3:
        parser.print_help()
        exit()
    main(*args, **options)
