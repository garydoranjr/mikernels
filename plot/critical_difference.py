#!/usr/bin/env python
import pylab as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

from nemenyi import critical_difference as cd

ALPHA = 0.05

TMAP = {
'twolevel': 'MMD',
'miGraph' : 'mi-Graph',
'MIGraph' : 'MI-Graph',
'nsk' : 'NSK',
'pmir': 'PMIR',
'sil_av': 'SIL-AVG',
'sil': 'SIL-MED',
'emd': 'EMD',
'yards': 'YARDS',
'miles': 'MILES',
'box': 'Box',
# Specific Box Kernels:
'and': '$k_{\\wedge}$',
'andor': '$k_{\\wedge/\\vee}$',
'and-emp': '$k_{\\wedge}^{emp}$',
'andor-emp': '$k_{\\wedge/\\vee}^{emp}$',
'and-emp-trans': '$k_{\\wedge}^{emp{-}trans}$',
'andor-emp-trans': '$k_{\\wedge/\\vee}^{emp{-}trans}$',
}

PAD = 0.5
LPAD = 0.1
GPAD = 0.05
FONT = 30
XPAD = 1.0

def critical_difference(algorithms, ranks, cd):
    n = len(algorithms)
    techniques = sorted(zip(ranks, algorithms))[::-1]
    groups = []
    last_group = set()
    for rank, algorithm in techniques:
        group = sorted([(r, a) for r, a in techniques if r >= rank - cd and rank >= r])
        if group[0] not in last_group:
            groups.append(group)
            last_group = set(group)

    # Filter singleton groups
    groups = [g for g in groups if len(g) > 1]

    fig = pl.figure(figsize=(10, 3))
    ax = pl.axes([0, 0, 1.0, 0.75])
    for i, g in enumerate(groups, 1):
        g_ranks = [r for r, a in g]
        start = min(g_ranks) - GPAD
        end = max(g_ranks) + GPAD
        y = i*(1.0/(len(groups) + 1))
        ax.plot([start, end], [-y, -y], 'k-', lw=2)

    # Determine whether to put the odd technique
    # on the left or on the right
    if n % 2 == 0:
        midpoint = (n / 2)
    elif ((techniques[n/2 - 1][0] - techniques[n/2][0])
        < (techniques[n/2][0] - techniques[n/2 + 1][0])):
        midpoint = (n / 2) + 1
    else:
        midpoint = (n / 2)

    techniques = techniques[:midpoint] + techniques[-1:midpoint-1:-1]
    for i, (rank, algorithm) in enumerate(techniques):
        if i < midpoint:
            ax.text(max(ranks) + PAD, -(i + 1), algorithm, horizontalalignment='right', verticalalignment='center', fontsize=FONT)
            ax.plot([rank, max(ranks) + (PAD - LPAD)], [-(i + 1), -(i + 1)], 'k-', lw=1)
            ax.plot([rank, rank], [-(i + 1), 0], 'k-', lw=1)
        else:
            ax.text(min(ranks) - PAD, -(i - (n/2) + 1), algorithm, horizontalalignment='left', verticalalignment='center', fontsize=FONT)
            ax.plot([rank, min(ranks) - (PAD - LPAD)], [-(i - (n/2) + 1), -(i - (n/2) + 1)], 'k-', lw=1)
            ax.plot([rank, rank], [-(i - (n/2) + 1), 0], 'k-', lw=1)
    ax.xaxis.set_ticks_position('top')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks(range(1, n+1))
    xleft = max(n + XPAD, max(ranks) + PAD + XPAD)
    xright = 1.0 - (xleft - n)
    ax.set_xlim(xleft, xright)
    ax.set_ylim(-n/2 - 1, 0)
    pl.setp(ax.get_xticklabels(), fontsize=FONT)

    ax2 = pl.axes([0, 0.9, 1.0, 0.08])
    ax2.axis('off')
    ax2.plot([n, n - cd], [0, 0], 'k|-')
    ax2.text(n + 2*LPAD, -0.04, 'CD', horizontalalignment='right', verticalalignment='center', fontsize=FONT)
    ax2.set_xlim(xleft, xright)
    ax2.yaxis.set_ticks([])
    ax2.xaxis.set_ticks([])
    ax2.spines['left'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    return fig

def main(rank_file, outputfile=None):
    ts = []
    rs = []
    dsets = None
    with open(rank_file, 'r') as f:
        for line in f:
            t, d, r = line.strip().split(',')
            #if t not in TMAP: continue
            ts.append(TMAP.get(t, t))
            rs.append(float(r))
            if dsets is None:
                dsets = int(d)
    fig = critical_difference(ts, rs, cd(ALPHA, len(ts), dsets))

    if outputfile is None:
        pl.show()
    else:
        pdf = PdfPages(outputfile)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog rank-file [outputfile]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 1 or len(args) > 2:
        parser.print_help()
        exit()
    main(*args, **options)
