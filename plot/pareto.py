#!/usr/bin/env python
import pylab as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

from critical_difference import TMAP
FONTSIZE = 20

def intuit_metric(rank_file):
    if 'bacc' in rank_file:
        return 'Balanced Accuracy'
    elif 'auc' in rank_file:
        return 'AUC'
    else:
        return 'Accuracy'

def main(acc_rank_file, time_rank_file, outputfile=None):
    metric = intuit_metric(acc_rank_file)
    acc_ranks = {}
    with open(acc_rank_file, 'r+') as f:
        for line in f:
            t, d, r = line.strip().split(',')
            acc_ranks[t, int(d)] = float(r)
    time_ranks = {}
    with open(time_rank_file, 'r+') as f:
        for line in f:
            t, d, r = line.strip().split(',')
            time_ranks[t, int(d)] = float(r)

    diff = set(acc_ranks.keys()) - set(time_ranks.keys())
    if len(diff) > 0:
        missing = ', '.join([('(%s, %d)' % p) for p in diff])
        raise ValueError('Missing "%s" from time ranks.' % missing)
    diff = set(time_ranks.keys()) - set(acc_ranks.keys())
    if len(diff) > 0:
        missing = ', '.join([('(%s, %d)' % p) for p in diff])
        raise ValueError('Missing "%s" from performance ranks.' % missing)

    algs = [TMAP.get(t, t) for t, d in sorted(acc_ranks.keys())]
    aranks = [acc_ranks[t, d] for t, d in sorted(acc_ranks.keys())]
    tranks = [time_ranks[t, d] for t, d in sorted(acc_ranks.keys())]
    ranks = zip(aranks, tranks)
    infrontier = [not any(((ao < a) and (to < t)) for ao, to in ranks)
                  for a, t in ranks]
    frontier = sorted([r for r, i in zip(ranks, infrontier) if i] + [(len(algs), 1), (1, len(algs))])

    fig = pl.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    for alg, (arank, trank), f in zip(algs, ranks, infrontier):
        #if   alg == 'mi-Graph': yadj = -0.3
        #elif alg == 'YARDS'   : yadj = -0.5
        #else: yadj = 0
        yadj = 0
        if f:
            ax.text(arank, trank + yadj, alg, va='bottom', ha='left', fontsize=FONTSIZE)
        else:
            ax.text(arank, trank + yadj, alg, va='bottom', ha='left', fontsize=FONTSIZE, alpha=0.75)

    afront, tfront = zip(*frontier)
    ax.plot(afront, tfront, 'k--', lw=3)
    if metric == 'Accuracy':
        ax.annotate('Pareto Frontier', frontier[-2], (1, 1),
                    ha="left", va="top", size=FONTSIZE-3,
                    arrowprops=dict(arrowstyle='wedge,tail_width=0.1', lw=1,
                                    fc="k", ec="k",
                                    connectionstyle="arc3,rad=-0.05"))

    ax.axis('equal')
    ax.set_xlabel(metric + ' Rank', fontsize=FONTSIZE)
    ax.set_ylabel('Training Time Rank', fontsize=FONTSIZE)
    ax.set_xlim(0, len(algs) + 1)
    ax.set_xticks(range(1, len(algs)+1))
    ax.set_ylim(0, len(algs) + 1)
    ax.set_yticks(range(1, len(algs)+1))
    pl.tick_params(axis='both', which='major', labelsize=FONTSIZE-3)

    if outputfile is None:
        pl.show()
    else:
        pdf = PdfPages(outputfile)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog perf-rank-file time-rank-file [outputfile]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 2 or len(args) > 3:
        parser.print_help()
        exit()
    main(*args, **options)
