#!/usr/bin/env python
import numpy as np
from scipy.stats import pearsonr
import pylab as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

FONTSIZE = 20
KERNEL = 'rbf'

def intuit_metric(rank_file):
    if 'bacc' in rank_file:
        return 'Balanced Accuracy'
    elif 'auc' in rank_file:
        return 'AUC'
    else:
        return 'Accuracy'

def parse_oracle_results(parts):
    dataset, _, _, k, stat = parts
    if k != KERNEL: return None
    if not dataset.endswith('+oracle'): return None
    return dataset[:-7], float(stat)

def parse_nsk_results(parts):
    dataset, _, _, k, normalization, stat = parts
    if k != KERNEL: return None
    if normalization != 'averaging': return None
    return dataset, float(stat)

def parse_twolevel_results(parts):
    dataset, _, _, k, second_level, stat = parts
    if k != KERNEL: return None
    if second_level != KERNEL: return None
    return dataset, float(stat)

def main(nsk_results, twolevel_results, oracle_results, outputfile=None):
    metric = intuit_metric(oracle_results)

    nsk_stats = dict()
    twolevel_stats = dict()
    oracle_stats = dict()

    for stats_file, stats_dict, parser in zip(
            (nsk_results, twolevel_results, oracle_results),
            (nsk_stats,   twolevel_stats,   oracle_stats),
            (parse_nsk_results, parse_twolevel_results, parse_oracle_results)):
        with open(stats_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                results = parser(parts)
                if results is None: continue
                dset, stat = results
                stats_dict[dset] = stat

    good_datasets = sorted((set(nsk_stats.keys())
                          & set(twolevel_stats.keys())
                          & set(oracle_stats.keys())))
    print '%d datasets.' % len(good_datasets)

    X = [(1 - oracle_stats[d]) for d in good_datasets]
    Y = [(-(nsk_stats[d] - twolevel_stats[d])/nsk_stats[d]) for d in good_datasets]
    r, pval = pearsonr(X, Y)
    print 'R: %f' % r
    print 'PVAL: %f' % pval

    fig = pl.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(X, Y, 'ko')

    if outputfile is None:
        pl.show()
    else:
        pdf = PdfPages(outputfile)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog nsk-stats twolevel-stats oracle-stats [outputfile]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 3:
        parser.print_help()
        exit()
    main(*args, **options)
