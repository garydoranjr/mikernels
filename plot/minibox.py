#!/usr/bin/env python
import os
from collections import defaultdict
import pylab as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

FONTSIZE=18
DATAFILE = os.path.join('data', 'natural_scene.data')
CLASSES = ['desert', 'mountains', 'sea', 'sunset', 'trees']
COLORS = {
    (0, 0) : 'darkred',
    (0, 1) : 'darkred',
    (1, 1) : 'darkred',
    (1, 0) : 'darkgreen',
}
ADJ = 0.05
TFONTSIZE=14
CPOS = {
    (0, 0) : (0.5, 0.5),
    (0, 1) : (0.0, 0.5),
    (1, 1) : (0.0, 1.0),
    (1, 0) : (0.5, 1.0),
}
TPOS = {
    (1, 1) : (0.5, 0.5),
    (0, 0) : (1.0, 0.0),
    (1, 0) : (1.0, 0.5),
    (0, 1) : (0.5, 0.0),
}

def parse_labels(label_str):
    return dict([map(int, l.split(':')) for l in label_str.split('|')])

def parse_dset(dataset):
    parts = dataset.split('_')
    return parts[0], parts[-1]

def parse_nsk_results(parts, kernel='rbf'):
    dataset, _, _, k, normalization, stat = parts
    if k != kernel: return None
    if normalization != 'averaging': return None
    return parse_dset(dataset), parse_labels(stat)

def parse_twolevel_results(parts, kernel='rbf'):
    dataset, _, _, k, second_level, stat = parts
    if k != kernel: return None
    if second_level != 'rbf': return None
    return parse_dset(dataset), parse_labels(stat)

def intuit_rows(labelfile):
    if 'nsk' in labelfile:
        return parse_nsk_results
    elif 'twolevel2' in labelfile:
        return parse_twolevel_results
    else:
        raise ValueError('Cannot figure out "%s"' % labelfile)

def get_truth(datafile=DATAFILE):
    labels = defaultdict(dict)
    with open(datafile, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            labels[int(parts[0])] = map(int, parts[-5:])
    return labels

def main(labelfile, outputfile=None):
    parser = intuit_rows(os.path.basename(labelfile))

    data = {}
    with open(labelfile, 'r') as f:
        for line in f:
            dset, labels = parser(line.strip().split(','))
            data[dset] = labels

    truth = get_truth()

    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(lambda: defaultdict(int))
    for ci in range(len(CLASSES)):
        for cj in range(len(CLASSES)):
            if ci == cj: continue
            dset = (CLASSES[ci], CLASSES[cj])
            for k, v in data[dset].items():
                a = truth[k][ci]
                b = truth[k][cj]
                totals[dset][a, b] += 1
                if int(v):
                    counts[dset][a, b] += 1

    fig = pl.figure(figsize=(12, 12))
    fig.suptitle('replulsive', fontsize=FONTSIZE)
    for ci in range(len(CLASSES)):
        for cj in range(len(CLASSES)):
            if ci == cj: continue
            dset = (CLASSES[ci], CLASSES[cj])
            ax = fig.add_subplot(len(CLASSES), len(CLASSES), ci*len(CLASSES) + cj + 1)
            if (ci == 1 and cj == 0) or (ci == 0):
                ax.set_xticks([0.25, 0.75])
                ax.set_xticklabels(['Yes', 'No'])
                ax.set_xlabel(CLASSES[cj], fontsize=FONTSIZE)
            else:
                ax.set_xticks([])

            if (cj == 1 and ci == 0) or (cj == 0):
                ax.set_yticks([0.25, 0.75])
                ax.set_yticklabels(['No', 'Yes'])
                ax.set_ylabel(CLASSES[ci], fontsize=FONTSIZE)
            else:
                ax.set_yticks([])

            for a in (0, 1):
                for b in (0, 1):
                    cpos = CPOS[a, b]
                    tpos = TPOS[a, b]
                    ax.text(cpos[0] + ADJ, cpos[1] - ADJ, str(counts[dset][a, b]), color=COLORS[a, b], fontsize=TFONTSIZE, va='top', ha='left')
                    ax.text(tpos[0] - ADJ, tpos[1] + ADJ, str(totals[dset][a, b]), color='k', fontsize=TFONTSIZE, va='bottom', ha='right')

            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.plot([0.5, 0.5], [0, 1], 'k-')
            ax.plot([0, 1], [0.5, 0.5], 'k-')
            ax.plot([0, 0.5], [0.5, 1.0], 'k-')
            ax.plot([0.5, 1.0], [0, 0.5], 'k-')
            ax.plot([0, 1], [0, 1], 'k-')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(axis=u'both', which=u'both',length=0)

    if outputfile is None:
        pl.show()
    else:
        pdf = PdfPages(outputfile)
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog labelfile [outputfile]")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 1 or len(args) > 2:
        parser.print_help()
        exit()
    main(*args, **options)
