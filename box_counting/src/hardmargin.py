#!/usr/bin/env python
import os
import numpy as np
from collections import defaultdict
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
try:
    from sklearn.metrics import roc_auc_score as auc_score
except:
    from sklearn.metrics import auc_score

from kernel_server import get_dset_size, Task

FOLDS = 10
FOLDDIR = os.path.join('..', 'folds')
FOLD_FORMAT = '%s.fold_%04d_of_%04d.%s.view'
TEST_SUFFIX = 'test'
TRAIN_SUFFIX = 'train'

IDXDIR = 'converted_datasets'
IDX_SUFFIX = 'idx'

LBLDIR = 'converted_datasets'
LBL_SUFFIX = 'db'

def accuracy(ytrue, ypred):
    wrong = np.logical_xor((ytrue > 0), (ypred > 0)).astype(int)
    return 1.0 - (float(np.sum(wrong)) / len(wrong))

def balanced_accuracy(ytrue, ypred):

    gold_pos = (ytrue > 0)
    true_pos = np.logical_and(gold_pos, (ypred > 0)).astype(int)
    sensitivity = float(np.sum(true_pos)) / np.sum(gold_pos)

    gold_neg = (ytrue <= 0)
    true_neg = np.logical_and(gold_neg, (ypred <= 0)).astype(int)
    specificity = float(np.sum(true_neg)) / np.sum(gold_neg)

    return (sensitivity + specificity) / 2.0

class CancelException(Exception): pass

TRAIN_DICT = defaultdict(list)
TEST_DICT  = defaultdict(list)
IDX_DICT = dict()
LBL_DICT = dict()

def get_labels(dataset):
    if dataset not in LBL_DICT:
        lblfile = os.path.join(LBLDIR, '%s.%s' % (dataset, LBL_SUFFIX))
        with open(lblfile, 'r') as f:
            LBL_DICT[dataset] = dict([(i, lbl) for i, lbl in enumerate([line.strip()[0] == '1' for line in f])])
    return LBL_DICT[dataset]

def get_indices(dataset):
    if dataset not in IDX_DICT:
        idxfile = os.path.join(IDXDIR, '%s.%s' % (dataset, IDX_SUFFIX))
        with open(idxfile, 'r') as f:
            IDX_DICT[dataset] = dict([(key, i) for i, key in enumerate([line.strip() for line in f])])
    return IDX_DICT[dataset]

def get_train_and_test(dataset, train_or_test):
    if dataset not in TRAIN_DICT:
        idx = get_indices(dataset)
        for f in range(FOLDS):
            train_file = os.path.join(FOLDDIR, FOLD_FORMAT % (dataset, f, FOLDS, TRAIN_SUFFIX))
            fold = []
            with open(train_file, 'r') as trainf:
                for line in trainf:
                    fold.append(idx[line.strip().split(',')[1]])
            TRAIN_DICT[dataset].append(fold)

            test_file = os.path.join(FOLDDIR, FOLD_FORMAT % (dataset, f, FOLDS, TEST_SUFFIX))
            fold = []
            with open(test_file, 'r') as testf:
                for line in testf:
                    fold.append(idx[line.strip().split(',')[1]])
            TEST_DICT[dataset].append(fold)

    if train_or_test == 'train':
        test = TRAIN_DICT[dataset]
    else:
        test = TEST_DICT[dataset]
    return TRAIN_DICT[dataset], test

def compute_prediction(t, kernel, train, true_labels):
    mantissa, exponent = kernel
    total = 0
    for tr in train:
        total += (2*true_labels[tr] - 1)*(mantissa[t, tr]/mantissa[tr, tr])*(10**(exponent[t, tr] - exponent[tr, tr]))
    return total

def compute_statistic(dataset, kernel, train_or_test, statistic):
    trains, tests = get_train_and_test(dataset, train_or_test)
    true_labels = get_labels(dataset)

    predictions = []
    actual = []
    for train, test in zip(trains, tests):
        for t in test:
            pred = compute_prediction(t, kernel, train, true_labels)
            predictions.append(pred)
            actual.append(true_labels[t])

    return statistic(np.asarray(actual), np.asarray(predictions))

STATISTICS = {
    'accuracy' : accuracy,
    'auc' : auc_score,
    'balanced_accuracy' : balanced_accuracy,
}

def compute_statistics(configuration_file, kerneldir, train_or_test, statistic, outputfile):
    if statistic not in STATISTICS:
        raise ValueError('Statistic must be one of: "%s"' % ', '.join(STATISTICS.keys()))
    statistic = STATISTICS[statistic]

    print 'Loading configuration...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)

    kernels = dict()
    for experiment in configuration['experiments']:
        dataset = experiment['dataset']
        ktype = experiment['ktype']
        epsilon = experiment['epsilon']
        delta = experiment['delta']
        seed = experiment['seed']

        n = get_dset_size(dataset)
        mantissa = np.zeros((n, n))
        exponent = np.zeros((n, n))

        try:
            for i in range(n):
                for j in range(i, n):
                    key = (dataset, ktype, epsilon, delta, seed, i, j)
                    task = Task(*key)
                    task.ground(kerneldir)
                    if not task.finished: raise CancelException()
                    mantissa[i, j], exponent[i, j] = task.value()
                    mantissa[j, i], exponent[j, i] = mantissa[i, j], exponent[i, j]
        except CancelException:
            continue

        kernels[dataset, ktype, epsilon, delta, seed] = (mantissa, exponent)

    if os.path.exists(outputfile):
        with open(outputfile, 'r+') as f:
            existing_lines = [line.strip() for line in f]
    else:
        existing_lines = []

    for key, kernel in kernels.items():
        (dataset, ktype, epsilon, delta, seed) = key
        line = '%s,%s,%f,%f,%d' % key
        if any(l.startswith(line) for l in existing_lines):
            continue

        stat = compute_statistic(dataset, kernel, train_or_test, statistic)

        line += (',%f\n' % stat)
        print line,
        with open(outputfile, 'a+') as f:
            f.write(line)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile kerneldir train_or_test statistic outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 5:
        parser.print_help()
        exit()
    compute_statistics(*args, **options)
