"""
Implements the actual client function to run the experiment
"""
import os
import numpy as np
import time

from data import get_dataset
from mi_svm import MIKernelSVM, MIKernelSVR
from vocabulary import EmbeddedSpaceSVM

INSTANCE_PREDICTIONS = False

CLASSIFIERS = {
    'svm': MIKernelSVM,
    'svr': MIKernelSVR,
    'embedded_svm' : EmbeddedSpaceSVM,
}

IDX_DIR = os.path.join('box_counting', 'converted_datasets')
PRECOMPUTED_DIR = os.path.join('box_counting', 'precomputed')
IDX_FMT = '%s.idx'
PRECOMPUTED_FMT = '%s_%s.db'

def get_base_dataset(train):
    parts = train.split('.')
    i = 1
    while not parts[i].startswith('fold_'):
        i += 1
    return '.'.join(parts[:i])

class Timer(object):

    def __init__(self):
        self.starts = {}
        self.stops = {}

    def start(self, event):
        self.starts[event] = time.time()

    def stop(self, event):
        self.stops[event] = time.time()

    def get(self, event):
        return self.stops[event] - self.starts[event]

    def get_all(self, suffix=''):
        times = {}
        for event in self.stops.keys():
            times[event + suffix] = self.get(event)
        return times

def client_target(task, callback):
    (experiment_name, experiment_id,
     train_dataset, test_dataset, _, _) = task['key']
    parameters = task['parameters']

    print 'Starting task %s...' % str(experiment_id)
    print 'Training Set: %s' % train_dataset
    print 'Test Set:     %s' % test_dataset
    print 'Parameters:'
    for k, v in parameters.items():
        print '\t%s: %s' % (k, str(v))

    train = get_dataset(train_dataset)
    test = get_dataset(test_dataset)

    submission = {
        'instance_predictions' : {
            'train' : {},
            'test'  : {},
        },
        'bag_predictions' : {
            'train' : {},
            'test'  : {},
        },
        'statistics' : {}
    }
    timer = Timer()

    if parameters['kernel'] == 'emp':
        dataset = get_base_dataset(train_dataset)
        idxfile = os.path.join(IDX_DIR, IDX_FMT % dataset)
        kernelfile = os.path.join(PRECOMPUTED_DIR,
            PRECOMPUTED_FMT % (dataset, parameters['ktype']))
        parameters['dataset'] = dataset
        parameters['idxfile'] = idxfile
        parameters['kernelfile'] = kernelfile
        empirical_labels = list(map(str, train.bag_ids))
        if parameters.pop('transductive', False):
            empirical_labels += list(map(str, test.bag_ids))
        parameters['empirical_labels'] = empirical_labels
        train.bags = train.bag_ids
        test.bags = test.bag_ids

    classifier_name = parameters.pop('classifier')
    if classifier_name in CLASSIFIERS:
        classifier = CLASSIFIERS[classifier_name](**parameters)
    else:
        print 'Technique "%s" not supported' % classifier_name
        callback.quit = True
        return

    print 'Training...'
    timer.start('training')
    if train.regression:
        classifier.fit(train.bags, train.bag_labels)
    else:
        classifier.fit(train.bags, train.pm1_bag_labels)
    timer.stop('training')

    print 'Computing test bag predictions...'
    timer.start('test_bag_predict')
    bag_predictions = classifier.predict(test.bags)
    timer.stop('test_bag_predict')

    if INSTANCE_PREDICTIONS:
        print 'Computing test instance predictions...'
        timer.start('test_instance_predict')
        instance_predictions = classifier.predict(test.instances_as_bags)
        timer.stop('test_instance_predict')

    print 'Computing train bag predictions...'
    timer.start('train_bag_predict')
    train_bag_labels = classifier.predict() # Saves results from training set
    timer.stop('train_bag_predict')

    if INSTANCE_PREDICTIONS:
        print 'Computing train instance predictions...'
        timer.start('train_instance_predict')
        train_instance_labels = classifier.predict(train.instances_as_bags)
        timer.stop('train_instance_predict')

    print 'Constructing submission...'
    # Add statistics
    for attribute in ('linear_obj', 'quadratic_obj'):
        if hasattr(classifier, attribute):
            submission['statistics'][attribute] = getattr(classifier,
                                                          attribute)
    submission['statistics'].update(timer.get_all('_time'))

    for i, y in zip(test.bag_ids, bag_predictions.flat):
        submission['bag_predictions']['test'][i] = float(y)
    for i, y in zip(train.bag_ids, train_bag_labels.flat):
        submission['bag_predictions']['train'][i] = float(y)
    if INSTANCE_PREDICTIONS:
        for i, y in zip(test.instance_ids, instance_predictions.flat):
            submission['instance_predictions']['test'][i] = float(y)
        for i, y in zip(train.instance_ids, train_instance_labels.flat):
            submission['instance_predictions']['train'][i] = float(y)

    # For backwards compatibility with older versions of scikit-learn
    if train.regression:
        from sklearn.metrics import r2_score as score
        scorename = 'R^2'
    else:
        try:
            from sklearn.metrics import roc_auc_score as score
        except:
            from sklearn.metrics import auc_score as score
        scorename = 'AUC'

    try:
        if train.bag_labels.size > 1:
            print ('Training Bag %s Score: %f'
                   % (scorename, score(train.bag_labels, train_bag_labels)))
        if INSTANCE_PREDICTIONS and train.instance_labels.size > 1:
            print ('Training Inst. %s Score: %f'
                   % (scorename, score(train.instance_labels, train_instance_labels)))
        if test.bag_labels.size > 1:
            print ('Test Bag %s Score: %f'
                   % (scorename, score(test.bag_labels, bag_predictions)))
        if INSTANCE_PREDICTIONS and test.instance_labels.size > 1:
            print ('Test Inst. %s Score: %f'
                   % (scorename, score(test.instance_labels, instance_predictions)))
    except Exception as e:
        print "Couldn't compute scores."
        print e

    print 'Finished task %s.' % str(experiment_id)
    return submission
