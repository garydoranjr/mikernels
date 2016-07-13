"""
Wraps the C++ boxkernel code
"""
import os
import time
import subprocess

BINARY = os.path.join('kernel', 'boxandmc')
DSET_DIR = 'converted_datasets'
DB_SUFFIX = 'db'
SPEC_SUFFIX = 'spec'

def client_target(task, callback):
    (dataset, ktype, epsilon, delta, seed, i, j) = task['key']

    print 'Dataset: %s' % dataset
    print 'Kernel Type: %s' % ktype
    print 'epsilon: %f' % epsilon
    print 'delta: %f' % delta
    print 'seed: %d' % seed
    print 'i: %d' % i
    print 'j: %d' % j

    dbfile = os.path.join(DSET_DIR, '%s.%s' % (dataset, DB_SUFFIX))
    specfile = os.path.join(DSET_DIR, '%s.%s' % (dataset, SPEC_SUFFIX))

    print 'Computing...'
    start = time.time()
    process = subprocess.Popen([BINARY, ktype, str(delta), str(epsilon),
                                specfile, dbfile, str(i), str(j), str(seed)],
                               stdout=subprocess.PIPE)
    output = process.communicate()[0].strip()
    if process.returncode != 0:
        raise Exception('boxandmc returned error code %d' % process.returncode)
    stop = time.time()

    print 'Constructing submission...'
    try:
        if output == 'inf':
            print 'Warning: substituting max long double for "inf"'
            mantissa = 1.18973
            exponent = 4932
        else:
            if not 'e' in output:
                output = ('%e' % float(output))
            mantissa, exponent = map(float, output.split('e'))
    except ValueError:
        raise Exception('Trouble parsing output "%s"' % output)

    submission = dict()
    submission['mantissa'] = mantissa
    submission['exponent'] = exponent
    submission['time'] = (stop - start)
    print '%fe%d' % (mantissa, exponent)

    print 'Finished task in %f seconds.' % submission['time']
    return submission
