#!/usr/bin/env python
import os
import numpy as np
from collections import defaultdict

OUTPUTDIR = 'scaled_datasets'

def main(datafile, factor):
    factor = float(factor)

    with open(datafile, 'r') as ifile:
        with open(os.path.join(OUTPUTDIR, os.path.basename(datafile)), 'w+') as ofile:
            for line in ifile:
                parts = line.split(',')
                for i in range(2, len(parts) - 1):
                    parts[i] = str(int(float(parts[i])*factor))
                ofile.write(','.join(parts))

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog datafile factor")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)

