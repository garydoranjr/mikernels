#!/usr/bin/env python
from hardmargin import compute_statistics

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 1:
        parser.print_help()
        exit()

    statsfile = args[0]
    with open(statsfile, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                compute_statistics(*parts)
            except Exception as e:
                print e
