#!/usr/bin/env python
import os

from kernelio import KernelManager
from progress import ProgressMonitor

SRC = 'BRCR'
DST = ('CBCH', 'DEJU', 'HAFL', 'HETH', 'HEWA', 'OSFL',
       'PSFL', 'RBNU', 'SWTH', 'VATH', 'WETA', 'WIWR',)

TYPES = ('and', 'or', 'andor')

DIR = 'precomputed'
FMT = '%s_%s.db'

def filename(dset, t):
    return os.path.join(DIR, FMT % (dset, t))

def copy_kernel(srck, dstk, dst, t):
    dst_did = dstk.get_dataset_id(dst)
    dst_tid = dstk.get_ktype_id(t)
    srccon = srck.get_connection()
    cursor = srccon.cursor()
    cursor.execute('SELECT * FROM kernel')
    new_entries = [
        (dst_did, dst_tid, epsilon, delta, seed, i, j,
         mantissa, exponent, time)
        for _, _, epsilon, delta, seed, i, j, mantissa, exponent, time
        in cursor.fetchall()
    ]

    dstcon = dstk.get_connection()
    with dstcon:
        dstcon.executemany(
            'INSERT INTO kernel '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', new_entries
        )

if __name__ == '__main__':

    prog = ProgressMonitor(total=len(DST)*len(TYPES), msg='Copying kernels')
    for dst in DST:
        for t in TYPES:
            srcfile = filename(SRC, t)
            dstfile = filename(dst, t)
            srck = KernelManager(srcfile)
            dstk = KernelManager(dstfile)
            copy_kernel(srck, dstk, dst, t)
            prog.increment()
