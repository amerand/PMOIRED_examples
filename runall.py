#!/usr/bin/env python3

import os
from multiprocessing import Pool

files = ['EX1 angular diameter alphaCenA.ipynb',
         'EX2 chromatic YSO disk.ipynb',
         'EX3 companion search AXCir.ipynb',
         'EX4 Be model comparison with AMHRA.ipynb',
         'EX5 Binary with spectroscopic lines.ipynb'
        ]

def processOne(f):
    print('Processing "'+f+'"')
    os.system('jupyter-nbconvert --inplace --execute "'+f+'"')
    os.system('jupyter-nbconvert --CoalesceStreamsPreprocessor.enabled=True --to html "'+f+'"')

if __name__=='__main__':  
    with Pool() as p:
        p.map(processOne, files)