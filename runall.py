#!/usr/bin/env python3

import os, time
from multiprocessing import Pool

directory = 'notebooks'
files = ['EX1 angular diameter alphaCenA.ipynb',
         'EX2 chromatic YSO disk.ipynb',
         'EX3 companion search AXCir.ipynb',
         'EX4 Be model comparison with AMHRA.ipynb',
         'EX5 Binary with spectroscopic lines.ipynb',
         'EX6 global orbital fit.ipynb'
        ]
files = [os.path.join(directory, f) for f in files]


def processOne(f):
    #print('Processing "'+f+'"')
    os.system('jupyter-nbconvert --inplace --execute "'+f+'"')
    os.system('jupyter-nbconvert --CoalesceStreamsPreprocessor.enabled=True --to html "'+f+'"')
    os.system('jupyter-nbconvert --clear-output --inplace "'+f+'"')

def conv2md(f):
    os.system('jupyter-nbconvert --to markdown "'+f+'"')

if __name__=='__main__':
    t0 = time.time()
    # -- process, generate HTML and clean the outputs
    with Pool() as p:
       p.map(processOne, files)
    # -- convert notebooks
    with Pool() as p:
        p.map(conv2md, files)
    print()
    for f in files:
        if os.path.exists(f.replace('.ipynb', '.html')):
            os.rename(f.replace('.ipynb', '.html'),
                      os.path.join('html', os.path.basename(f).replace('.ipynb', '.html')))

        if os.path.exists(f.replace('.ipynb', '.md')):
            os.rename(f.replace('.ipynb', '.md'),
                      os.path.join('markdown', os.path.basename(f).replace('.ipynb', '.md')))

    print('it took %.1f minutes'%((time.time()-t0)/60))
