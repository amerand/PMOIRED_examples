# `PMOIRED` Example 6: global fit of orbital motion, including radial velocities

In this example, we analyse the data from binary HD210763 [Gallenne et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...672A.119G/abstract). The goal here is to fit the orbit simultaneously to all data (i.e. all epochs) with one model. For this, we introduce how to use the orbital parameters as model's parameters for `PMOIRED`. We also show how to fit additional data simultaneously to interfereomtric ones, with the help of support functions in [hd210763vrad.py](hd210763vrad.py), to also fit the radial velocities from the UVES and SOPHIE spectrograph, tabulated in the same paper (table C.1).

Overall, we find very similar results, with a few ceveats:
- we do not take into account the phase error in the field of view which affect the apparent semi-major axis $a$
- we do not account for systematics, for instance due to spectral calibration, which affect the apparent semi-major axis $a$



```python
# -- uncomment to get interactive plots
#%matplotlib widget
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy import units as U

import pmoired

# where radial velocity are defined, and function to return residuals to the fitimport hd210763vrad
import hd210763vrad
```

## load GRAVITY data, binned


```python
oi = pmoired.OI('/Users/amerand/DATA/Science/HD210763/*fits', insname='GRAVITY_SC', binning=100, verbose=0)

allMJD = []
for d in oi.data:
    allMJD.extend(list(d['MJD']))
print(f"{sorted(allMJD)=}")
print(f"{np.median(allMJD)=}")
```


```python
oi.show(obs=['T3PHI', '|V|'])
```

## global fit of interfereomtric data only

The orbital motion is defined by setting the position `2,x` and `2,y` to special keyword `orbit`. By doing so, `PMOIRED` will look to the orbital parmaeters in the form `2,orb ___`, where `___` are:
- `P`: the orbital period in days
- `MJD0`: the modified Julian date of the peri passage
- `e`: the eccentricity
- `incl`: the inclination in degrees
- `omega`: the argument of the periapsis ($\omega$) in degrees
- `OMEGA`: the longitude of the ascending node ($\Omega$) in degrees
- `a`: apparent semi-major axis in mas
  
Some work is required on the parametrisation:
- `MJD0` need to be chosen within the range of observed dates, otherwise it create large correlation between its values and other parameters such as `omega` or `P`
- values with have very small uncertainties compared to their value do not play well with the minimiser. the is the case for `MJD0` or `P` for instance, For this reason, we only fit a small offset to a fixed values
- within the uncertainties, the orbit is circular (`e`$\approx$0), hence the $\sim$100% correlation between `MJD0` and `omega` and their large incertainties. We fix `omega` and `e` (after setting the later to 0).

Comparing our result to [Gallenne et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...672A.119G/abstract) table 6 we find an overall good agreement, *except for the definition of $\omega$ and $\Omega$, which are 180º apart!*. The parameters in `PMOIRED` are consistent with [`orbitize!`](https://orbitize.readthedocs.io/en/latest/)


```python
# -- original parameters from Gallenne et al. 2023, table 6, not the -180 in omega and OMEGA
m0 = {'2,ud': 0.1,
     '1,ud': 0.2,
     '2,f': 0.371,
     '2,x': 'orbit',
     '2,y': 'orbit',
     '2,orb P': '$P-42.38 + 42.38',
     'P-42.38': 0.0011,
     '2,orb MJD0': '$MJD0-59800 + 59800',
     'MJD0-59800': 54275.93+42.3811*130-59800,
     '2,orb e': 0.6228,
     '2,orb omega': 293.965-180,
     '2,orb OMEGA': 257.6-180,
     '2,orb incl': 71.0,
     '2,orb gamma': 14.94,
     'additional residuals':hd210763vrad.resi
    }

if False:
    # -- parametrisation using physical quantities -> more correlations
    m0.update({'2,orb M'  : 1.7377+1.4871,
               '2,orb q'  : 1.4871/1.7377,
               '2,orb plx': 10.692,})
else:
    # -- parametrisation using observables -> less correlations
    m0.update({'2,orb Ka': 50.28,
               '2,orb Kb': 58.76,
               '2,orb a' : 3.76,})

def compareParams(bestfit, title='PMOIRED'):
    _best = pmoired.oimodels.computeLambdaParams(bestfit['best'])
    _uncer =  bestfit['uncer'].copy()
    _uncer['2,orb MJD0'] = _uncer['MJD0-59800']
    _uncer['2,orb P'] = _uncer['P-42.38']
    print('parameter       Gallenne+23        '+title)
    for k in m0.keys():
        if not type(m0[k])==str and not type(m0[k])==type(hd210763vrad.resi):
            dif = _best[k]-m0[k]
            if k=='2,orb MJD0':
                # MJD0 is compared modulo the period
                dif = dif%(_best['2,orb P'])
            if _uncer[k]>0:
                dif /= _uncer[k]
                unit = 'sigma'
            else:
                dif = None
            if not dif is None:
                print('%-12s'%k, '%13.6f -> %13.6f ± %11.6f (%5.1f%s)'%(m0[k], _best[k], _uncer[k], dif, unit))
```


```python
m = m0.copy()
# -- remove radial velocities
m.pop('additional residuals')
# -- these can only be fitted if we have radial velocities
doNotFit=['2,orb Ka', '2,orb Kb', '2,orb plx', '2,orb gamma', '2,orb M', '2,orb q', '2,orb plc']
# -- too small
doNotFit.append('2,ud')

# -- set up context of the fit
oi.setupFit({'obs':['T3PHI', '|V|'], 
            'min error':{'T3PHI':1},
            'min relative error':{'|V|':0.01},
            })

oi.doFit(m, doNotFit=doNotFit)
oi.show(showUV=False)

# -- compare with original parameters
compareParams(oi.bestfit,  'PMOIRED interf only (fit)')
```


```python
oi.bootstrapFit(100)
oi.showBootstrap()
# -- compare with original parameters
compareParams(oi.boot, 'PMOIRED interf only (bootstrapped)')
```

## Taking into account radial velocities

The orbital calculator can be also used to compute radial velocities, as exploited in [oLeo_vrad.py](oLeo_vrad.py). In that case, we need to parametrise the semi-major axis using the parallax. 

Because of Kepler 2rd law, masses, paralaxes and apparent semi-major axis are redundant. We can parametrise using:
- `plx`, `M` and `q`==Msecondary/Mprimary: the parallax in mas, total mass in solar masse and mass ratio.
- `plx`, `Ma` and `Mb`: the parallax in mas and masses of the primary and secondary, in solar masses
- `plx`, `a` and `q`: parallax an apparent semi-major axis in mas, mass ratio.

We choose the last parametrisation because `a` is really what we measure in interferometry and, as explained in Gallenne et al. (2023), section 2.3, there is a systematic uncertainty on the scaling of intereferometric separation arising from the uncertainty on the spectral calibration. IN the case of GRAVITY in high resolution, this is 0.02%.


```python
m = m0.copy()
# -- too small
doNotFit=['2,ud']

# -- set up context of the fit
oi.setupFit({'obs':['T3PHI', '|V|'], })

oi.doFit(m, doNotFit=doNotFit)
oi.show(showUV=False)

# -- compare with original parameters
compareParams(oi.bestfit,  'PMOIRED interf only (fit)')
```

## bootstrapping

Radial velocities require also randomisation. `PMOIRED` method `bootstrapFit` accepts an optional function `additionalRandomise` such that `additionalRandomise(True)` will randomise the data, and `additionalRandomise(False)` will reverse the data to their original order and weights. see function `randomise` in [hd210763vrad.py](./hd210763vrad.py).


```python
oi.bootstrapFit(300, additionalRandomise=hd210763vrad.randomise)
oi.showBootstrap()
compareParams(oi.boot, 'PMOIRED interf+vrad (bootstrapped)')
```

## grid search orbit

All the previous work was made by fitting the data starting from the known solution. In case the orbital solution is not known, one needs to explore the parameters' space. `gridFit` allows to start many fits with randomise parameters. In the case below, to limit the search, we assume the orbit is circular and we know the period (e.g. by looking at the radial velocity curves). The search is now guaranteed to find the global minimum, but most of the time it will. You might also find several solutions with $\chi^2$ close to 2, but they will have their `MJD0` a whole number of periods apart. Not knowing the period and using brute force is not recommended: you better use a periodogram on the Vrad before hand.



```python
# -- original parameters from Gallenne et al. 2023, table 6, not the -180 in omega and OMEGA
m = {'2,ud': 0.1,
     '1,ud': 0.2,
     '2,f': 0.371,
     '2,x': 'orbit',
     '2,y': 'orbit',
     '2,orb P': '$P-42.38 + 42.38',
     'P-42.38': 0.0011,
     '2,orb MJD0': '$MJD0-59800 + 59800',
     'MJD0-59800': 54275.93+42.3811*130-59800,
     '2,orb e': 0.6228,
     '2,orb omega': 293.965-180,
     '2,orb OMEGA': 257.6-180,
     '2,orb incl': 71.0,
     '2,orb gamma': 14.94,
     '2,orb Ka': 50.28,
     '2,orb Kb': 58.76,
     '2,orb a' : 3.76,
     'additional residuals':hd210763vrad.resi
    }

# -- exploration pattern
expl = {'rand':{'2,orb incl':(30, 150), # >90 to reverse rotation direction
                '2,orb OMEGA':(0, 180), 
                '2,orb omega':(-180, 180), 
                'MJD0-59800': (-20, 20), 
                'P-42.38': (-1, 1),
                 '2,orb Ka':(45, 60), # from max separation, plx and period
                 '2,orb Kb':(45, 60), # from max separation, plx and period                       
               }}

# -- prior:
prior = [('2,f', '<', 1), # secondary is dimmer
         ('2,orb e', '>=', 0),
         ('2,orb e', '<', 1),
         ('2,orb incl', '>=', 0),
         ('2,orb incl', '<=', 180),
         ('2,orb Ka', '<=', '2,orb Kb'),
        ]

oi.setupFit({'obs':['T3PHI', '|V|']})
doNotFit = ['2,ud']
#oi.doFit(m, doNotFit=doNotFit, prior=prior)

oi.gridFit(expl, Nfits=100, model=m, doNotFit=doNotFit, prior=prior)
```


```python
# -- show best orbits from the grid
chi2 = sorted([g['chi2'] for g in oi.grid])
chi2min = chi2[0]
print("first 5 solutions' chi2:", np.round(chi2[:5], 1))
deltaChi2 = 1 # shows solution between min(chi2) and min(chi2)+deltaChi2
for g in oi.grid:
    if g['chi2']<chi2min+deltaChi2:
        m = pmoired.oimodels.computeLambdaParams(g['best'])
        orb = {k.split('orb ')[1]:round(m[k], 4) for k in m if k.startswith('2,orb')} 
        print('chi2=%.4f'%g['chi2'], '\n > orbit:', orb)

```


```python

```
