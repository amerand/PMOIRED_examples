# `PMOIRED` Example 6: global fit of orbital motion, including radial velocities

In this example, we re-analyse the $\omicron$ Leo data from example 5. The goal here is to fit the orbit simultaneously to all data (i.e. all epochs) with one model. For this, we introduce how to use the orbital parameters as model's parameters for `PMOIRED`. We also show how to fit additional data simultaneously to interfereomtric ones, with the help of support functions (in [oLeo_vrad.py](oLeo_vrad.py)). 

We will use the GRAVITY data showed in [Gallenne et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...672A.119G/abstract), as well as the radial velocities from the UVES and SOPHIE spectrograph, tabulated in the same paper (table C.1).

Overall, we find very similar results, with a few ceveats:
- we do not take into account the phase error in the field of view which affect the apparent semi-major axis $a$
- we do not account for systematics, for instance due to spectral calibration, which affect the apparent semi-major axis $a$
- the orbit is quasi circular. The means parameters such as $\omega$ or MJD0 are difficult to compare. We chose below to assume the orbit is circular (but this is easily reverted)


```python
# -- uncomment to get interactive plots
#%matplotlib widget
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy import units as U

import sys
sys.path = ['../../PMOIRED/'] + sys.path

import pmoired

import oLeo_vrad # where radial velocity are defined, and function to return residuals to the fit
```

## load GRAVITY data, binned


```python
oleo = pmoired.OI('../DATA/o_Leo/G*fits', insname='GRAVITY_SC', binning=200, verbose=0)
allMJD = []
for d in oleo.data:
    allMJD.extend(list(d['MJD']))
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


Comparing our result to [Gallenne et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...672A.119G/abstract) table 6 we find an overall good agreement. The orbit plot is just indicative: the blue points are the predictions from the model, not the fitted position at each epoch. In this model, there is not fit at each epoch. 


```python
# -- original parameters from Gallenne et al. 2023, table 6
plx = 24.412 # parallax in mas
m0 = {'1,ud':1.281, 
     '2,ud':(2*2.43*U.Rsun).to(U.au).value*plx,
     '2,x':'orbit',
     '2,y':'orbit',
     '2,f':10**(-(1.1+0.39)/2.5),
     '2,orb e':0.00007, 
     '2,orb omega': 214,
     '2,orb OMEGA': 191.6,
     '2,orb incl': 57.8,
     '2,orb MJD0': 2450623.9-2400000.5,
     '2,orb P': 14.498068,
     '2,orb a': 4.477,
     }
def compareParams(bestfit, title='PMOIRED'):
    _best = pmoired.oimodels.computeLambdaParams(bestfit['best'])
    _uncer =  bestfit['uncer'].copy()
    _uncer['2,orb MJD0'] = _uncer['MJD0-59690']
    _uncer['2,orb P'] = _uncer['P-14.2']
    print('parameter       Gallenne+23        '+title)
    for k in m0.keys():
        if not type(m0[k])==str and not type(m0[k])==type(oLeo_vrad.resiVrad):
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

# -- force the orbit to be circular
forceCircular = True
if forceCircular:
    m['2,orb e'] = 0
    m['2,orb MJD0'] -= m['2,orb omega']/360*m['2,orb P']
    m['2,orb omega'] = 0
    doNotFit = ['2,orb e', '2,orb omega']
else:
    doNotFit = []

# -- recenter the MJD0 around interferometric data
m['2,orb MJD0'] -= m['2,orb P']*round((m['2,orb MJD0'] - round(np.mean(allMJD), 0))/m['2,orb P'], 0)

# -- only fit decimal part of the MJD0 to help fit converge
m['MJD0-59690'] = m['2,orb MJD0']-59690
m['2,orb MJD0'] = '$MJD0-59690 + 59690'

# -- only fit decimal part of the Period to help fit converge
m['P-14.2'] = m['2,orb P']-14.2
m['2,orb P'] = '$P-14.2 + 14.2'
#display(m)

# -- set up context of the fit
oleo.setupFit({'obs':['T3PHI', '|V|'], })

oleo.doFit(m, doNotFit=doNotFit)
oleo.show(showUV=False)

# -- compare with original parameters
compareParams(oleo.bestfit,  'PMOIRED interf only (fit)')

oLeo_vrad.showOrbit(oleo.bestfit['best'], allMJD)
```


```python
oleo.bootstrapFit(100)
oleo.showBootstrap()
# -- compare with original parameters
compareParams(oleo.boot, 'PMOIRED interf only (bootstrapped)')
```

## Taking into account radial velocities

The orbital calculator can be also used to compute radial velocities, as exploited in [oLeo_vrad.py](oLeo_vrad.py). In that case, we need to parametrise the semi-major axis using the parallax. 

Because of Kepler 2rd law, masses, paralaxes and apparent semi-major axis are redundant. We can parametrise using:
- `plx`, `M` and `q`==Msecondary/Mprimary: the parallax in mas, total mass in solar masse and mass ratio.
- `plx`, `Ma` and `Mb`: the parallax in mas and masses of the primary and secondary, in solar masses
- `plx`, `a` and `q`: parallax an apparent semi-major axis in mas, mass ratio.

We choose the last parametrisation because `a` is really what we measure in interferometry and, as explained in Gallenne et al. (2023), section 2.3, there is a systematic uncertainty on the scaling of intereferometric separation arising from the uncertainty on the spectral calibration. IN the case of GRAVITY in high resolution, this is 0.02%.


```python
# -- original parameters from Gallenne et all 2023, table 6
plx = 24.412 # parallax in mas
m0 = {'1,ud':1.281, 
     '2,ud':(2*2.43*U.Rsun).to(U.au).value*plx,
     '2,x':'orbit',
     '2,y':'orbit',
     '2,f':10**(-(1.1+0.39)/2.5),
     '2,orb e':0.00007, 
     '2,orb omega': 214,
     '2,orb OMEGA': 191.6,
     '2,orb incl': 57.8,
     '2,orb MJD0': 50623.4,
     '2,orb P': 14.498068,
     '2,orb a': 4.477,
     # -- radial velocities
     '2,orb plx': plx,
     '2,orb q':  1.841/2.074,
     '2,orb gamma': 26.24,
     'additional residuals': oLeo_vrad.resiVrad,
     }

m = m0.copy()

forceCircular = True
if forceCircular:
    m['2,orb e'] = 0
    m['2,orb MJD0'] -= m['2,orb omega']/360*m0['2,orb P']
    m['2,orb omega'] = 0
    doNotFit = ['2,orb e', '2,orb omega']
else:
    doNotFit = []

# -- recenter the MJD0 around interferometric data
m['2,orb MJD0'] -= m['2,orb P']*round((m['2,orb MJD0']-round(np.mean(allMJD), 0))/m['2,orb P'], 0)

# -- only fit decimal part of the MJD0 to help fit converge
m['MJD0-59690'] = m['2,orb MJD0']-59690
m['2,orb MJD0'] = '$MJD0-59690 + 59690'

# -- only fit decimal part of the Period to help fit converge
m['P-14.2'] = m['2,orb P']-14.2
m['2,orb P'] = '$P-14.2 + 14.2'


# -- set observables and minimum errors: it will affect the final result as it
# -- changes the relative weight to raidal velocities
oleo.setupFit({'obs':['T3PHI', '|V|'], 
               'min error':{'T3PHI':1.0},
               'min relative error':{'|V|':0.01},
             })

oleo.doFit(m, doNotFit=doNotFit)
oleo.show(showUV=False)

# -- show radial velocity data
oLeo_vrad.resiVrad(oleo.bestfit['best'], fig=oleo.fig)

# -- compare with original parameters
compareParams(oleo.bestfit, 'PMOIRED interf+vrad (fit)')
```

## bootstrapping

Radial velocities require also randomisation. `PMOIRED` method `bootstrapFit` accepts an optional function `additionalRandomise` such that `additionalRandomise(True)` will randomise the data, and `additionalRandomise(False)` will reverse the data to their original order and weights. see function `randomise` in [oLeo_vrad.py](./oLeo_vrad.py).


```python
oleo.bootstrapFit(100, additionalRandomise=oLeo_vrad.randomise)
oleo.showBootstrap()
compareParams(oleo.boot, 'PMOIRED interf+vrad (bootstrapped)')
```

## grid search orbit

All the previous work was made by fitting the data starting from the known solution. In case the orbital solution is not known, one needs to explore the parameters' space. `gridFit` allows to start many fits with randomise parameters. In the case below, to limit the search, we assume the orbit is circular and we know the period (e.g. by looking at the radial velocity curves). The search is now guaranteed to find the global minimum, but most of the time it will. You might also find several solutions with $\chi^2$ close to 2, but they will have their `MJD0` a whole number of periods apart. If the orbit is not restricted to be circular, then a lot more minima are found: the all have very similar parameters except for the degeneracy `MJD0`/`omega`.


```python

plx = 24.412 # parallax in mas
m0 = {'1,ud': 1.26, 
     '2,ud': 0.72,
     '2,x':'orbit',
     '2,y':'orbit',
     '2,f': 0.255,
     '2,orb e':0.00, 
     '2,orb omega': 214,
     '2,orb OMEGA': 191.6,
     '2,orb incl': 57.8,
     '2,orb MJD0': 2450623.9-2400000.5,
     '2,orb P': 14.498068,
     '2,orb a': 4.477,
     # -- radial velocities
     '2,orb plx': plx,
     '2,orb q':  1.841/2.074,
     '2,orb gamma': 26.24,
     'additional residuals': oLeo_vrad.resiVrad,
     }

m = m0.copy()

# -- recenter the MJD0 around interferometric data
m['2,orb MJD0'] -= m['2,orb P']*round((m['2,orb MJD0']-round(np.mean(allMJD), 0))/m['2,orb P'], 0)

# -- only fit decimal part of the MJD0 to help fit converge
m['MJD0-59690'] = m['2,orb MJD0']-59690
m['2,orb MJD0'] = '$MJD0-59690 + 59690'

# -- only fit decimal part of the Period to help fit converge
m['P-14.2'] = m['2,orb P']-14.2
m['2,orb P'] = '$P-14.2 + 14.2'

# -- exploration pattern
expl = {'rand':{'2,orb incl':(30, 150), # >90 to reverse rotation direction
                '2,orb OMEGA':(0, 180), 
                'MJD0-59690': (-7,+7), 
                #'P-14.2':(-1, 1), # not knowing the period makes the search much more difficult...
                '2,orb a':(3, 5), # based on max observed separation
                '2,orb q':(0.8, 0.9), # based on similar semi-amplitude
               }}

forceCircular = True
if forceCircular:
    m['2,orb e'] = 0
    m['MJD0-59690'] -= m['2,orb omega']/360*m0['2,orb P']
    m['2,orb omega'] = 0
    doNotFit = ['2,orb e', '2,orb omega']
else:
    expl['rand']['2,orb e'] = (0.01, 0.1)
    expl['rand']['2,orb omega'] = (-180, 180)
    doNotFit = []

# -- prior:
prior = [('2,orb q', '<', 1), # secondary is lighter
         ('2,f', '<', 1), # secondary is dimmer
         ('2,orb e', '>=', 0),
         ('2,orb e', '<', 1),
         ('2,orb incl', '>=', 0),
         ('2,orb incl', '<=', 180),
        ]

oleo.setupFit({'obs':['T3PHI', '|V|']})

oleo.gridFit(expl, Nfits=100, model=m, doNotFit=doNotFit, prior=prior)
# oleo.save('oLeo_all_orbits.pmrd', overwrite=True)
```


```python
# oleo = pmoired.OI('oLeo_all_orbits.pmrd')
chi2 = sorted([g['chi2'] for g in oleo.grid])
chi2min = chi2[0]
print("first 5 solutions' chi2:", np.round(chi2[:5], 1))
deltaChi2 = 1 # shows solution between min(chi2) and min(chi2)+deltaChi2
for g in oleo.grid:
    if g['chi2']<chi2min+deltaChi2:
        m = pmoired.oimodels.computeLambdaParams(g['best'])
        orb = {k.split('orb ')[1]:m[k] for k in m if k.startswith('2,orb')} 
        print('chi2=', g['chi2'], '\n > orbit:', orb)

oleo.show()
# -- show radial velocity data
oLeo_vrad.resiVrad(oleo.grid[0]['best'], fig=oleo.fig)
```


```python

```
