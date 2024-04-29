# PMOIRED example #3: AX Cir and companion search

binary search *à la* [CANDID](https://github.com/amerand/CANDID). Results are currently slightly different from CANDID, as bandwidth smearing in PMOIRED is still not computed properly, and in the case of AX Cir, the separation is quite large compared to to the bandwidth semaring radius (R$\lambda$/B).  

In this example:
- model the observations with a simple gray binary 
- using `gridFit` and `showGrid` to find the global minimum for the sepration vector of the binary
- bootstrapping to better estimate the object's parameters
- usinf `detectionLimit` to estimate detection limit on a third companion

*https://github.com/amerand/PMOIRED - Antoine Mérand (amerand@eso.org)*


```python
#-- uncomment to get interactive plots
#%matplotlib widget
import numpy as np
import pmoired
```

# Load Data


```python
oi = pmoired.OI('../DATA/AXCir/AXCir.oifits')
```

# Grid Search
To do a grid search, we need to define:
- a basic model, here composed of a primary star `*` and a a companion `c`. The 
- an exploration dictionnary `expl`: here we want to do explore `c,x` and `c,y` in a grid, each between -40 and +40mas (R$_\mathrm{spectro}$ <$\lambda$>/B$_{max}$/2), with a step of 4mas (<$\lambda$>/B$_{max}$).    
- as usual, which observable we will fit: `V2` and `T3PHI`
- additionnaly, we can define priors and constraints. Priors are apply during the fit, whereas constraints are applied to the grid of inital parameters

It is important to let `gridFit()` which parameters to fit. As usual, one needs to fix one of the fluxes, here the flux of the primary star. We also assume that the companion is unresolved, so its angular diameter is fixed to 0. Searching beyond R$_\mathrm{spectro}$ <$\lambda$>/B$_{max}$/2, the effects of bandwidth smearing start to be quite important and will reduce the interferometric signal of the binary. Since `PMOIRED` does not compute (yet) correctly the bandwidth smearing, the contrast ratio will be incorrect for large separation, which is the case for AX Cir.   


```python
# -- smallest lambda/B in mas (first data set) 
step = 180*3600*1000e-6/np.pi/max([np.max(oi.data[0]['OI_VIS2'][k]['B/wl']) for k in oi.data[0]['OI_VIS2']])

# -- spectral resolution (first data set) 
R = np.mean(oi.data[0]['WL']/oi.data[0]['dWL'])

print('step: %.1fmas, range: +- %.1fmas'%(step, R/2*step))

# -- initial model dict: 'c,x' and 'c,y' do not matter, as they will be explored in the fit
param = {'*,ud':0.8, '*,f':1, 'c,f':0.01, 'c,x':0, 'c,y':0, 'c,ud':0.0}

# -- define the exploration pattern
expl = {'grid':{'c,x':(-R/2*step, R/2*step, step), 'c,y':(-R/2*step, R/2*step, step)}}

# -- setup the fit, as usual
oi.setupFit({'obs':['V2', 'T3PHI']})

# -- actual grid fit
oi.gridFit(expl, model=param, doNotFit=['*,f', 'c,ud'], prior=[('c,f', '<', 1)], 
           constrain=[('np.sqrt(c,x**2+c,y**2)', '<=', R*step/2),
                      ('np.sqrt(c,x**2+c,y**2)', '>', step/2) ])
```

# Inspect grid search

`showGrid()` shows a 2D map of the $\chi^2$ of the minima. The black crosses are the starting points of the fits, and the coloured dots are the location / corresponding $\chi^2$ of the local minima.


```python
# -- show the 2D grid of reduced chi2
oi.showGrid()
```

# Show best fit model and fit to the data


```python
# -- show data with best fit 
oi.show()
```

# Parameters estimation using bootstrapping
One can refine the estimation of the companion's parameters, by using bootstrapping. This usually results in larger uncertainties, because bootstrapping mitigate the effects of correlated data. In the case of this example, because we have only one sequence of observations, data resampling does not help much. However, it shows that the data set is consistent: there are no part of the dataset improving / degrading the companion detection.


```python
oi.bootstrapFit(300)
oi.showBootstrap()
```

# Look for a third component: detection limit

Assuming that the best model is the one we found before, we add a third unresolved component. Using `detectionLimit` in a way very similar to `gridSearch`. We define an exploration pattern randomising on the position of the third components, and estimating the flux leading to a 3$\sigma$ detection. This method was described in [Absil et al (2011)](https://ui.adsabs.harvard.edu/abs/2011A%26A...535A..68A/abstract) and implemented in [CANDID](https://github.com/amerand/CANDID). Using `showLimGrid`, we see the detection level as function of position of third component (left), as well as the histogram of the its 3$\sigma$ magnitude. Note that we need to set `mag=1` option to get the display in magnitude, rather than straight fluxes. Note also that interpretating the result need to take into account the flux of the primary, which is 1 in our case.  

CANDID finds a $\sim$5.6mag detection limit for 99% of the position. Here, the median third star has a 3$\sigma$ detection limit of 6.0mag, and 95% are between 5.7 and 6.3mag, so the agreement is excellent. 


```python
# -- smallest lambda/B in mas (first data set) 
step = 180*3600*1000e-6/np.pi/max([np.max(oi.data[0]['OI_VIS2'][k]['B/wl']) for k in oi.data[0]['OI_VIS2']])

# -- spectral resolution (first data set) 
R = np.mean(oi.data[0]['WL']/oi.data[0]['dWL'])

print('step: %.1fmas, range: +- %.1fmas'%(step, R/2*step))

# -- best model from above
best = {'*,ud':0.8278, # +/- 0.0078
        'c,f': 0.00853, # +/- 0.00039
        'c,x': 6.226, # +/- 0.059
        'c,y': -28.503, # +/- 0.078
        '*,f': 1,
        'c,ud':0.0,
        '3,ud':0, '3,x':-15, '3,y':5, '3,f':0.01
       }

# -- grid exploration
#expl = {'grid':{'3,x':(-20, 20, 1), '3,y':(-20, 20, 1)}}
#oi.detectionLimit(expl, '3,f', model=best)

# -- random exploration
expl = {'rand':{'3,x':(-R*step/2, R*step/2), '3,y':(-R*step/2, R*step/2)}}

oi.detectionLimit(expl, '3,f', model=best, Nfits=500, nsigma=3, 
                 constrain=[('np.sqrt(3,x**2+3,y**2)', '<=', R*step/2 ),
                            ('np.sqrt(3,x**2+3,y**2)', '>', step/2) ])
 
oi.showLimGrid(mag=1)
```
