# PMOIRED example #1: Alpha Cen A (VLTI/PIONIER)

In this example, we look a basic interferometric data with the goal of estimating an angular diameter. The data set is from [Kervella et al. A&A 597, 137 (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A.137K/abstract): observations of Alpha Cen A with the PIONIER beam combiner. The data have very little spectral resolution and will be treated as monochromatic. Data obtained from [JMMC OIDB](http://oidb.jmmc.fr/search.html?conesearch=alpha%20cen%20%2CJ2000%2C2%2Carcmin&instrument=PIONIER&order=%5Etarget_name), except date 2016-05-28, for which V2 is too high compared to other data.

### This example covers:
- Loading multiple oifits files
- Displaying all data in a same plot
- Least square fit:
    + uniform disk diameter
    + diameter with fixed center-to-limb darkening (Claret 4 parameters)
    + diameter and adjusted center-to-limb darkening (power law)
- Better uncertainties estimates with bootstrapping
- Access model's prediction to make custom plots

*https://github.com/amerand/PMOIRED - Antoine Mérand (amerand@eso.org)*


```python
# -- uncomment to get interactive plots
#%matplotlib widget
import pmoired
```

## Load Data
`pmoired.OI` loads a single file of a list of files. The result contains method to manipulate data, fit them and display results. 


```python
oi = pmoired.OI('../DATA/alphaCenA/*fits')
```

## Show data and u,v
This is done using the `show` method in the `oi` object: 
- `allInOne=True` plots everything on one plot. By default, every file will be shown on a separate figure
- `perSetup=True` groups files per instruments and (spectral) setups. 
- `spectro=True` show data spectrally. This behavior is automatic if `spectro` is not set: data with "a lot" spectral channels will show data as spectra per telescope / baseline / triangle  
- `fig` sets which figure is used (int)
- `logV=True` shows visibilities (V2 or |V|) in log scale.
- `logB=True` shows baselines (for V2 or |V|) in log scale.
- `obs` list of observables to plot (in `['V2', '|V|', 'T3PHI', 'DPHI', 'T3AMP', 'NFLUX']`). By default, all reckognized data are plotted. Once you start doing fits (as we'll see below), this behavior changes to plot fitted observables only. 
- `showFlagged=True` to show data flagged (i.e. rejected data)


```python
oi.show()
```

## Fit uniform disk model
In order to fit data, we need to set up with method `setupFit` using a dict containing the context of the fit. only `obs` is mandatory:
- `obs`: the list of observables to take into account, in `['V2', '|V|', 'T3PHI', 'DPHI', 'NFLUX']`. `T3PHI` stands for the closure phase. In addition, there are specific observables for spectrally dospersed data: `DPHI` differential phase and `NFLUX` the flux, normalised to the continuum.
- `min error`: a dict to set the minimum error (overrinding what's in the data file) for each observable. e.g. `d['fit']['min error'] = {'V2':0.04}`
- `min relative error`: a dict to set the minimum relative error (overrinding what's in the data file) for each observable. e.g. `d['fit']['min relative error'] = {'V2':0.04}`
- `max error`: a dict to ignore data with errors larger than a certain value. e.g. `d['fit']['max error'] = {'V2':0.1}`
- `wl ranges`: list of ranges ($\lambda_{min}[\mu m]$, $\lambda_{max}[\mu m]$) to restrict fit. e.g. `d['fit']['wl ranges'] = [(1.6, 1.9), (2.2, 2.3)]`. Especially useful for spectral data, to restric fit around a single line
- `baseline ranges`: list of ranges (B$_{min}[m]$, B$_{max}[m]$) to restrict fit. e.g. `d['fit']['baseline ranges'] = [(10, 50), (50, 100)]`. Note that is also applies to closure phases: all baselines in the triangles must satisfy the constrainss!

The fitting method, `oi.doFit` takes the first guess as input parameter: the parameters stored in a dictionnary define the model (models ). For example, a uniform disk of 8 milli-ardsecond in diameter is  `{'ud':8.0}`. The result is a dict (`oi.bestfit`) containing (among other things):
- `best`: the dictionnary of best fit parameters
- `uncer`: the dictionnary of uncertainties for the best fit parameters
- `chi2`: the reduced chi2
- `covd`: the covariance dictionnary

The `show` method now show the data and the best fit model.


```python
oi.setupFit({'obs':['V2'], 
             'min relative error':{'V2':0.01},
             #'baseline ranges': [(50, 100)],
            })
oi.doFit({'ud':8.5})
oi.show(logV=1)
```

## Fit a diameter with fixed Claret 4-parameters center-to-limb darkening
From Kervella et al. A&A 597, 137 (2017), table 3:
- paper: https://ui.adsabs.harvard.edu/abs/2017A%26A...597A.137K/abstract
- table 3: https://www.aanda.org/articles/aa/full_html/2017/01/aa29505-16/T3.html

In this example, we use an arbitrary profile for the center-to-limb darkening. The `profile` for the model uses special syntax with `$R` and `$MU` for the reduced radius (0..1) and its cosine. Note that the reduced $\chi^2$ ($\sim$5.5) is much lower than for uniform disk model ($\sim$18).

We add `imFov=10` to set the field-of-view (in mas) of a synthetic image of the model. The pixel size is set to a default, but can also be adjusted using `imPix=...` with a value in mas. In this case, one can see the limb darkened compared to the center of the star. Additional parameters for image are `imX, imY`: the center of the image in mas (default 0,0); `imPow`: to show $\mathrm{image}^\mathrm{imPow}$ (default 1.0, for compressed contrast use 0<`imPow`<1); `imWl0` gives a list of wavelength to display, in microns (for chromatic models). 

Note that the model's visibility is not computed from the image, so `im...` parameters do not affect the visibility computation!


```python
oi.setupFit({'obs':['V2'], 
             'min relative error':{'V2':0.01},
            })
# -- first guess with Claret 4 parameters model
oi.doFit({'diam':8., 'profile':'1 - 0.7127*(1-$MU**0.5) + 0.0452*(1-$MU**1) + 0.2643*(1-$MU**1.5) - 0.1311*(1-$MU**2)'})
oi.show(logV=1, imFov=10)
```

## Fit a power law center-to-limb darkening
The power law center-to-limb darkening has been proposed by [Hestroffer (1997)](https://ui.adsabs.harvard.edu/abs/1997A%26A...327..199H/abstract). Here, both diameter and power law index `alpha` are fitted. When more than one parameters are fitted, correlations between parameters will be shown, whith colors to indicate the strength of the correlation. In our example below, the correlation is very high. Note that evaluation of `profile` is pretty lazy and sollely bsed on string replacement and only reckognize other parameters if they start with the special character `$`. The parameter is not to be defined with `$` in the dictionnary.


```python
oi.setupFit({'obs':['V2'], 
             'min relative error':{'V2':0.01}
            })
param = {'diam':8.0, 'profile':'$MU**$alpha', 'alpha':0.5}
oi.doFit(param)
oi.show(logV=True)
```

## Bootstrapping for better estimate of uncertainties
The reduced $\chi^2$ of fits are large. This seems to indicate that errors in data are understimated, or that our model is inadequate. The V2 plot seem to indicate that our model is pretty good. The absolute value of the reduced $\chi^2$ is not used in the parameters' uncertainties estimation. Rather, `PMOIRED` use the convention that uncertainties are scaled to the data scatter, to match $\chi_{red}^2=1$. 

Another way to estimate uncertainties is to bootstrap on the data and do multiple fits to estimate the scatter of the fitted parameters. It is achieved by drawing data randomly to create new data sets. The final parameters and uncertainties are estimated as the average and standard devitation of all the fits which were performed.

The default number of bootstrapped fit is 2xnumber of data, where the atomic data is a spectral vector of `V2`, `|V|`, `T3PHI` etc. You can set this by hand using `Nfits=` keyword in `bootstrapFit`. To accelerate the computation, it is parallelized. `bootstrapFit` can take an additional parameter, `multi=`, to set the number of threads in case you do not wish to swamp your computer with too much activity.

The bootstrap fits are filtered using a recursive sigma clipping algorithm. You can analyse the results by using `showBootstrap` with option `sigmaClipping` (default is 4.5). `showBootstrap` shows the scatter plots and histograms, as well as as comparison with the fit to all data and its uncertainties. When more than one parameter is explored, covariance is computed from the 2D scatter plot. `showChi2=True` also shows the scatter plot for reduced $\chi^2$.


```python
oi.bootstrapFit()
```


```python
oi.showBootstrap(showChi2=True)
```

# Advanced features

Data are stored in the variable `data` which is a list of dictionnary (one dict per file). Raw data in the `oi` object can be accessed as `oi.data[6]['OI_VIS2']['G2D0']`: `6` is index of the data file, `OI_VIS2` is the extension and `G2D0` is the baseline. In practice there is little need to access data manually.


```python
print(oi.data[6]['filename'])
display(oi.data[6]['WL'])
display(oi.data[6]['OI_VIS2']['G2D0'])
```

## Access and plot model's predictions for custom plots

You can access the last computed model in the list `oi._model`. All data from the same instrument are merged in `oi._merged`, so even if you had loaded many files, `oi._model` will not have the same structure as `oi.data`. In `oi._merged` and `oi._model`, all the baselines are grouped under the keyword `all`.

Alternatively, one can compute a model using `pmoired.oimodels.VmodelOI`.
 
If you want to plot the V2 and T3PHI for a set of models, as well as the data, you can proceed as follow: 


```python
import matplotlib.pyplot as plt
plt.close(100)
plt.figure(100, figsize=(10,4))

axV2 = plt.subplot(121)
axT3 = plt.subplot(122)

models = {'UD':{'diam':8.2987},
          'power law': {'alpha':  0.1372, 'diam':   8.4923, 'profile':'$MU**$alpha'},
          'Claret-4': {'diam':   8.5098, 'profile':'1 - 0.7127*(1-$MU**0.5) + 0.0452*(1-$MU**1) + 0.2643*(1-$MU**1.5) - 0.1311*(1-$MU**2)',}
        }
# -- symbols and colors for each model 
colors= {'UD':'dr', 'power law':'vb', 'Claret-4':'sy'}

# -- compute observables mirroring pmoired structure
Vmodels = {}
for k in models:
    Vmodels[k] = pmoired.oimodels.VmodelOI(oi._merged, models[k])

_labelV2, _labelT3PHI = True, True

for i in range(len(oi._merged)): # -- for each instrument / setup

    for k in set(oi._merged[i]['OI_VIS2']['all']['NAME']): # V2 for each baseline
        # -- select baseline
        w = oi._merged[i]['OI_VIS2']['all']['NAME']==k
        # -- select valid data
        f = ~oi._merged[i]['OI_VIS2']['all']['FLAG'][w,:].flatten()
        # -- plot model
        for m in Vmodels:
            axV2.plot(Vmodels[m][i]['OI_VIS2']['all']['B/wl'][w,:].flatten()[f],
                      Vmodels[m][i]['OI_VIS2']['all']['V2'][w,:].flatten()[f], 
                      colors[m], alpha=0.2, label=m if _labelV2 else '')
        _labelV2 = False
        # -- plot data
        axV2.errorbar(oi._merged[i]['OI_VIS2']['all']['B/wl'][w,:].flatten()[f],
                     oi._merged[i]['OI_VIS2']['all']['V2'][w,:].flatten()[f],
                     yerr=oi._merged[i]['OI_VIS2']['all']['EV2'][w,:].flatten()[f],
                     linestyle='none', marker='.', capsize=2, color='k', alpha=0.2)
                     
    for k in set(oi._merged[i]['OI_T3']['all']['NAME']): # T3PHI for each triangle
        # -- select triangle
        w = oi._merged[i]['OI_T3']['all']['NAME']==k
        # -- select valid data
        f = ~oi._merged[i]['OI_T3']['all']['FLAG'][w,:].flatten()
        # -- plot models
        for m in Vmodels:
            axT3.plot(Vmodels[m][i]['OI_T3']['all']['Bmax/wl'][w,:].flatten()[f],
                      Vmodels[m][i]['OI_T3']['all']['T3PHI'][w,:].flatten()[f], 
                      colors[m], alpha=0.2, label=m if _labelT3PHI else '')
        _labelT3PHI = False

        # -- plot data
        axT3.errorbar(oi._merged[i]['OI_T3']['all']['Bmax/wl'][w,:].flatten()[f],
                     oi._merged[i]['OI_T3']['all']['T3PHI'][w,:].flatten()[f],
                     yerr=oi._merged[i]['OI_T3']['all']['ET3PHI'][w,:].flatten()[f],
                     linestyle='none', marker='.', capsize=2, color='k', alpha=0.2,
                     )
    
axV2.legend(fontsize=12)
axV2.set_xlabel(r'B/$\lambda$ (m/$\mu$m)')
axV2.set_ylabel(r'V$^2$')
axV2.set_yscale('log')
axV2.set_ylim([1e-4, 1e-1]) # zoom in on secont and third lobes

axT3.legend(fontsize=12)
axT3.set_xlabel(r'B$_\mathrm{max}$/$\lambda$ (m/$\mu$m)')
axT3.set_ylabel('T3PHI (deg)')

plt.tight_layout()
```


```python
import numpy as np
np.random.randn(10)

x = np.linspace(0,1,10)
y = 0.4*x + 5 
e = 0.1
#np.random.seed(123)
y += e*np.random.randn(len(y))

plt.close(10); plt.figure(10)

chi2 = np.mean((y - 0.4*x-5)**2/e**2)

plt.errorbar(x, y, yerr=e, label='chi2=%.3f'%(chi2))
plt.legend()
```

## How to access the parameters of the fit, uncertainties and correlations 
The dictionnary `oi.bestfit` contains the result of the fitting process. The information there are mostly self-explantory


```python
print('* best fit parameters:')
print('   fit: ', oi.bestfit['best'])
print('   boot:', oi.boot['best'])

print("* parameters' uncertainties:")
print('   fit: ', oi.bestfit['uncer'])
print('   boot:', oi.boot['uncer'])

print("* parameters' correlations (dict):")
print('   fit: ', oi.bestfit['cord'])
print('   boot:', oi.boot['cord'])
```

## Check how the fitted parameter and $\chi^2$ evolved during the minimisation
It can be interesting to check how the parameters evolved during the fitting process. The method `showFit` displays the information contained in `bestfit`, in particular `bestfit['track']`. For each parameter, the orange line and shadded area show the best fitted value and uncertainties. Data are plotted as function of iteration number.


```python
oi.showFit()
```


```python

```
