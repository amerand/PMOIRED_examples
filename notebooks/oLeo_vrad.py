import numpy as np
import matplotlib.pyplot as plt
import pmoired 

# -- Data from https://www.aanda.org/articles/aa/pdf/2023/04/aa45712-22.pdf, table C.1
MJD =  [59509.36177, 59533.34135, 59536.33618, 59547.33935, 59566.28758, 59589.32020, 59592.12908, 59595.12107, 59596.12677, 59903.33916, 
        59959.35809, 55259.95943, 55261.95506, 55266.99879, 55268.02101, 55270.95938, 55271.96791, 55281.86844, 55283.91363, 55285.87765]
V1 =  [-28.463, 58.279, -7.796, 66.906, -22.310, 80.659, 41.217, -20.433, -28.252, -0.123,
       -27.395, -16.980, -27.322, 66.284, 78.391, 57.327, 35.285, 71.859, 78.545, 48.729]
eV1 =  [0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085,
        0.085, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054, 0.054]
V2 =  [88.090,-9.871,64.638,-19.924,80.770,-35.189,9.082,78.642,87.513,55.737, 86.393,
       75.010, 86.810, -18.776, -32.472, -8.642, 16.053, -24.846, -32.661, 1.006]
eV2 =  [0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085,
        0.085, 0.082, 0.082, 0.082, 0.082, 0.082, 0.082, 0.082, 0.082, 0.082]

MJD, V1, eV1, V2, eV2 = np.array(MJD), np.array(V1), np.array(eV1), np.array(V2), np.array(eV2)

# -- sampling function
W = np.arange(len(MJD))

def randomise(b=False):
    """
    randomise data by setting the index table 'W' if b=True, otherwise reset 'W'. 
    """
    global W
    if b:
        # -- random sampling with replacement
        W = np.random.randint(0, len(MJD), len(MJD))
    else:
        # -- all data
        W = np.arange(len(MJD))
        
def resiVrad(p, fig=False):
    """
    return residuals to radial velocities based on parameters 'p'. 

    see also: pmoired.oimodels._orbit
    """
    # -- compute parameters defined as function of others:
    t = pmoired.oimodels.computeLambdaParams(p)
    
    # -- extract only the orbital parameters
    t = {k.split('2,orb ')[1]:t[k] for k in p if k.startswith('2,orb ')}

    # -- compute residuals to velocity of primary
    v1 = pmoired.oimodels._orbit(MJD[W], t, Vrad='a')
    res = (V1[W] - v1)/eV1[W]
    
    # -- compute residuals to velocity of secondary
    v2 = pmoired.oimodels._orbit(MJD[W], t, Vrad='b')
    res = np.append(res, (V2[W] - v2)/eV2[W])
    
    if not fig:
        return res
    
    # -- plot
    phi = ((MJD-t['MJD0'])/t['P'])%1
    _phi = np.linspace(-0.1,1.1,240)
    _mjd = t['MJD0']+_phi*t['P']

    plt.close(int(fig))
    plt.figure(int(fig), figsize= (pmoired.FIG_MAX_WIDTH*0.5, 
                                    min(pmoired.FIG_MAX_WIDTH*0.5, pmoired.FIG_MAX_HEIGHT)))

    ax1 = plt.subplot(211)
    plt.plot(phi[W], V1[W], 'or', alpha=0.3, label='1')
    plt.plot(_phi, pmoired.oimodels._orbit(_mjd, t, Vrad='a'), '-r')
    plt.plot(phi[W], V2[W], 'sb', alpha=0.3, label='2')
    plt.plot(_phi, pmoired.oimodels._orbit(_mjd, t, Vrad='b'), '-b')
    plt.hlines(t['gamma'], -0.1, 1.1, linestyle='dotted', color='k', alpha=0.2)
    plt.ylabel('velocity (km/s)')        
    plt.legend()

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(phi[W], V1[W]-v1, 'or', alpha=0.5)
    plt.errorbar(phi[W], V1[W]-v1, yerr=eV1[W], linestyle='none', color='r')
    plt.plot(phi[W], V2[W]-v2, 'sb', alpha=0.5)
    plt.errorbar(phi[W], V2[W]-v2, yerr=eV2[W], linestyle='none', color='b')
    plt.ylabel('ressiduals (km/s)')
    plt.xlabel('orbital phase')
    plt.hlines(0, -0.1, 1.1, linestyle='dotted', color='k', alpha=0.2)
    plt.xlim(-0.05,1.05)
    plt.tight_layout()

def showOrbit(p, mjd=None, fig=0):
    # -- compute parameters defined as function of others:
    t = pmoired.oimodels.computeLambdaParams(p)
    
    # -- extract only the orbital parameters
    t = {k.split('2,orb ')[1]:t[k] for k in p if k.startswith('2,orb ')}
    print(t)
    _phi = np.linspace(0.,1.,240)
    _mjd = t['MJD0']+_phi*t['P']
    x, y = pmoired.oimodels._orbit(_mjd, t)
    plt.close(fig)
    plt.figure(fig)
    ax = plt.subplot(111, aspect='equal')

    plt.plot(x, y, '-k', alpha=0.5, linewidth=2)
    plt.plot(0,0,'*r', label='1')
    
    ax.invert_xaxis()
    if not mjd is None:
        phi = ((np.array(mjd)-t['MJD0'])/t['P'])%1
        x, y = pmoired.oimodels._orbit(np.array(mjd), t)
        plt.plot(x, y, 'sb', alpha=0.5, label='2')
        for i in range(len(mjd)):
            plt.text(x[i], y[i], '%.2f'%mjd[i], fontsize=6, rotation=-45, ha='center', va='center', )
    plt.legend()
    plt.xlabel(r'E $\leftarrow$ (mas)')
    plt.ylabel(r'$\rightarrow$ N (mas)')
    