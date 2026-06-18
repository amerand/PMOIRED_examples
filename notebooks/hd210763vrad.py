import pmoired
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
data = {'RVa': [6.805, 77.493, 59.289, 48.205, 35.941, 21.353, 1.999, -4.649, -16.343,
                -22.477, 44.534, 7.924, -2.009, -7.632, -20.703, -16.946, 69.377,],
 'RVb': [24.530, -58.502, -36.978, -23.703, -9.283, 7.949, 30.262, 37.526,
         51.187,58.547,-19.768,23.192,34.779,41.407,56.594,52.280,-48.499,],
 'MJD': [59510.16368, 59533.15095, 59536.08947, 59538.11783, 59541.10544, 59546.07020, 59555.03566, 59855.02342,    
         59861.13714,59865.16489, 59870.14780, 59891.04787,59896.12739, 59899.06219, 59906.03501, 59910.02409, 
         59916.03523,]}

data['RVa_err']=0.072*np.ones(len(data['RVa']))
data['RVb_err']=0.172*np.ones(len(data['RVb']))

for k in data:
    data[k] = np.array(data[k])

Verr = 1.0
Vsign = 1

# -- sampling function
W = np.arange(len(data['MJD']))

C = {'catg':np.array([1]*len(data['MJD'])+[2]*len(data['MJD'])), 
    'err':{1:1, 2:1},
    'rho':{1:0.1, 2:0.1}
    }

def randomise(b=False):
    """
    randomise data by setting the index table 'W' if b=True, otherwise reset 'W'. 
    """
    global W
    if b:
        # -- random sampling with replacement
        W = np.random.randint(0, len(data['MJD']), len(data['MJD']))
    else:
        # -- all data
        W = np.arange(len(data['MJD']))

AX = {}
def resi(p, plot=False, altp=None):
    # extract orbital parameters:
    pp = pmoired.oimodels.computeLambdaParams(p)
    pp = {k.split('2,orb ')[1]:pp[k] for k in pp if k.startswith('2,orb ')}
    res = (data['RVa'][W] - Vsign*pmoired.oimodels._orbit(data['MJD'][W], pp, Vrad='a'))/data['RVa_err'][W]
    res = np.append(res, (data['RVb'][W] - Vsign*pmoired.oimodels._orbit(data['MJD'][W], pp, Vrad='b'))/data['RVb_err'][W])
    if not altp is None:
        if type(altp) is dict or altp is None:
            altp=[altp]
        # -- assumes list!
        for i,_a in enumerate(altp):
            _a = pmoired.oimodels.computeLambdaParams(_a)
            _a = {k.split('2,orb ')[1]:_a[k] for k in _a if k.startswith('2,orb ')}
            altp[i] = _a
        
    if not plot:
        return res
    else:
        if type(plot)==int:
            plt.close(plot)
            fig = plt.figure(plot, figsize=(pmoired.FIG_MAX_WIDTH, pmoired.FIG_MAX_WIDTH/2.5))
        print('chi2r rad only:', np.mean(res**2))
        spec = gridspec.GridSpec(ncols=2, nrows=2,
                                 width_ratios=[2, 1], 
                                 height_ratios=[1.5, 1], 
                                 wspace=0.15, hspace=0.25,
                                 top=0.98, right=0.98, left=0.1, bottom=0.2,
                                  )
        
        #ax1 = plt.subplot(221)
        #axp1 = plt.subplot(222, sharey=ax1)
        ax1 = plt.subplot(spec[0])
        axp1 = plt.subplot(spec[1], sharey=ax1)
        
        # -- William's plot
        #phi = ((data['MJD'][W]-pp['MJD0'])/pp['P'] + 0.5)%1. - 0.5
        #_phi = np.linspace(-0.6,0.6,200)

        phi = ((data['MJD'][W]-pp['MJD0'])/pp['P'])%1. 
        _phi = np.linspace(-0.1,1.1,100)
        
        ax1.errorbar(data['MJD'][W], data['RVa'][W], yerr=data['RVa_err'][W], color='b', 
                     linestyle='none', marker='o', label='data Ba')
        ax1.errorbar(data['MJD'][W], data['RVb'][W], yerr=data['RVb_err'][W], color='r', 
                     linestyle='none', marker='s', label='data Bb')
        for dphi in [-1,0,1]:
            axp1.errorbar(phi+dphi, data['RVa'][W], yerr=data['RVa_err'][W], color='b', 
                         linestyle='none', marker='o', label='data Ba')
            axp1.errorbar(phi+dphi, data['RVb'][W], yerr=data['RVb_err'][W], color='r', 
                         linestyle='none', marker='s', label='data Bb')
        
        t = np.linspace(min(data['MJD'])-0.05*np.ptp(data['MJD']), 
                        max(data['MJD'])+0.05*np.ptp(data['MJD']), 1000)
        
        Va = Vsign*pmoired.oimodels._orbit(t, pp, Vrad='a')
        ax1.plot(t, Va, '-b', alpha=0.5, label='model Ba')
        Vb = Vsign*pmoired.oimodels._orbit(t, pp, Vrad='b')
        ax1.plot(t, Vb, '--r', alpha=0.5, label='model Bb')
            
        _Va = Vsign*pmoired.oimodels._orbit(_phi*pp['P']+pp['MJD0'], pp, Vrad='a')
        axp1.plot(_phi, _Va, '-b', alpha=0.5, label='model Ba')
        _Vb = Vsign*pmoired.oimodels._orbit(_phi*pp['P']+pp['MJD0'], pp, Vrad='b')
        axp1.plot(_phi, _Vb, '--r', alpha=0.5, label='model Bb')

        ax1.legend(loc='center left', fontsize=7)
        ax1.set_ylabel('Vrad (km/s)')

        #ax2 = plt.subplot(223, sharex=ax1)
        ax2 = plt.subplot(spec[2], sharex=ax1)
        ax2.errorbar(data['MJD'][W], data['RVa'][W]-Vsign*pmoired.oimodels._orbit(data['MJD'][W], pp, Vrad='a'), 
                     yerr=data['RVa_err'][W], color='b', linestyle='none', marker='o', label='data Ba')
        ax2.errorbar(data['MJD'][W], data['RVb'][W]-Vsign*pmoired.oimodels._orbit(data['MJD'][W], pp, Vrad='b'), 
                     yerr=data['RVb_err'][W], color='r', linestyle='none', marker='s', label='data Bb')
            
        ax2.set_xlabel('MJD')
        ax2.set_ylabel('residuals (km/s)')
        ax2.hlines(0, ax2.get_xlim()[0], ax2.get_xlim()[1], color='0.5', linestyle='dotted')
        if not altp is None:
            for _a in altp:
                Vap = Vsign*pmoired.oimodels._orbit(t, _a, Vrad='a')
                Vbp = Vsign*pmoired.oimodels._orbit(t, _a, Vrad='b')
                
                ax1.plot(t, Vap, '-b', alpha=0.1, #label='alt model Ba'
                        )
                ax1.plot(t, Vbp, '--r', alpha=0.1, #label='alt model Bb'
                        )

                ax2.plot(data['MJD'][W], data['RVa'][W]-pmoired.oimodels._orbit(data['MJD'][W], _a, Vrad='a'), 
                     color='b', linestyle='none', marker='o', alpha=0.1)
                ax2.plot(data['MJD'][W], data['RVb'][W]-pmoired.oimodels._orbit(data['MJD'][W], _a, Vrad='b'), 
                    color='r', linestyle='none', marker='s', alpha=0.1)
                
                # ax2.plot(t, Vap-Va, '-b', alpha=0.1, #label='alt model Ba'
                #         )
                # ax2.plot(t, Vbp-Vb, '--r', alpha=0.1, #label='alt model Bb'
                #         )

                _Vap = Vsign*pmoired.oimodels._orbit(_phi*pp['P']+pp['MJD0'], _a, Vrad='a')
                _Vbp = Vsign*pmoired.oimodels._orbit(_phi*pp['P']+pp['MJD0'], _a, Vrad='b')
                
                axp1.plot(_phi, _Vap, '-b', alpha=0.1, #label='alt model Ba'
                         )
                axp1.plot(_phi, _Vbp, '--r', alpha=0.1, #label='alt model Bb'
                         )

        #axp2 = plt.subplot(224, sharex=axp1, sharey=ax2)
        axp2 = plt.subplot(spec[3], sharex=axp1, sharey=ax2)
        
        for dphi in [-1,0,1]:
            axp2.errorbar(phi+dphi, data['RVa'][W]-Vsign*pmoired.oimodels._orbit(data['MJD'][W], pp, Vrad='a'), 
                         yerr=data['RVa_err'][W], color='b', linestyle='none', marker='o', label='data Ba')
            axp2.errorbar(phi+dphi, data['RVb'][W]-Vsign*pmoired.oimodels._orbit(data['MJD'][W], pp, Vrad='b'), 
                         yerr=data['RVb_err'][W], color='r', linestyle='none', marker='s', label='data Bb')
        
        axp2.set_xlabel('orbital phase')
        axp2.hlines(0, axp2.get_xlim()[0], axp2.get_xlim()[1], color='0.5', linestyle='dotted')

        #ax1.set_ylim(-14, 16)

        Y = ax2.get_ylim()
        ax2.set_ylim(-np.abs(Y).max(), np.abs(Y).max())
        axp1.set_xlim(_phi.min(), _phi.max())
        AX['MJD'] = ax1
        AX['rMJD'] = ax2
        AX['phi'] = axp1
        AX['rphi'] = axp2
        #plt.suptitle('radial velocities from CRIRES+')
        #plt.tight_layout()
    return
                       
    
    