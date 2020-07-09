from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
from lightkurve.utils import TessQualityFlags
import pandas as pd

from .gaia import get_nearby_gaia
from .utils import amplitude_spectrum

class Platypus:
    def __init__(self, target):
        self.target = target
        self.tpfs = []

    def correct(self, threshold=60, npca=5, cutout_size=50):
        print("Downloading cutout")
        self.download_tpf(cutout_size=cutout_size)

        lcs = []
        for tpf in self.tpfs[:]:
            lcs.append(self.correct_tpf(tpf, threshold=threshold, npca=npca))
        lcs = lk.LightCurveCollection(lcs)
        
        self.corr = lcs
        return lcs

    def download_tpf(self, cutout_size):
        self.tpfs = lk.search_tesscut(self.target).download_all(cutout_size=(cutout_size, cutout_size))
        print(self.tpfs)

    def correct_tpf(self, tpf, threshold=5, npca=5, diagnose=True):
        aper = tpf.create_threshold_mask(threshold=threshold)
        raw_lc = tpf.to_lightcurve(aperture_mask=aper)
        m = np.isnan(raw_lc.flux_err) | np.isnan(raw_lc.flux) | (raw_lc.flux <= 0) | (raw_lc.flux_err <= 0)
        raw_lc = raw_lc[~m]

        # Design matrix
        dm = lk.DesignMatrix(tpf.flux[~m][:, ~aper], name='regressors').pca(npca).append_constant()
        rc = lk.RegressionCorrector(raw_lc)
        corrected_lc = rc.correct(dm)

        # Scattered light removal
        corrected_lc = raw_lc - rc.model_lc + np.percentile(rc.model_lc.flux, 5)
        corrected_lc = corrected_lc.remove_nans().normalize()
        bitmask = TessQualityFlags.create_quality_mask(corrected_lc.quality, bitmask='hard')
        corrected_lc = corrected_lc[bitmask]
        
        if diagnose:
            self.diagnostic_plot(tpf, aper, dm, rc, corrected_lc, npca)
        
        return corrected_lc

    def diagnostic_plot(self, tpf, aper, dm, rc, corrected_lc, npca):
        fig = plt.figure(figsize=[11., 8.], constrained_layout=True)
        gs = fig.add_gridspec(5,5)

        ax = fig.add_subplot(gs[0:2, 0:2])
        tpf.plot(aperture_mask=aper, ax=ax, cmap='Blues',
                show_colorbar=False,
                mask_color='lightgrey');

        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        try:
            src = get_nearby_gaia(tpf)
            ax.scatter(src['x'], src['y'], s=src['size'].values, alpha=1,cmap='Reds_r',
                c=src['Gmag'],#colors[m],
                    zorder=50)
            ax.set(xlim=xlims, ylim=ylims)
        except:
            pass
        ax.set_title(tpf.targetid)
        
        ax = fig.add_subplot(gs[0:1, 2:5])
        time, flux = corrected_lc.time, corrected_lc.flux
        ax.plot(time, flux, c='black', lw=0.7)
        ax.set(xlabel='Time [BTJD]', ylabel='Flux', xlim=[time[0], time[-1]], title=f'Sector {tpf.sector}')
        
        ax = fig.add_subplot(gs[1:2, 2:5])
        freq, amp = amplitude_spectrum(time, flux)
        ax.plot(freq, amp*1e3, c='black', lw=0.7)
        ax.set(xlabel='Frequency [1/day]', ylabel='Amplitude [ppt]', xlim=[freq[0], freq[-1]], ylim=[0, None])
        # ax.set_title('amp spec')

        ax = fig.add_subplot(gs[2:3, 0:4])
        ax.plot(rc.lc.time, rc.lc.flux, label='Original')
        for key in rc.diagnostic_lightcurves.keys():
            m = (rc.diagnostic_lightcurves[key] - np.median(rc.diagnostic_lightcurves[key].flux) + np.median(rc.lc.flux))
            ax.plot(m.time, m.flux, label='Regressors', c='red')
        ax.set(xlabel='Time [BTJD]', ylabel='Flux [e/s]')
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[3:4, 0:4])
        ax.plot(rc.lc.time, rc.lc.flux, label='Original')
        m = rc.corrected_lc[~rc.cadence_mask]
        ax.scatter(m.time, m.flux, label='Outliers', marker='x', c='r')
        ax.plot(rc.corrected_lc.time, rc.corrected_lc.flux, label='Corrected', c='k')
        ax.legend(fontsize=8)
        ax.set(xlabel='Time [BTJD]', ylabel='Flux [e/s]')

        ax = fig.add_subplot(gs[2:4, 4:5])
        ax.imshow(dm.values[:,:-1], cmap='coolwarm', aspect='auto', interpolation='none')
        ax.set(xlabel='Component', ylabel='X')

    def plot(self):
        tpfs.plot()