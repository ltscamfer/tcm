import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from scipy.signal import welch
import numpy as np


colorm = LinearSegmentedColormap.from_list('', ['white', *plt.cm.get_cmap('magma_r').colors])

def tcm_plot(st, data, freq_min, freq_max, baz_weighted, baz_unweighted, time_smooth, freq_vector, time, Cxy2, mean_coherence, median_coherence, freq_min_array, freq_max_array, search_2Hz, plot_method):  # noqa
    """ Return a plot of the TCM results.

    Plots (a) the vertical seismic trace, (b) the magnitude squared coherence
     between the infrasound signal and the vertical seismic displacement,
     and (c) the estimated back-azimuth from the TCM algorithm.

    Args:
        st (stream): Obspy stream.
        freq_min (float):
        freq_max (float):
        baz_weighted (array):
        baz_unweighted (array):
        time_smooth (array):
        freq_vector (array):
        time (array):
        Cxy2 (array):
        mean_coherence (array):
        median_coherence (array):
        freq_min_array (array):
        freq_max_array (array):
        search_2Hz: (bool): Calculate and plot 2 Hz coherence band
        plot_method (str): chooses baz estimate to use. 'weighted', 'unweighted', or 'both'

    Returns:
        (tuple):
            ``fig``: Output figure handle.
            ``axs``: Output axis handle.
    """
    #filter the data for plotting
    stf = st.copy()
    stf.detrend(type='linear')
    stf.taper(max_percentage=.02)
    stf.filter('bandpass', freqmin=freq_min, freqmax=freq_max, corners=2,
               zerophase=True)

    # Specify the colormap.
    cm = colorm

    # Colorbar/y-axis limits for the vertical coherence
    c_lim = [0.4, 1.0]
    spec_yl = [0.1, freq_max]
    tr_z = st[3]
    tr_f = st[1]

    # Specify the time vector for plotting the trace.
    tvec_f = tr_f.times('matplotlib')
    tvec_z = tr_z.times('matplotlib')
    # Pressure spectrogram
    PspecdB = np.full((len(freq_vector), data.nits), np.nan)
    for jj in range(0, data.nits):  # loop through time windows
        t0_ind = data.intervals[jj]  # time at window start
        tf_ind = data.intervals[jj] + data.winlensamp  # time at window end
        try:
            data.t[jj] = data.tvec[t0_ind + int(np.round(data.winlensamp / 2))]
        except:
            data.t[jj] = np.nanmax(data.t, axis=0)

        f, Pspec = welch(data.Infra[t0_ind:tf_ind], fs=data.sampling_rate,  # pressure spectrogram
                         window=data.window, scaling='density', nperseg=data.sub_window,
                         noverlap=data.noverlap)
        PspecdB[:, jj] = 10 * np.log10(abs(Pspec) / np.power(20e-6, 2))

    cmin = np.nanpercentile(PspecdB, 15)
    cmax = np.nanpercentile(PspecdB, 99.5)

    fig, axs = plt.subplots(5, 1, sharex='col', figsize=(10, 10))
    # Infrasound
    axs[0].plot(tvec_f, stf[1].data, c='k')
    axs[0].set_ylabel('Pressure \n [Pa]')
    axs[0].text(0.75, 0.8, tr_f.id, transform=axs[0].transAxes)

    # Vertical component of seismic trace (displacement)
    axs[1].plot(tvec_z, stf[3].data, c='k')
    axs[1].set_ylabel('Displacement \n [m]')
    axs[1].text(0.75, 0.8, tr_z.id, transform=axs[1].transAxes)

    # Pressure spectrogram
    im = axs[2].imshow(PspecdB, extent=[tvec_f[0], tvec_f[-1], f[0], f[-1]],
                        origin='lower', aspect='auto', interpolation=None, cmap=colorm)
    axs[2].set_yscale('linear')
    im.set_clim(cmin, cmax)
    axs[2].set_ylabel('Frequency \n [Hz]')
    axs[2].set_ylim(spec_yl[0], spec_yl[1])
    axs[2].set_xlim(tvec_f[0], tvec_f[-1])

    pos1 = axs[2].get_position()
    cloc = [pos1.x0+pos1.width+0.03, pos1.y0, 0.02, pos1.height]
    cbaxes = fig.add_axes(cloc)
    hc = plt.colorbar(im, cax=cbaxes)
    hc.set_label('PSD [dB re 20\u03bc$Pa^2$/Hz]')

    # Magnitude-squared coherence
    sc0 = axs[3].pcolormesh(time, freq_vector, Cxy2,
                            cmap=cm, shading='auto')
    axs[3].axis('tight')
    axs[3].set_xlim(time[0], time[-1])
    axs[3].set_ylim(freq_min, freq_max)
    axs[3].set_ylabel('Frequency \n [Hz]')
    p1 = axs[3].get_position()
    sc0.set_clim(c_lim)

    if search_2Hz:
        #plot narrow band frequency boxes, align time vector with coherence
        tdiff = time[1]-time[0]
        for i in range(len(time)):
            rect = patches.Rectangle((time[i]+tdiff/2,freq_min_array[i]), tdiff, 2,
                                     linewidth=.5, edgecolor='k', facecolor='none')
            axs[3].add_patch(rect)

    # Back-azimuth Estimate (choosing between weighted, unweighted, or both).
    if plot_method == 'weighted':
        sc1 = axs[4].scatter(time_smooth, baz_weighted, c=mean_coherence, cmap=cm,
                             edgecolors=None, lw=0.3)
        axs[4].set_ylim(0, 360)
        axs[4].set_yticks([0, 90, 180, 270, 360])
        axs[4].set_ylabel('Back-Azimuth \n [Deg.]')
        p2 = axs[4].get_position()
        sc1.set_clim(c_lim)

        cbot = p2.y0
        ctop = p1.y1
        cbaxes = fig.add_axes([p2.x0+p2.width+0.03, cbot, 0.02, ctop-cbot])
        hc = plt.colorbar(sc0, cax=cbaxes)
        hc.set_label('Max Weighted Coherence')

    if plot_method == 'unweighted':
        sc1 = axs[4].scatter(time_smooth, baz_unweighted, c=median_coherence, cmap=cm,
                             edgecolors=None, lw=0.3)
        axs[4].set_ylim(0, 360)
        axs[4].set_yticks([0, 90, 180, 270, 360])
        axs[4].set_ylabel('Back-Azimuth \n [Deg.]')
        p2 = axs[4].get_position()
        sc1.set_clim(c_lim)

        cbot = p2.y0
        ctop = p1.y1
        cbaxes = fig.add_axes([p2.x0 + p2.width + 0.03, cbot, 0.02, ctop - cbot])
        hc = plt.colorbar(sc0, cax=cbaxes)
        hc.set_label('Median Coherence')

    if plot_method == 'both':
        sc1 = axs[4].scatter(time_smooth, baz_unweighted, c=median_coherence, cmap=cm,
                             marker= "D", edgecolors=None, lw=0.2)
        sc2 = axs[4].scatter(time_smooth, baz_weighted, c=mean_coherence, cmap=cm,
                             edgecolors=None, lw=0.2)
        axs[4].set_ylim(0, 360)
        axs[4].set_yticks([0, 90, 180, 270, 360])
        axs[4].set_ylabel('Back-Azimuth \n [Deg.]')
        p2 = axs[4].get_position()
        sc1.set_clim(c_lim)
        sc2.set_clim(c_lim)

        cbot = p2.y0
        ctop = p1.y1
        cbaxes = fig.add_axes([p2.x0 + p2.width + 0.03, cbot, 0.02, ctop - cbot])
        hc = plt.colorbar(sc0, cax=cbaxes)
        hc.set_label('Median and Max Weighted Coherence')
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor=cm(1.0), markersize=10, label='Unweighted'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cm(1.0), markersize=10, label='Weighted')]
        legend1 = axs[4].legend(handles=legend_elements, loc='upper left')
        axs[4].add_artist(legend1)
    axs[4].xaxis_date()
    axs[4].tick_params(axis='x', labelbottom='on')
    axs[4].fmt_xdata = dates.DateFormatter('%HH:%MM')
    axs[4].xaxis.set_major_formatter(dates.DateFormatter("%H:%M:%S"))
    axs[4].set_xlabel('UTC Time')

    return fig, axs
