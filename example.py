from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from tcm import tcm
from tcm.tools import plotting
from tcm.classes import tcm_data_class
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
#%%
rcParams.update({'font.size': 10})

# Filter range [Hz]
freq_min = 2.0
freq_max = 16.0

# Use 2 Hz narrowband [True] or broadband [False] coherence maxima for calculation
search_2Hz = False

# Window length [sec]
window_length = 15.0

# Fraction of window overlap [0.0, 1.0)
window_overlap = 0.90

# Azimuths to scans over [degrees]
az_min = 0.0
az_max = 359.0
az_delta = 1.0

STARTTIME = UTCDateTime('2012-10-16T04:44:00')
ENDTIME = STARTTIME + 2*60

# Station information, including fdsn/etc client to read from
SOURCE = 'IRIS'
NETWORK = 'TA'
STATION = 'Z45A'
LOCATION = '*'
CHANNEL = 'BDF,BHZ,BHN,BHE'

source_lat = 32.571447 # Source latitude (Camp Minden)
source_lon = -93.369086 # Source longitude (Camp Minden)

#%% Read in data and pre-process
client = Client(SOURCE)
st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNEL,
                          starttime=STARTTIME, endtime=ENDTIME,
                          attach_response='True')
inv = client.get_stations(network=NETWORK, station=STATION, location=LOCATION, channel=CHANNEL, starttime=STARTTIME, endtime=ENDTIME, level="response")
# Remove response
for tr in st:
    fs_resp = tr.stats.sampling_rate
    # Pre-filt for response removal
    pre_filt = [0.0005, 0.001, fs_resp/2-2, fs_resp/2]
    if tr.stats.channel[1:] == 'DF':
        tr.remove_response(pre_filt=pre_filt, output='VEL', water_level=None)
    else:
        tr.remove_response(pre_filt=pre_filt, output='DISP', water_level=None)

# Sort by component: E, F, N, Z
st.sort(keys=['component'])

# Ensure sample rate consistency
st.interpolate(sampling_rate=st[0].stats.sampling_rate, method='lanczos', a=15)
st.detrend(type='linear')

#%% Run the transverse coherence minimization (TCM) algorithm
station_lat = inv[0][0].latitude
station_lon = inv[0][0].longitude

distance_m, backaz, _ = gps2dist_azimuth(station_lat, station_lon, source_lat, source_lon)
distance_km = np.round(distance_m / 1000,decimals=1)

baz_weighted, baz_unweighted, sigma_weighted, sigma_unweighted, time_smooth, frequency_vector, time, Cxy2, mean_coherence, median_coherence, freq_lim_min, freq_lim_max = tcm.run_tcm(st, freq_min, freq_max, window_length, window_overlap, az_min, az_max, az_delta, search_2Hz) # noqa
data = tcm_data_class.DataBin(freq_min, freq_max, window_length, window_overlap, az_min, az_max, az_delta, search_2Hz)
data.build_data_arrays(st)
#%% Plot the results
#-------------------------------------------------------------------------------------------------
# Here are the options for plot_method:
# 'weighted' : only plots the backazimuth estimates using the weighted coherence
# 'unweighted' : only plots the backazimuth estimates using only the minimized transverse-infrasound coherence
# 'both' : plots both the weighted and unweighted backazimuth estimates
#-------------------------------------------------------------------------------------------------
fig, axs = plotting.tcm_plot(st, data, freq_min, freq_max, baz_weighted, baz_unweighted,
                             time_smooth, frequency_vector, time,
                             Cxy2, mean_coherence, median_coherence, freq_lim_min, freq_lim_max,
                             search_2Hz, plot_method = 'both')
# Plot uncertainties
#axs[4].scatter(time_smooth, baz_weighted + sigma_weighted, c='gray', marker='_', linestyle=':')
#axs[4].scatter(time_smooth, baz_weighted - sigma_weighted, c='gray', marker='_', linestyle=':')
#axs[4].scatter(time_smooth, baz_unweighted + sigma_unweighted, c='gray', marker='_', linestyle=':')
#axs[4].scatter(time_smooth, baz_unweighted - sigma_unweighted, c='gray', marker='_', linestyle=':')

axs[4].axhline(backaz, color='blue', linestyle='--', linewidth=1, label='Ground-truth backazimuth')
if backaz < 180:
    axs[4].axhline(backaz + 180, color='green', linestyle='--', linewidth=1,label='Ground truth backazimuth + 180')
else:
    axs[4].axhline(backaz - 180, color='green', linestyle='--', linewidth=1,label='Ground truth backazimuth - 180')

#-----------------------------------------------------------CUSTOMIZE Y LIMIT AND TICKS HERE
ticks = np.arange(0, 370, 10)
axs[4].set_yticks(ticks)
axs[4].set_ylim(250, 265)
#-----------------------------------------------------------CUSTOMIZE Y LIMIT AND TICKS HERE
axs[4].legend(loc='lower left')
plt.suptitle(f'STATION: {STATION}, {str(distance_km)[:-2]} km from source\nSTART TIME: {str(STARTTIME)[:-8]}\nWINDOW LENGTH: {str(window_length)[:-2]} s, OVERLAP: {str(window_overlap*100)[:-2]}%', fontsize=20)
#plt.savefig(f"/Users/ltscamfer/ltscamfer_uaf/figures/TCM_comparison/{STATION}_method_comparison.png", dpi=300)
plt.show()