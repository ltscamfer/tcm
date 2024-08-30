import numpy as np


class DataBin:
    """ Data container for TCM processing. """

    def __init__(self, freq_min, freq_max, window_length,
                 window_overlap, az_min, az_max,
                 az_delta, search_2Hz):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.az_min = az_min
        self.az_max = az_max
        self.az_delta = az_delta
        self.search_2Hz = search_2Hz

    def build_data_arrays(self, st):
        """ Construct data arrays from Obspy Stream object. """
        # Assumes all traces have the same sample rate and length
        self.sampling_rate = st[0].stats.sampling_rate
        self.winlensamp = int(self.window_length * self.sampling_rate) # noqa
        # Sample increment (delta_t)
        self.sampinc = int((1 - self.window_overlap) * self.winlensamp) + 1
        # Time intervals to window data
        self.intervals = np.arange(0, len(st[0].data) - self.winlensamp, self.sampinc, dtype='int') # noqa
        self.nits = len(self.intervals)
        # Pull time vector from stream object
        self.tvec = st[0].times('matplotlib')

        self.sub_window = int(np.round(self.winlensamp / 4))
        self.noverlap = int(self.sub_window * 0.5)
        self.window = 'hann'
        self.t = np.full(self.nits, np.nan)
        # Assign the traces to individual arrays
        if len(st) == 4:
            self.Infra = st[1].data
            self.E = st[0].data
            self.N = st[2].data
            self.Z = st[3].data
        else:
            raise TypeError("The input data stream must have 4 channels.")
