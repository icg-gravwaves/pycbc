from scipy import signal

import pycbc.psd
import pycbc.waveform
import pycbc.strain
import pycbc.noise


def generate_data_lisa_ew(
    waveform_params,
    psds_for_datagen,
    psds_for_whitening,
    window_length,
    cutoff_time,
    sample_rate,
    extra_forward_zeroes=0,
    seed=137,
    zero_noise=False,
    no_signal=False,
):
    """Generate data for early warning data.
    
    Parameters
    ----------
    waveform_params : dict
        Dictionary of waveform parameters
    psds_for_datagen : dict
        PSDs for data generation.
    psds_for_whitening : dict
        PSDs for whitening.
    window_length : int
        Length of the hann window use to taper the start of the data.
    cutoff_time : float
        Time before merge to cutoff the data.
    sample_rate : float
        Sampling rate in Hz. 
    extra_forward_zeros : float
        Duration at the start of the data to set to zero.
    seed : int
        Random seed used for generating the noise.
    zero_noise : bool
        If true, the noise will be set to zero.
    no_signal : bool
        If true, the signal will not be added to data and only noise will
        be returned.

    Returns
    -------
    pycbc.types.TimeSeries
        Data for the LISA A channel
    pycbc.types.TimeSeries
        Data for the LISA E channel
    """
    
    window = signal.windows.hann(window_length * 2 + 1)[:window_length]

    nefz = int(extra_forward_zeroes * sample_rate)
    nctf = int(cutoff_time * sample_rate)

    outs = pycbc.waveform.get_fd_det_waveform(
        ifos=['LISA_A','LISA_E','LISA_T'],
        **waveform_params
    )
    outs['LISA_A'].resize(len(psds_for_datagen["LISA_A"]))
    outs['LISA_E'].resize(len(psds_for_datagen["LISA_E"]))

    outs['LISA_A'] = outs['LISA_A'].cyclic_time_shift(-waveform_params['additional_end_data'])
    outs['LISA_E'] = outs['LISA_E'].cyclic_time_shift(-waveform_params['additional_end_data'])
    tout_A = outs['LISA_A'].to_timeseries()
    tout_E = outs['LISA_E'].to_timeseries()

    strain_w_A = pycbc.noise.noise_from_psd(
        len(tout_A),
        tout_A.delta_t,
        psds_for_datagen['LISA_A'],
        seed=seed,
    )
    strain_w_E = pycbc.noise.noise_from_psd(
        len(tout_E),
        tout_E.delta_t,
        psds_for_datagen['LISA_E'],
        seed=seed + 1,
    )

    # We need to make sure the noise times match the signal
    strain_w_A._epoch = tout_A._epoch
    strain_w_E._epoch = tout_E._epoch

    if zero_noise:
        strain_w_A *= 0.0
        strain_w_E *= 0.0

    if not no_signal:
        strain_w_A[:] += tout_A[:]
        strain_w_E[:] += tout_E[:]

    strain_w_A.data[:nefz] = 0
    strain_w_A.data[nefz:nefz + window_length] *= window
    strain_w_A.data[-nctf:] = 0

    strain_w_E.data[:nefz] = 0
    strain_w_E.data[nefz:nefz+window_length] *= window
    strain_w_E.data[-nctf:] = 0

    strain_fout_A = pycbc.strain.strain.execute_cached_fft(
        strain_w_A,
        copy_output=False,
        uid=1236
    )
    strain_fout_A = strain_fout_A * (psds_for_whitening['LISA_A']).conj()
    strain_ww_A = pycbc.strain.strain.execute_cached_ifft(
        strain_fout_A,
        copy_output=False,
        uid=1237
    )
    strain_ww_A.data[:nefz] = 0
    strain_ww_A.data[-nctf:] = 0

    strain_fout_E = pycbc.strain.strain.execute_cached_fft(
        strain_w_E,
        copy_output=False,
        uid=1238
    )
    strain_fout_E = strain_fout_E * (psds_for_whitening['LISA_E']).conj()
    strain_ww_E = pycbc.strain.strain.execute_cached_ifft(
        strain_fout_E,
        copy_output=False,
        uid=1239
    )
    strain_ww_E.data[:nefz] = 0
    strain_ww_E.data[-nctf:] = 0

    return strain_ww_A, strain_ww_E


_WINDOW = None
def generate_waveform_lisa_ew(
    waveform_params,
    psds_for_whitening,
    sample_rate,
    window_length,
    cutoff_time,
    kernel_length,
    extra_forward_zeroes=0,
):
    """
    
    Parameters
    ----------
    waveform_params: dict
        A dictionary of waveform parameters that will be passed to the waveform
        generator.
    psds_for_whitening: dict[str: FrequencySeries]
        Power spectral denisities for whitening in the frequency-domain.
    sample_rate : float
        Sampling rate.
    window_length : int
        Length (in samples) of time-domain window applied to the start of the
        waveform.
    cutoff_time: float
        Time (in seconds) from the end of the waveform to cutoff.
    kernel_length : int
        Unused.
    extra_foward_zeroes : int
        Time (in seconds) to set to zero at the start of the waveform. If used,
        the window will be applied starting after the zeroes.
    """
    global _WINDOW
    if _WINDOW is None or len(_WINDOW) != window_length:
        _WINDOW = signal.windows.hann(window_length * 2 + 1)[:window_length]
    window = _WINDOW
    
    nefz = int(extra_forward_zeroes * sample_rate)
    nctf = int(cutoff_time * sample_rate)

    outs = pycbc.waveform.get_fd_det_waveform(ifos=['LISA_A','LISA_E','LISA_T'], **waveform_params)
    tout_A = pycbc.strain.strain.execute_cached_ifft(
        outs['LISA_A'],
        copy_output=False,
        uid=1234
    )
    tout_E = pycbc.strain.strain.execute_cached_ifft(
        outs['LISA_E'],
        copy_output=False,
        uid=1233
    )
    
    if extra_forward_zeroes:
        tout_A.data[:nefz] = 0
    tout_A.data[nefz:nefz+window_length] *= window
    tout_A.data[-nctf:] = 0
    
    if extra_forward_zeroes:
        tout_E.data[:nefz] = 0
    tout_E.data[nefz:nefz+window_length] *= window
    tout_E.data[-nctf:] = 0

    fout_A = pycbc.strain.strain.execute_cached_fft(
        tout_A,
        copy_output=True,
        uid=1235
    )
    fout_A.data[:] = fout_A.data[:] * (psds_for_whitening['LISA_A'].data[:]).conj()

    fout_E = pycbc.strain.strain.execute_cached_fft(
        tout_E,
        copy_output=True,
        uid=12350
    )
    fout_E.data[:] = fout_E.data[:] * (psds_for_whitening['LISA_E'].data[:]).conj()

    fout_ww_A = pycbc.strain.strain.execute_cached_ifft(
        fout_A,
        copy_output=False,
        uid=5237
    )

    if extra_forward_zeroes:
        fout_ww_A.data[:nefz] = 0
    fout_ww_A.data[-nctf:] = 0

    fout_A = pycbc.strain.strain.execute_cached_fft(
        fout_ww_A,
        copy_output=False,
        uid=5238
    )

    fout_ww_E = pycbc.strain.strain.execute_cached_ifft(
        fout_E,
        copy_output=False,
        uid=5247
    )
    if extra_forward_zeroes:
        fout_ww_E.data[:nefz] = 0
    fout_ww_E.data[-nctf:] = 0

    fout_E = pycbc.strain.strain.execute_cached_fft(
        fout_ww_E,
        copy_output=False,
        uid=5248
    )

    fouts = {
        'LISA_A': fout_A,
        'LISA_E': fout_E,
    }

    return fouts
