"""
qblox_data_treatment.py
-----------------------

Module to process qblox data more conveniently.

Authors: jbv, sp

"""

import copy
import inspect
import typing
import matplotlib.pyplot as plt
import numpy as np
import scipy


def echo_data_treatment(signal, echo_start, echo_stop, nco_f, tres=1e-9,
                        baseline=True, baseline_start=200, baseline_stop=400,
                        downconvert=True,
                        filt=True, filt_opt={'bandwidth': 200e6, 'centre':0},
                        ph_corr=True,
                        plot=False,
                        plot_dem_filt=False):

    n_pt = len(signal[echo_start:echo_stop])
    t = np.linspace(0, tres*(n_pt-1)*1e9, n_pt)

    if plot:
        plt.figure(figsize=(12,4))
        plt.plot(t, np.real(signal[echo_start:echo_stop]))
        plt.plot(t, np.imag(signal[echo_start:echo_stop]))
        plt.xlim(t[0], t[-1])
        plt.xlabel('t (ns)')
        plt.title('untreated signal')
        plt.show()
    
    # baseline correction
    if baseline:
        signal -= np.average(
                      signal[baseline_start:baseline_stop])
    
    # crop the signal
    signal = signal[echo_start:echo_stop]
    
    # downconvert the echo
    if downconvert:
        signal = demodulate_fft_shift(signal,
                                      nco_f, tres=1e-9,
                                      plot=plot_dem_filt)
    
    # filter the echo (200MHz spectral window)
    if filt:
        signal = gaussian_filter(signal, plot=plot_dem_filt, **filt_opt)
        t = np.linspace(0, t[-1], len(signal))

    # phase correction
    if ph_corr:
        signal = auto_0th_ph_corr(np.array(signal))

    if plot:
        plt.figure(figsize=(12,4))
        plt.plot(t, np.real(signal))
        plt.plot(t, np.imag(signal))
        plt.xlim(t[0], t[-1])
        plt.xlabel('t (ns)')
        plt.title('treated signal')
        plt.show()

    return t, signal


def auto_0th_ph_corr(signal):
    """
    Automatically apply 0th order phase correction to maximise a signal real
    part.

    Parameters
    ----------
    signal: ndarray
        1D-array of complex numbers with signal to phase

    Returns
    -------
    signal_phased: ndarray
        1D-array of complex numbers with phased signal
    """
    signal_phased = signal  # TODO: maybe needs deepcopy if list is passed to the function.

    for phi0 in np.linspace(-180, 180, 10*360+1):
        signal_phi0 = signal * np.exp(-1j * phi0 * np.pi/180)
        if sum(np.real(signal_phi0)) > sum(np.real(signal_phased)):
            signal_phased = signal_phi0

    return signal_phased


def gaussian_filter(signal, tres= 1e-9, bandwidth=200e6, centre=0,
                    supergaussian_index=40, plot=False):
    """
    Apply a filter to a time-domain signal in the frequency domain with a
    superGaussian.
    
    Parameters
    ----------
    signal : ndarray
        1D-array of complex numbers with signal to filter
    tres: float, default 1e-9
        signal timing resolution, sampling period (s)
    bandwidth : float, default 200e-6
        bandwidth of the filter (Hz)
    centre : float, default 0
        centre position of the filter (Hz)
    supergaussian_index : int, default 40
        superGaussian power index
    plot : boolean, default False
        allows to plot signal pre- and post-filtering, in both time and
        frequency domains

    Returns
    -------
    signal_dem_filtered : ndarray
        1D-array of complex numbers with filtered signal
    """
    spec= np.fft.fftshift(np.fft.fft(signal, n=8*len(signal)))
    sampling_rate = 1/tres
    f=np.linspace(-sampling_rate/2, sampling_rate/2, len(spec))
    
    # Gaussian filter
    b, b0, c, p = bandwidth, sampling_rate, centre, supergaussian_index
    # NB: sampling_rate = bandwidth of the spectrum to filter
    grid = (np.linspace(-b0/2, b0/2, len(spec))-c)/b
    supergaussian = np.exp(-(2 ** (p + 1)) * (grid ** p))

    filtered_spec = spec * supergaussian
    
    filtered_f, filtered_spec_cut = list(), list()
    for i, point in enumerate(filtered_spec):
        if c-b/2 <= f[i] <= c+b/2:
            filtered_f.append(f[i])
            filtered_spec_cut.append(point)

    filtered_spec_cut = np.asarray(filtered_spec_cut)
    
    filtered_f = np.asarray(filtered_f)
                        
    signal_filtered = np.fft.ifft(np.fft.ifftshift(
                              filtered_spec_cut))[0:int(len(filtered_spec_cut)/8)]
    # normalisation
    signal_filtered *= b/b0

    if plot:

        plt.subplots(2, 2, figsize=(15, 8 / 1.61))
        
        plt.subplot(221)
        plt.plot(1e-6*f, np.real(spec))
        plt.plot(1e-6*f, np.imag(spec))
        plt.plot(1e-6*f, supergaussian*np.max(np.abs(spec)))
        plt.xlim(1e-6*f[0], 1e-6*f[-1])
        plt.title('before filtering')
        plt.xlabel('f (MHz)')
        
        plt.subplot(222)
        plt.plot(1e-6*filtered_f, np.real(filtered_spec_cut))
        plt.plot(1e-6*filtered_f, np.imag(filtered_spec_cut))
        plt.xlim(1e-6*filtered_f[0], 1e-6*filtered_f[-1])
        plt.title('after filtering')
        plt.xlabel('f (MHz)')
        
        plt.subplot(223)
        plt.plot(np.real(signal))
        plt.plot(np.imag(signal))
        plt.xlabel('number of points')
        
        plt.subplot(224)  # TODO time axis
        plt.plot(np.real(signal_filtered))
        plt.plot(np.imag(signal_filtered))
        plt.xlabel('number of points')
    
    return signal_filtered


def demodulate_fft_shift(signal, freq_mod, tres=1e-9, plot=False):
    """
    Demodulate signal in the frequency domain by shift its fft and ifft it back
    in the time domain.
    
    Parameters
    ----------
    signal: ndarray
        1D-array of complex numbers with signal to phase
    freq_mod: float
        intermediate frequency to be demodulated.
    tres: float, default 1e-9
        signal timing resolution, sampling period (s)
    plot: boolean, default False
        allows to plot signal pre- and post-demodulation

    Returns
    -------
    signal_dem : ndarray
        1D-array of complex numbers with demodulated signal
    """

    # signal information
    sweep_width = 1 /tres

    # switching to frequency domain
    # points added in the FFT to improve precision of the shift
    signal_f = np.fft.fftshift(np.fft.fft(signal,n=8*len(signal)))

    # distance between each point in frequency domain
    d = sweep_width / len(signal_f)

    # nco shift in number of points
    shift = -int(freq_mod / d)

    # shifting signal in frequency domain
    signal_f_shifted = np.roll(signal_f, shift)

    # demodulated signal from ifft (and zero-padding deleted)
    signal_dem = np.fft.ifft(np.fft.ifftshift(signal_f_shifted))[0:len(signal)]

    if plot:

        plt.subplots(2, 2, figsize=(15, 7.5 / 1.61))

        plt.subplot(221)
        plt.plot(np.linspace(-500, 500, len(signal_f)), np.real(signal_f), label = 'Re(S(f))')
        plt.plot(np.linspace(-500, 500, len(signal_f)), np.real(signal_f_shifted), label = 'Re(S_shifted(f))')
        plt.xlabel('f (MHz)')
        plt.ylabel('Real')
        plt.legend()
        
        plt.subplot(222)
        plt.plot(np.real(signal), label = 'Re(S(t))')
        plt.plot(np.real(signal_dem), label = 'Re(S_shifted(t))')
        plt.xlabel('points')
        plt.legend()
        
        plt.subplot(223)
        plt.plot(np.linspace(-500, 500, len(signal_f)), np.imag(signal_f), label = 'Im(S(f))')
        plt.plot(np.linspace(-500, 500, len(signal_f)), np.imag(signal_f_shifted), label = 'Im(S_shifted(f))')
        plt.xlabel('f (MHz)')
        plt.ylabel('Imag.')
        plt.legend()
        
        plt.subplot(224)
        plt.plot(np.imag(signal), label = 'Im(S(t))')
        plt.plot(np.imag(signal_dem), label = 'Im(S_shifted(t))')
        plt.xlabel('points')
        plt.legend()

    return signal_dem


def demodulate_cos_sin(signal, freq_mod, tres=1e-9, freq_cutoff=None,
                       filter_properties:typing.Union[str,dict]='auto'):
    """
    Demodulate signal in time domain.
    
    Parameters
    ----------
    signal : ndarray
        1D-array of complex numbers with signal to demodulate
    freq_mod : float
        modulation frequency to be demodulated (Hz)
    tres: float, default 1e-9
        signal temporal resolution (sampling period) (s)
    freq_cutoff: float, default None
        cutoff frequency for signal filtering. it should be <2*freq_mod to cut
        the higher frequency component generated by the demodulation.
    filter_properties: str or dict, default 'auto'
        arguments to be used with scipy.signal.filtfilt function
    Returns
    -------
    signal_dem : ndarray
        1D-array of complex numbers with demodulated signal
    
    Notes
    -----
    
    Signal multiplied by cos and sine components oscillating at the modulation
    frequency omega_mod. Since signal is proportional to sinusoidal oscillation
    at omega_mod, we have:
    cos(omega_mod*t) cos(omega_mod*t) = 1/2 * cos((omega_mod - omega_mod)*t)
                                        + 1/2 * cos((omega_mod + omega_mod)*t)
                                      = 1/2 + 1/2*cos(2*omega_mod*t)
    It works better than demodulate_fft_shift, in particular for sharp pulse
    shapes. Howver, it requires a less flexible filter, which has to be
    somewhere around freq_mod to delete the additional higher frequency 
    component. It can therefore be unpractical for echoes.
    """
    # 1. Define helper variables
    if freq_cutoff is None:
        freq_cutoff = freq_mod
    f_digital_cutoff = freq_cutoff / nyquist_frequency(tres)
    t = np.arange(0, len(signal), 1)

    # 2. "Analog demodulation:" 
    cos_signal = np.cos(2*np.pi*freq_mod*t) * signal
    sin_signal = np.sin(2*np.pi*freq_mod*t) * signal
    
    # 3. Filter out the higher-frequency component
    if filter_properties is None:
        kwargs = {'N': 3, 'Wn': f_digital_cutoff, 'btype': 'lowpass',
                      'analog': False}
        filter_name = 'butter'
    else:
        kwargs = copy.deepcopy(filter_properties)
        kwargs['btype'] = 'lowpass'
        filter_name = kwargs['filter_name']
        del kwargs['filter_name']

    b, a = lowpass_filter(filter_name, **kwargs)  # filter parameters
    
    # filtfilt is called in time domain!
    dem_I = scipy.signal.filtfilt(b, a, cos_signal)
    dem_Q = scipy.signal.filtfilt(b, a, sin_signal)
    
    signal_dem = dem_I + 1j * dem_Q

    return signal_dem


def nyquist_frequency(tres):
    """
    Returns the Nyquist frequency associated with a signal temporal resolution
    (sampling period).

    """
    f_sampling = 1.0/tres

    return f_sampling/2.0


def lowpass_filter(filter_name:str, **kwargs):
    """
    Get b, a for a low-pass filter
    
    Parameters
    ----------
    filter_name: str
        name of the filter from scipy.signal.filtfilt
    
    Returns
    -------
    b, a
        low-pass filter parameters for scipy.signal.filtfilt
    
    Notes
    -----
    User has to know the correct input arguments
    """
    filter_name_is_in_signal = filter_name in list(scipy.signal.__dict__.keys())
    
    try:
        filter_outputs_ba = inspect.signature(getattr(scipy.signal, filter_name)).parameters['output'].default == 'ba'
    except KeyError:
        filter_outputs_ba = False 
    
    if (not filter_name_is_in_signal) or (not filter_outputs_ba): 
        raise ValueError("Invalid filter_name. filter_name must correspond to"
                         "a function in scipy.signal that outputs b, a")

    b, a = getattr(scipy.signal, filter_name)(**kwargs)

    return b, a

