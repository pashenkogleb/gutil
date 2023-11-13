import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_periods(s, period_min=2, period_max = 400):
    '''
    uses fft and plots magnitudes of each period
    for example if data has weekly seasonality would expect period =7 to have high magnitude
    period_cutoff can be used to cut noisy short periods
    returns sorted frequencies with their magnitudes
    '''
    freq_space = np.fft.fft(s)[1:] # first component is simply average
    frequencies = np.arange(1, len(s))
    periods = len(s)/frequencies
    res = pd.Series(np.abs(freq_space), periods)
    if period_min is not None:
        res = res[res.index>=period_min]
    if period_max is not None:
        res = res[res.index<=period_max]
    res.plot(marker = '*')
    plt.xlabel("period")
    plt.ylabel("magnitude")
    return res.sort_values(ascending=False)
    

def plot_frequency(s, frequency = None, period=  None, plot=True):
    '''
    can specify frequency or period
    '''
    assert (frequency is None) ^ (period is None)
    if period is not None:
        frequency = int(len(s)/period)
        print("frequency: ", len(s)/period, " using: ", frequency)
        
        
    new_freq = np.zeros(len(s), dtype = np.complex128)
    new_freq[frequency] =  np.fft.fft(s)[frequency]
    if frequency!=0:
        new_freq[len(s)-frequency] = np.fft.fft(s)[len(s)-frequency]
    res = pd.Series(np.real(np.fft.ifft(new_freq)))
    if plot:
        res.plot()
    return res