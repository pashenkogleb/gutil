from sktime.transformations.series.difference import Differencer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def plot_seasonal(series, apply_differencer = False):
    '''
    takes a series as an input
    '''
    if apply_differencer:
        series = Differencer(na_handling="keep_na").fit_transform(series)

    base = pd.Timestamp(2000,1,1)
    uniform = []
    for year, s in series.groupby(series.index.year):
        uniform_s = pd.Series(s.values, base + (s.index -pd.Timestamp(s.index[0].year, 1,1)))
        plt.plot(uniform_s,label= year, alpha=0.1)
        uniform.append(uniform_s)
    uniform = pd.concat(uniform,axis=1)
    uniform.mean(axis=1).plot()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    return uniform