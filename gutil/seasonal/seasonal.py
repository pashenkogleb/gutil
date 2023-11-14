from sktime.transformations.series.difference import Differencer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import numba
import numpy as np
import scipy
import time

def plot_seasonal(series, apply_differencer = False, resample = None):
    '''
    takes a series as an input
    can provide resample = "1w" to aggregate by week if data is not available for every day
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
    m = uniform.mean(axis=1)
    if resample is not None:
        m = m.resample(resample).mean()
                    
    m.plot()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    return uniform

class MyExponential:
    def __init__(self, sps):
        '''
        sps is a list of seasonalities to consider
        model is additive
        fits exponential model that has level, trend and seasonal components
        all initial parameters + smoothers are fitted 
        can handle nans effortlessly
        '''
        assert isinstance(sps, list)
        self.sps = sps



    @staticmethod
    @numba.njit
    def __loss(initial_level, initial_trend, 
            alpha, beta, gammas, initial_seasonals,y):
        '''
        initial_seasonals is  a list of numpy arrays
        initial_seasonals are modified in place !
        '''
        #list_type = numba.types.ListType(numba.types.Array(numba.types.int64, 1, 'C'))
        assert len(gammas) == len(initial_seasonals)
        #assert isinstance(gammas,numba.types.Array)
        
        cur_level = initial_level
        cur_trend = initial_trend
        cur_seasonals = initial_seasonals
        n_seasons = len(cur_seasonals)

        loss = 0
        trends = np.empty(len(y), dtype=np.float64)
        levels = np.empty(len(y), dtype=np.float64)
        predicted = np.empty(len(y), dtype=np.float64)
        deseasoned = np.empty(len(y), dtype=np.float64)
        for i in range(len(y)):
            seasonal_adj =0
            for j in range(n_seasons):
                cs = cur_seasonals[j]
                seasonal_adj += cs[i%len(cs)]
            pred = cur_level + cur_trend + seasonal_adj
            
            error =y[i] - pred
            if np.isfinite(y[i]):
                loss+=error**2
                cur_level += cur_trend + alpha*error
                cur_trend += beta * error
                for j in range(n_seasons):
                    cs = cur_seasonals[j]
                    cs[i%len(cs)] += gammas[j] * error
            levels[i] = cur_level
            trends[i] = cur_trend
            predicted[i] = pred
            deseasoned[i] = y[i] -seasonal_adj
        rmse = np.sqrt(loss/len(y))
        return (rmse,predicted, levels, trends,deseasoned)



    
    def prepare_args(self,x, y):
        '''
        converts uniform args to  make a call to loss function
        '''
        args = list(x[:4])
        gammas = np.array(x[4:4+len(self.sps)], dtype=np.float64)
        args.append(gammas)
        
        seasonals= []
        cur_index=  4+len(self.sps)
        for s in self.sps:
            sarr = x[cur_index:cur_index+s]
            seasonals.append(np.array(sarr,dtype =np.float64))
            cur_index+=s
        slist = numba.typed.List(seasonals)
        args.append(slist)
        args.append(y)
        return args

 


    def fit(self, y, method = "L-BFGS-B",options = {"maxiter": 100000}, debug=False):
        '''
        options are passed to scipy.optimize
        '''
        
        initial_level=y[0]
        initial_trend = y[1]-y[0]
        initial_seasonal =[np.zeros(sp) for sp in self.sps]
        initial_alpha=0.1
        initial_beta=0.1
        initial_gammas = np.ones(len(self.sps))* 0.1

        flattened_seasonal = []
        for x in initial_seasonal:
            flattened_seasonal +=list(x)
        flattened_seasonal

        initial_x0 = [
            initial_level, 
            initial_trend,
            initial_alpha,initial_beta
            ] + list(initial_gammas) + \
            flattened_seasonal
            
        initial_x0 = tuple(initial_x0)
        bounds = [
            (None, None), # level
            (None,None), # trend 
            (0,1), #alpha
            (0,1), # beta
        ] + [(0,1)] * len(initial_gammas) + [(None, None)] *len(flattened_seasonal)

        bounds=tuple(bounds)
        assert len(bounds) == len(initial_x0)

        func_evals=0

        def __loss_wrapper(x):
                '''
                accepts tuple of values:
                0: l0
                1:b0
                2: \alpha
                3: \beta
                4 - 4+n_seasons: gammas
                4+n_seasons+1 : end - first k seasonals for first season,etc.
                '''
                nonlocal func_evals
                func_evals+=1
                args = self.prepare_args(x, y)
                res = self.__loss(*args)[0]
                return res
        if debug:
            return initial_x0
        
        start_time = time.time()
        opt = scipy.optimize.minimize(__loss_wrapper,initial_x0,method=method,
                bounds = bounds, options = options
                )
        end_time = time.time()
        if not opt['success']:
            print("optimization failed with message:", opt['message'])
            raise ValueError("optimization failed")

        self.fit_time_ = end_time-start_time
        self.func_evals_ = func_evals
        self.opt_flattened_ = opt['x']
        self.initial_level_ = opt['x'][0]
        self.initial_slope_ = opt['x'][1]
        self.opt_alpha_ = opt['x'][2]
        self.opt_beta_ = opt['x'][3]
        self.opt_gammas_ = opt['x'][4:4+len(self.sps)]
        self.initial_seasonalities_= list(self.prepare_args(opt['x'], y)[-2])
        loss,predicted, levels, trends,deseasoned = self.__loss(*self.prepare_args(opt['x'], y))
        self.loss_=loss
        self.df_ = pd.DataFrame({"act":y, "predicted":predicted, "resid": y-predicted, "levels":levels, "trends":trends,"deseasoned":deseasoned})
        return self


    
