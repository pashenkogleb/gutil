from collections import OrderedDict
import tqdm
import itertools
from sklearn.model_selection import KFold
import sklearn.model_selection
import pandas as pd
import numpy as np
import lightgbm 
from lightgbm import LGBMClassifier
import sklearn


class LgbClf(LGBMClassifier):
    '''
    suports early stopping internally
    cv should support split method. Only uses first split.
    uses KFold(n_splits=5, shuffle=True) by default
    '''
    def __init__(self,cv=None,stopping_rounds=5, **kwargs):
        if cv is None:
            cv=KFold(n_splits=5, shuffle=True, random_state=1)
            
        assert int(lightgbm.__version__.split(".")[0])>=4, "otherwise lgb uses different syntax"
        super().__init__(**kwargs)
        self.cv =cv
        self.stopping_rounds=stopping_rounds


    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # Remove or replace cv here to avoid serialization errors
        params.pop('cv', None)
        params.pop('stopping_rounds', None)
        return params

    def set_params(self, **params):
        if 'cv' in params:
            self.cv = params.pop('cv')
        if "stopping_rounds" in params:
            self.stopping_rounds= params.pop("stopping_rounds")
        super().set_params(**params)
        return self
    
    def fit(self, X,y, **kwargs):
        tr_ind, val_ind = next(self.cv.split(X, y))
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(lightgbm.early_stopping(self.stopping_rounds))

        Xtr = self.__index(X, tr_ind)
        ytr = self.__index(y, tr_ind)
        Xval = self.__index(X, val_ind)
        yval = self.__index(y, val_ind)
        # return (Xtr,ytr,Xval, yval)
        
        super().fit(Xtr, ytr,
            eval_set=[(Xval,yval )],
            callbacks=callbacks,
            eval_metric='logloss',
            **kwargs
        )
        return self

    @staticmethod
    def __index(X, ind):
        if isinstance(X,np.ndarray):
            return X[ind]
        elif isinstance(X, pd.DataFrame) or isinstance(X,pd.Series):
            return X.iloc[ind]
        else:
            raise ValueError(f"unknown type: {type(X)}")


def cv_optimize(clf_type, params,  X, Y, num_splits = 5, return_search_space = False):
    '''
    Finds optimal model among clf_type by using exhaustive search with cross validation
    params is a dictionary of lists
    for example:
    clf_type = Ridge
    params = {"alpha": [0,0.5,1,2,3,5,10], "l1_ratio": [0,1/3,0.5,2/3,1]}
    '''
    params = {x:y for x,y in zip(params.keys(), [y if isinstance(y,list) else [y] for y in params.values()])} #convert all to lists for ease

    all_combos = list(dict(zip(params, x)) for x in itertools.product(*params.values()))
    print(f"trying out {len(all_combos)} models")
    results = []
    split = list(KFold(num_splits).split(X))
    for params in tqdm.tqdm(all_combos, desc = "trained models.."):
        model  = clf_type(**params)
        score = sklearn.model_selection.cross_validate(model, X, Y, verbose = 0, error_score = "raise", cv = split )['test_score'].mean()
        d = params.copy()
        d['score'] = score
        results.append(d)

    results = pd.DataFrame(results)


    opt_params = results.sort_values("score").iloc[0].drop("score").to_dict()
    print("optimal params:", opt_params)
    best_model = clf_type(**opt_params)
    if return_search_space:
        return best_model, results
    else:
        return best_model

class Pipeline:
    '''
    initial idea was to add caching, but maybe it makes more sense to just inherit this on estimator level
    '''
    def __init__(self,estimators):
        self.estimators = OrderedDict()
        for key, estimator in estimators:
            assert key not in self.estimators
            self.estimators[key] = estimator
    def fit(self,X, y=None):
        out = X
        for est in tqdm.tqdm(self.estimators.values(), total = len(self.estimators), desc = "fitting.."):
            out = est.fit(out,y=y)
    def transform(self,X):
        out = X
        for est in tqdm.tqdm(self.estimators.values(), total = len(self.estimators), desc = "transforming.."):
            out = est.transform(out)
        return out

class CachedTransformer:
    '''
    stores last output, so does not have to recompute, if input is same as last
     USES MEMORY ADDRESS, so if object has same address would throw same results
    '''
    def __init__(self, transformer):
        self.transformer = transformer
        self.last_fit = None
        self.last_transform = None #tuple with input and output
    def fit(self,X,y=None):
        if self.last_fit == (X,y):
            print("fit already cached")
            return
        else:
            self.last_fit = (X,y)
            self.transformer.fit(X,y)
    def transform(self,X):
        if self.last_transform[0] == X:
            print("returning stored transform")
            return self._transform[1]
        else:
            out = self.transformer.transform(X)
            self.last_transform = (X,out)
            return out
            
        
