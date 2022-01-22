from collections import OrderedDict
import tqdm
import itertools
from sklearn.model_selection import KFold
import sklearn.model_selection
import pandas as pd


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

    best_model = clf_type(**results.sort_values("score").iloc[0].drop("score").to_dict())
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
            
        
