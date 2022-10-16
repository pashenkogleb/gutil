import pandas as pd
import tqdm

def apply(s, func):
    '''
    apply with progress tracker
    '''
    res =[]
    for x in tqdm.tqdm(s):
        res.append(func(x))
    return pd.Series(res,index= s.index)

