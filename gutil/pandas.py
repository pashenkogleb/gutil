import pandas as pd
import tqdm
import numpy as np

def apply(s, func):
    '''
    apply with progress tracker
    '''
    res =[]
    for x in tqdm.tqdm(s):
        res.append(func(x))
    return pd.Series(res,index= s.index)


def lookup(df,rows,columns):
    assert isinstance(df,pd.DataFrame)
    col_index = df.columns.get_indexer(columns)
    row_index = df.index.get_indexer(rows)
    out =  df.values[row_index,col_index]
    mask = (row_index==-1) | (col_index==-1)
    if np.sum(mask)>0: # has incorrect indices
        out = out.astype(np.float64)
        out[mask] =np.nan
    return out