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


def lookup(df,rows,columns):
    assert isinstance(df,pd.DataFrame)
    col_index = df.columns.get_indexer(columns)
    row_index = df.index.get_indexer(rows)
    return df.values[row_index,col_index]