import numpy as np

def combine_key(to_join, sep = "|"):
    '''
    # TODO: check that sep is not present in the keys
    '''
    assert isinstance(to_join, list)
    
    res = np.array(to_join[0]).astype(str)
    for col in to_join[1:]:
        t = np.char.add(res, sep)
        res = np.char.add(t, np.array(col).astype(str))
    return res