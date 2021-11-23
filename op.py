import numpy as np



def from_codes(indices, categories, res_dtype = "float"):
    '''
    similar to pd.Categorical.from_codes but supports multidimensional array
    res_type can be either float or object
    '''
    assert res_dtype in ['float','object']
    
    fake_indices = indices.copy()
    invalid_mask = fake_indices==-1
    fake_indices[invalid_mask] = 0 # I will change this to nan later


    res =categories[fake_indices].astype(res_dtype)
    res[invalid_mask] = np.nan
    return res