import importlib
import gutil


def reload():
    '''
    seems to work so far, have to reload first all submodules and then the main module
    can be used as from the main script gutil = gutil.reload()
    '''
    importlib.reload(gutil.basic)
    importlib.reload(gutil.pandas)
    importlib.reload(gutil.ml)
    importlib.reload(gutil)
    return gutil


def mydir(x):
    res = dir(x)
    return [x for x in res if not x.startswith("_")]
