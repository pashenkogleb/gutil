import importlib
import gutil
import subprocess
import os
import IPython


def widen():
    IPython.display.display(IPython.display.HTML("<style>.container { width:100% !important; }</style>"))

def reload():
    '''
    seems to work so far, have to reload first all submodules and then the main module
    can be used as from the main script gutil = gutil.reload()
    '''
    importlib.reload(gutil.basic)
    importlib.reload(gutil.pandas)
    if hasattr(gutil, "ml"):
        importlib.reload(gutil.ml)
    importlib.reload(gutil)
    return gutil


def mydir(x):
    res = dir(x)
    return [x for x in res if not x.startswith("_")]

def shell(x):
    p = subprocess.Popen(x, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return_code = p.poll()
    assert len(stderr) == 0, stderr
    assert return_code ==0, return_code
    return stdout.decode()
