from setuptools import setup, find_packages



setup(
   name='gutil',
   version='1.0',
   description='All my useful functions',
   author='Gleb Pashchenko',
   author_email='pashenkogleb@gmail.com',
   packages=find_packages(),  #same as name
   install_requires=["lightgbm>=4.0.0", "scikit-learn>=0.24.0", "pandas>=1.2.0", "numpy>=1.19.0", "tqdm>=4.62.0", "selenium", "sympy", "IPython", "requests" , "beautifulsoup4", "matplotlib", "sktime", "numba"], #external packages as dependencies

)