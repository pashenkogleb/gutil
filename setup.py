from setuptools import setup, find_packages

setup(
   name='gutil',
   version='1.0',
   description='All my useful functions',
   author='Gleb Pashchenko',
   author_email='pashenkogleb@gmail.com',
   packages=find_packages(),  #same as name
   install_requires=[], #external packages as dependencies
)