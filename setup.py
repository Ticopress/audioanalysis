'''
Created on Dec 13, 2015

@author: justinpalpant
This file is part of the Jarvis Lab Audio Analysis program.

Audio Analysis is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Audio Analysis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Audio Analysis. If not, see http://www.gnu.org/licenses/.

To build this module using pyinstaller, use something like:

pyinstaller --onefile --distpath=./executables/mac_osx --name=audioanalysis
        --paths=./audioanalysis {audioanalysis/__main__.py
        
Called from the project folder (same level as setup.py)
    
--distpath puts the executable somewhere that is NOT /dist
--paths makes sure to include all modules inside audioanalysis, like the critical
    ones, which does not happen by default
--name makes sure the output name isn't __main__

Make a pip installable package by calling 
    python setup.py bdist_wheel
For the python module to be called from the path, make sure that entry_points has
    'console_scripts': ['audioanalysis=audioanalysis.__main__:main']
For the package to be importable, make sure packages=['audioanalysis'] and that 
    install_requires=[list of package names, as in PyPI]
In order to be able to run python -m audioanalysis, all Qt packages must be
    pre-installed before calling pip install
'''
from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='audioanalysis',
      version='0.1.0.dev1',
      description='Jarvis Lab Audio Analyzer',
      author='Justin Palpant',
      author_email='justin@palpant.us',
      url='https://github.com/jpalpant/jarvis-lab-audio-analysis',
      license='GPLv3',
      packages=['audioanalysis'],
      install_requires=[],
      long_description=read('README.txt'),
      entry_points = {
        'console_scripts': ['audioanalsys=audioanalysis.__main__:run_as_executable'],
      },
      )