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
'''
from setuptools import setup
import os

from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='audioanalysis',
      version='0.1.0',
      description='Jarvis Lab Audio Analyzer',
      author='Justin Palpant',
      author_email='justin@palpant.us',
      url='https://github.com/jpalpant/jarvis-lab-audio-analysis',
      license='GPLv3',
      packages=['audioanalysis'],
      install_requires=reqs,
      long_description=read('README.rst'),
      )
