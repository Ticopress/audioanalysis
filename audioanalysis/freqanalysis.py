'''
Created on Nov 16, 2015

@author: justinpalpant

Copyright 2015 Justin Palpant

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
from scipy import signal
import numpy as np

class AudioAnalyzer():
    '''
    The Model class for the Audio Analysis program
    This class stores all of the audio information, is responsible for data processing, and handles playback
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.window = ('tukey', 0.25)
        self.nperseg = 256
        self.overlap = None
        self.NFFT = None
        self.detrend = 'constant'
        self.onesided = True
        self.scaling = 'density'
        self.data = None
        self.Fs = None
        self.t = None
        self.f = None
        self.Sxx = None
        self.audio_key = None
        self.max_window = None
        self.selection = None
        self.marker = None
        self.tmat = None
        self.fmat = None
    
    def set_data(self, data, Fs):
        self.data = data
        self.Fs = Fs

        (self.f, self.t, self.Sxx) = signal.spectrogram(self.data, self.Fs, self.window,
                                                        self.nperseg, self.overlap, self.NFFT, self.detrend, self.onesided,
                                                        self.scaling)
        
        self.max_window = (min(self.t), max(self.t), min(self.f), max(self.f))
        
        self.marker = min(self.t)
        self.selection = None
        
        self.tmat, self.fmat = np.meshgrid(self.t, self.f)
    
    def import_wavfile(self, filename, downsampling=None):
        pass
        
        
        
        
        