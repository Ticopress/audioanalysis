"""
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
"""

from scipy import signal
from scikits.audiolab import Sndfile
import numpy as np
import logging, sys, time

class AudioAnalyzer():
    """AudioAnalyzer docstring goes here TODO
    
    """

    
    def __init__(self):
        """Constructor docstrings goes here TODO
        """
        self.logger = logging.getLogger('AudioAnalyzer.logger')
        
        #Spectrogram parameters
        self.window = ('hamming')
        self.fft_width = 256
        self.nfft = 512
        self.time_step_ms = 2
        self.detrend = 'constant'
        self.onesided = True
        self.scaling = 'density'
        self.process_chunk = 15
        
        #Information about all songs, and about the currently active one
        self.songs = []
        
        self.active_song = None
        self.Sxx = None
        self.time = None
        self.freq = None
        self.tmesh = None
        self.fmesh = None
        self.domain = None
        self.freq_range = None
        self.marker = None
        self.selection = None
        
        self.entropy = None
        self.amplitude = None
        self.classification = None
        
        
        
    
    def set_active(self, idx):
        """Select a songfile from the current list of files and designate one
        as the active songfile
        """
        
        if self.active_song:
            self.update_songfile(self.active_song, self.time, self.classification)
        
        self.active_song = self.songs[idx]
        self.process(self.active_song)
    
    def process(self, sf):
        """Take a songfile and using its data, create the processed statistics
        
        
        """
        noverlap = self.fft_width - sf.Fs/1000 * self.time_step_ms
        noverlap = noverlap if noverlap > 0 else 0
        
        if sf.data.shape[0] > self.process_chunk * sf.Fs:
            split_count = int(np.ceil(sf.data.shape[0]/(self.process_chunk*sf.Fs)))
            nchunk = self.process_chunk*sf.Fs
        else:
            split_count = 1
            nchunk = sf.data.shape[0]
         
        for i in range(0, split_count):
            self.logger.info('Processing songfile part %d of %d', i+1, split_count)
            (self.freq, self.time_part, self.Sxx_part) = signal.spectrogram(
                    sf.data[i*nchunk:(i+1)*nchunk], 
                    fs=sf.Fs,
                    window=self.window, 
                    nperseg=self.fft_width, 
                    noverlap=noverlap, 
                    detrend=self.detrend, 
                    return_onesided=self.onesided, 
                    scaling=self.scaling, 
                    nfft=self.nfft
                    )
            if i == 0:
                self.time = self.time_part
                self.Sxx = self.Sxx_part
            else:
                self.time = np.append(self.time, self.time[-1]+self.time_part)
                self.Sxx = np.hstack((self.Sxx, self.Sxx_part))
            
        
        self.Sxx = 10*np.log10(self.Sxx)
        self.logger.debug('Size of one STFT: %d bytes', sys.getsizeof(self.Sxx))
        self.logger.debug('STFT dimensions %s', str(self.Sxx.shape))
        
        self.domain = (min(self.time), max(self.time))
        self.freq_range = (min(self.freq), max(self.freq))
        self.marker = min(self.time)
        self.selection = None
                        
        self.entropy = np.zeros(self.time.size)
        self.amplitude = np.zeros(self.time.size)
        self.classification = np.zeros(self.time.size)
        
    def update_songfile(self, sf, new_t=None, new_class=None):
        """A setter for the attributes of a songfile that AudioAnalyzer will
        need to be able to write to.  May go away later"""
        if new_t is not None:
            sf.time = new_t
        if new_class is not None:
            sf.classification = new_class
 
    def class_integer_to_vectorized(self, int_classification):
        """DEPRECATED.  Exists in np_utils, I think
        
        Convert the integer class values to a one-hot vector encoding
        
        The class NeuralNetwork requires a one-hot encoding for output state.
        That is, if there are three states (0,1,2) possible, it expects that
        these be represented as three 3x1 ndarrays [1 0 0], [0 1 0], and 
        [0 0 1].  It will also accept fuzzy representations to indicate
        uncertainty of state, for instance [0.9 0 0.43] is also a valid
        state encoding to a NeuralNetwork indicating a high probability of
        state 0, with small possibility of state 2. 
        
        This method, however, converts an array of integer state labels to the
        respective one-hot encoding and returns that ndarray
        """
        num_categories = np.max(int_classification) + 1
        
        nn_vectors = np.zeros((num_categories, int_classification.size))
        
        for idx, i in enumerate(int_classification):
            nn_vectors[i, idx] = 1
            
        return nn_vectors
    
    def class_vectorized_to_integer(self, nn_vector_classifications, **kwargs):
        """Convert a classification ndarray created by a NeuralNetwork to a
        set of integer classification labels
        
        This method is not yet implemented, and will require advanced
        processing techniques.  The NN is not required to return a clear
        classification, and in fact is expected to make small mistakes and
        to return 'spiky' data.  This method will parse the spiky data and
        smooth it out, but that will require knowledge of the context of 
        each data point, which the NN does not use.
        
        It may be enough to simply use the weighted average of the classes of
        neighboring points - however, that may be insufficient. 
        
        Inputs:
            nn_vector_classifications: the ndarray returned by a trained NN
                attempting to classify a spectrogram
        Keyword Arguments
            window_type: a string describing the windowing function to be used.
                Defaults to 'hamming'
            window_size: a string describing the width of window to be used.
                Defaults to 40
            beta: an argument for window window_type 'kaiser', defaults to 14, with a
                valid range of 0<beta<infinity
            sigma: an argument for window window_type 'gaussian', defaults to 0.4,
                with a valid range 0<sigma<0.5
        """
        
        window_type = kwargs.get('window_type', 'hamming')
        N = kwargs.get('window_size', 40)
        
        length = nn_vector_classifications.shape[1]
        new_classification = np.zeros(length)

        new_kwargs = {'beta':kwargs.get('beta', 14), 'sigma':kwargs.get('sigma', 0.4)}

        for idx in range(length):
            side = ''
            left = idx - N/2
            right = idx + N/2
            if left < 0:
                left = 0
                side = 'right'
            if right > length-1:
                right = length-1
                side = 'left'
            
            subset = nn_vector_classifications[:, left:right:1]
            windowed = self.apply_window(subset, window_type, side=side, 
                    **new_kwargs)
            windowed_average = windowed.mean(1)
            new_classification[idx] = windowed_average.argmax()
        
        return new_classification
    
    def class_to_motifs(self):
        """Take a vector of integer classifications and determine start and
        end times for motifs.
        """
        pass
    
    def apply_window(self, data, window_type, **kwargs):
        """Takes a window from a set of standard windows and applies it to a
        classification (in integer or vectorized format).
        
        Inputs:
            data: a numpy ndarray, either 1d or 2d
            window: a string representing the window_type of window.  Must be one of
                'gaussian', 'blackman', 'hanning', 'hamming', or 'kaiser'
        Keyword Arguments
            beta: a parameter for kaiser windows
            sigma: a parameter for gaussian windows
            side: a string determining if the window is one-sided or not
        """
        
        if data.ndim == 1:
            N = data.size
        else:
            N = data.shape[1]

        side = kwargs.get('side', '')
        if side:
            N = 2*N
            
        n = np.array(range(N))
        coeffs = np.zeros(N)
        coeffs[(N-1)/2] = 1
        if window_type == 'gaussian':
            sigma = kwargs.get('sigma', 0.25)
            coeffs = np.exp(-0.5 * ((n - 0.5*(N-1))/(sigma * 0.5*(N-1)))**2)
        if window_type == 'blackman':
            coeffs = np.blackman(N)
        if window_type == 'hamming':
            coeffs = np.hamming(N)
        if window_type == 'hanning':
            coeffs = np.hanning(N)
        if window_type == 'kaiser':
            beta = kwargs.get('beta', 14)
            coeffs = np.kaiser(N, beta)
        
        if side=='left':
            coeffs = coeffs[:N/2]
        if side=='right':
            coeffs = coeffs[N/2:]
                
        return np.multiply(coeffs, data)

class SongFile:
    """Class for storing data related to each song
    
    Critical values like entropy, STFT, and amplitude are not held in this class
    because they are memory expensive.  Only one set of critical values will
    be stored in memory at a time, and that will be for the active song as
    determined by the AudioAnalyzer class.

    Instead, this stores the basic song data: Fs, analog signal data"""
    
    def __init__(self, data, Fs):
        self.logger = logging.getLogger('SongFile.logger')
        
        self.data = data
        self.Fs = Fs
        
        self.time = None
        self.classification = None
        
        self.id = None
        
        
    @classmethod
    def load(cls, filename, split=300, downsampling=None):
        """Loads a file, splitting it into multiple SongFiles if necessary
        
        Inputs: 
            filename: a .WAV file path in filename
            split: a length, in seconds, at which the audio file should be split.
                Defaults to 600 seconds, or 10 minutes, if not specified
            downsampling: the integer ratio by which the song should be sampled
        
        Returns an array of SongFiles"""
        
        f = Sndfile(filename, mode='r')
        data = f.read_frames(f.nframes)
        fs = float(f.samplerate)
        
        if data.ndim is not 1:
            data = data[:, 0]
        
        if downsampling:
            fs = fs/downsampling
            data = data[::downsampling]  
                
        if split:
            nperfile = split * fs
            split_count = int(np.ceil(data.shape[0]/nperfile))
        else:
            nperfile = data.shape[0]
            split_count = 1
            if nperfile / fs > 300:
                logging.getLogger('SongFile.Loading.logger').warning(
                        'Current song is %d seconds long and is not'
                        ' split.  This may cause substantial memory use and '
                        'lead to a program crash.  It is recommended to either '
                        'enable splitting or use a shorter file for NN training'
                        , nperfile/fs)
            
        sfs = []
        
        for i in range(0, split_count):
            next_sf = cls(data[i*nperfile:(i+1)*nperfile], fs)
            
            next_sf.fname = filename
            next_sf.start = int(i*nperfile/fs)
            next_sf.end = int((i+1)*nperfile/fs)
            
            
            sfs.append(next_sf)
            
        return sfs
    
    def __str__(self):
        return '%s.%04d_%04d'.format(self.fname, self.start, self.end)
    
    def __repr__(self):
        return str(self)
    
    def export(self):
        pass
    
    def load_export(self):
        pass