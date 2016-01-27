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
import sys, os
import logging

from scipy import signal
import scipy.io.wavfile
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import keras.layers.core as corelayers
import keras.layers.convolutional as convlayers
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

class AudioAnalyzer():
    """AudioAnalyzer docstring goes here TODO
    
    """
    logger = logging.getLogger('AudioAnalyzer.logger')
    
    def __init__(self, **params):
        """Create an AudioAnalyzer
        
        All keyword arguments are gathered and stored in the instance's 
        self.params.  Keyword arguments relate to STFT parameters, 
        """        
        #List of loaded songs
        self.songs = []
        self.motifs = []
        #Reference to and spectrogram of currently active song
        self.active_song = None
        self.Sxx = None
        
        self.params = params
        
        #Reference to the neural net used for processing
        self.classifier = None
    
    def build_neural_net(self):
        """Construct and compile a Keras neural net
        
        Keyword Arguments:
            layers: a list of layerspecs, as defined in make_layer
            loss: a string specifying a Keras loss function.  Defaults to 
                'categorical_crossentropy'
            optimizer: a string specifying a Keras optimizer.  Defaults to 'sgd'
        """
        self.logger.info('Constructing parameterized neural network')
        nn = Sequential()
        
        layers = self.params.get('layers', [])
        img_rows = self.params.get('img_rows', self.Sxx.shape[0])
        img_cols = self.params.get('img_cols', 1)
                
        for i, layerspec in enumerate(layers):
            if i==0: #size the input layer correctly
                layerspec['kwargs']['input_shape'] = (1, img_rows, img_cols)

            l = self.make_layer(layerspec)
            nn.add(l)
            
            self.logger.debug('Layer input: %s', str(l.input_shape))
            self.logger.debug('Layer output: %s', str(l.output_shape))
            
        self.logger.info('Building the output layer for %d classes', 
                self.active_song.num_classes)
        
        l = self.make_layer({'type':'Dense', 'args':(self.active_song.num_classes,)})   
        nn.add(l)
        
        l = self.make_layer({'type':'Activation', 'args':('softmax',)})
        nn.add(l)
        
        self.logger.info('Compiling deep neural network')
        loss = self.params.get('loss', 'categorical_crossentropy')
        optimizer = self.params.get('optimizer', 'sgd')
        
        nn.compile(loss=loss, optimizer=optimizer)
        self.logger.info('Successfully constructed a new neural net')
        
        return nn
            
    def make_layer(self, layerspec):
        """Build a layer from a dictionary specifying the layer parameters
        
        The layerspec should contain the following entries:
            type: a class name
            args: a tuple of arguments for the class's __init__
            kwargs: a dictionary of kwargs for the class's __init__
        """
        
        name = layerspec.get('type')
        self.logger.info('Building layer specified by %s', str(layerspec))

        try:
            cls = getattr(corelayers, name)
        except AttributeError:
            try:
                cls = getattr(convlayers, name)
            except AttributeError as e:
                self.logger.error('Unreadable layerspec provided, cannot build neural net')
                raise e
            
        args = layerspec.get('args', ())
        kwargs = layerspec.get('kwargs', {})
        
        l = cls(*args, **kwargs)
        
        return l
    
    def load_neural_net(self, folder):
        """Load a neural net from json and h5 files exported with export_neural_net"""
        
        self.logger.info('Loading neural net model')
        model = model_from_json(open(os.path.join(folder,'nn_model.json')).read())
        self.logger.info('Loading neural net weights')
        model.load_weights(os.path.join(folder,'nn_weights.h5'))
        self.logger.info('Done loading neural net')

        return model
    
    def export_neural_net(self, folder):
        """Export the analyzer's neural net to the given folder
        
        Creates two files, one a json string describing the model and one an
        HDF5 file storing the model's weights
        """
        
        with open(os.path.join(folder, 'nn_model.json'), 'w') as outfile:
            outfile.write(self.classifier.to_json()) 
            
        self.classifier.save_weights(os.path.join(folder, 'nn_weights.h5'))
    
    def train_neural_net(self):
        """Using the currently active song, train_neural_net the neural net"""
        nb_epoch = self.params.get('epochs', 1)
        batch_size = self.params.get('batch_size', 16)
        nb_classes = self.active_song.num_classes
        validation_split = self.params.get('validation_split', 0.25)
        
        
        indices = np.arange(0,self.active_song.time.size-50)
        (X_train, Y_train) = self.get_data_sample(indices)

        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        
        self.logger.info('X_train shape %s', str(X_train.shape))
        self.logger.info('Y_train shape: %s', str(Y_train.shape))
        
        self.logger.info('X_train max %s', str(np.amax(X_train)))
        self.logger.info('X_train min %s', str(np.amin(X_train)))
    
        X_train, X_test, Y_train, Y_test = train_test_split(
                X_train, Y_train,
                test_size=validation_split,
                random_state=np.random.randint(0,100000,1)[0]
                )
   
        self.logger.info('Begin training process: ')
    
        self.classifier.fit(
                X_train, Y_train, 
                batch_size=batch_size, 
                nb_epoch=nb_epoch, 
                show_accuracy=True, 
                verbose=1, 
                validation_data=(X_test, Y_test)
                )


    def get_data_sample(self, idx):
        """Get a sample from the spectrogram and the corresponding class
        
        Inputs:
            idx: an ndarray of integer indices
        
        Returns:
            (data, classification): a tuple where data is a 
                (1,1,img_rows,img_cols) slice from Sxx and classification is the 
                corresponding integer class from self.active_song.classification
                
        If idx exceeds the dimensions of the data, throws IndexError
        If there is not a processed, active song, throws TypeError
        """
        
        img_rows = self.params.get('img_rows', self.Sxx.shape[0])
        img_cols = self.params.get('img_cols', 1)
        
        if self.Sxx is None or self.active_song.classification is None:
            raise TypeError('No active song from which to get data')
        
        if np.amax(idx) > self.Sxx.shape[1]-img_cols:
            raise IndexError('Data index of sample out of bounds, only {0} '
                    'samples in the dataset'.format(self.Sxx.shape[1]-img_cols))
        
        if np.amin(idx) < 0:
            raise IndexError('Data index of sample out of bounds, '
                    'negative index requested')
            
        #index out the data   
        classification = self.active_song.classification[idx]

        data_slices = [self.Sxx[0:img_rows, idx+i].T.reshape(idx.size, 1, img_rows) for i in range(img_cols)]
        data = np.stack(data_slices, axis=-1)

        #scale the input
        data = np.log10(data)
        data -= np.amin(data)
        data /= np.amax(data)
        
        return (data, classification)

    def set_active(self, sf):
        """Select a SongFile from the current list and designate one as the 
        active SongFile
        """
        self.active_song = sf
        
        try:
            self.Sxx = self.active_song.Sxx
        except AttributeError:
            self.Sxx = self.process(self.active_song)
    
    def process(self, sf):
        """Take a songfile and using its data, create the processed statistics
        
        This method both updates the data stored in the SongFile (for those
        values that are stored there) and RETURNS the calculated spectrogram of
        the SongFile.  You must catch the returned value and save it, it is not
        written to self.Sxx by default
        """
        time_window_ms = self.params.get('fft_time_window_ms', 10)
        time_step_ms = self.params.get('fft_time_step_ms', 2)
        nfft = self.params.get('nfft', 512)
        process_chunk = self.params.get('process_chunk_s', 15)
        
        noverlap = (time_window_ms - time_step_ms)* sf.Fs/1000
        noverlap = noverlap if noverlap > 0 else 0
        
        if sf.data.shape[0] > process_chunk * sf.Fs:
            split_count = int(np.ceil(sf.data.shape[0]/(process_chunk*sf.Fs)))
            nchunk = process_chunk*sf.Fs
        else:
            split_count = 1
            nchunk = sf.data.shape[0]
            
        nperseg = time_window_ms * sf.Fs / 1000
        
        try:
            min_freq = self.params['min_freq']
            self.logger.info('Highpass filter %g Hz applied', min_freq)
            data = AudioAnalyzer.butter_highpass_filter(sf.data, min_freq, sf.Fs, 5)
        except KeyError:
            data = sf.data
            self.logger.info('No highpass filter applied')
        
        if nfft < nperseg:
            nfft = 2**np.ceil(np.log2(nperseg))
            self.logger.warning('NFFT (%d) cannot be less than the number of '
                    'samples in each time_list window (%d).  Temporarily increasing '
                    'nfft to %d, which will require more memory.  To avoid this,'
                    ' decrease FFT Time Window in the parameters menu.', 
                    nfft, nperseg, 2**np.ceil(np.log2(nperseg)))
         

        for i in range(0, split_count):
            self.logger.info('Processing songfile from %d seconds to %d seconds', 
                    i*process_chunk, (i+1)*process_chunk)
            
            (freq, time_part, Sxx_part) = signal.spectrogram(
                    data[i*nchunk:(i+1)*nchunk], 
                    fs=sf.Fs,
                    nfft=nfft,  #number of bins; must be 2^z
                    nperseg=nperseg, #width in time domain
                    noverlap=noverlap, #overlap in time domain
                    detrend='constant', 
                    return_onesided=True, 
                    scaling='density', 
                    window=('hamming'), 
                    )
            
            if i == 0:
                time_list = time_part
                Sxx = Sxx_part
            else:
                time_list = np.append(time_list, time_list[-1]+time_part)
                Sxx = np.hstack((Sxx, Sxx_part))
           
        self.logger.debug('Size of one STFT: %d bytes', sys.getsizeof(Sxx))
        self.logger.debug('STFT dimensions %s', str(Sxx.shape))               

        if sf.classification is None:
            sf.classification = np.zeros(time_list.size) 
        
        sf.time = time_list
        sf.freq = freq[0:nfft/2]
        sf.entropy = self.calc_entropy(Sxx)
        sf.power = self.calc_power(Sxx)
        
        return Sxx[0:nfft/2, :]
    
    
    @staticmethod
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    @staticmethod
    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = AudioAnalyzer.butter_highpass(cutoff, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y
    
    @staticmethod
    def calc_entropy(Sxx):
        """Calculates the Wiener entropy (0 to 1) for each time slice of Sxx"""
        n = Sxx.shape[0]
        return np.exp(np.sum(np.log(Sxx),0)/n) / (np.sum(Sxx, 0)/n)
    
    @staticmethod
    def calc_power(Sxx):
        """Calculates average signal power"""
        return np.sum(Sxx, 0) / Sxx.shape[0]
    
    def classify_active(self):
        """Creates a classification for the active song using classifier
        
        """
        input = -np.log10(self.Sxx[:, :]).T
        input /= np.max(input)
        input = input.reshape(input.shape[0], 1, input.shape[1], 1)
        prbs = self.classifier.predict_proba(input, batch_size=1000, verbose=1)

        #prbs is an samples_x_classes array of likelihoods
        #Need to smooth it, window it, and convert to categorie

class SongFile(object):
    """Class for storing data related to each song
    
    Critical values like entropy, STFT, and amplitude are not held in this class
    because they are memory expensive.  Only one set of critical values will
    be stored in memory at a time, and that will be for the active song as
    determined by the AudioAnalyzer class.

    Instead, this stores the basic song data: Fs, analog signal data"""
    logger = logging.getLogger('SongFile.logger')
    
    def __init__(self, data, Fs, name='', start=0):
        """Create a SongFile for storing signal data
        
        Inputs:
            data: a numpy array with time series data.  For use with PyAudio,
                ensure the format of data is the same as the player
            Fs: sampling frequency, ideally a float
        Keyword Arguments:
            name: a string identifying where this SongFile came from
            start: a value in seconds indicating that the SongFile's data does
                not come from the start of a longer signal
        """
        
        #Values passed into the init
        self.data = data
        self.Fs = Fs
        
        #Post-processed values (does not include spectrogram)
        self.time = None
        self.freq = None
        self.classification = None
        self.entropy = None
        self.power = None
        
        
        self.name = name
        self.start = start
        
        self.length = len(self.data)/self.Fs
     
    @property
    def domain(self):
        return (min(self.time), max(self.time))

        
    @property    
    def range(self):
        return (min(self.freq), max(self.freq))
        
    @property
    def num_classes(self):
        return len(np.unique(self.classification))
        
    @classmethod
    def load(cls, filename, split=600, downsampling=None):
        """Loads a file, splitting it into multiple SongFiles if necessary
        
        Inputs: 
            filename: a .WAV file path in filename
            split: a length, in seconds, at which the audio file should be split.
                Defaults to 300 seconds, or 5 minutes, if not specified
            downsampling: the integer ratio by which the song should be sampled
        
        Returns an array of SongFiles"""
        
        rate, data = scipy.io.wavfile.read(filename)
        fs = np.float64(rate)
        data = np.float32(data) / np.max(data)
                
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
            
        if nperfile / fs > 600:
            logging.getLogger('SongFile.Loading.logger').warning(
                    'Current song is %d seconds long and is not'
                    ' split.  This may cause substantial memory use and '
                    'lead to a program crash.  It is recommended to either '
                    'enable splitting or use a shorter file for NN training'
                    , nperfile/fs)
            
        sfs = []
        
        for i in range(0, split_count):
            songdata = data[i*nperfile:(i+1)*nperfile]
            fname = os.path.splitext(os.path.basename(filename))[0]
            next_sf = cls(songdata, fs, name=fname, start=int(i*nperfile/fs))
                        
            sfs.append(next_sf)
            
        return sfs
    
    @classmethod
    def find_motifs(cls, sf, **params):
        """Cut motifs from a classified songfile and build songfiles from them
        
        This method takes a SongFile, assumes it has already been correctly
        classified and therefore has a classification that is not None, it scans
        through that classification and determines (with some resistance to 
        noise) the regions where there appears to be a motif.
        
        Note: motifs are indicated anywhere the classification is nonzero.
        """
        
        min_dur=params.get('min_dur',0)
        max_dur=params.get('max_dur', float('inf'))
        smooth_gap=params.get('smooth_gap', 0)
            
        motifs = []
        
        try:
            times = sf.time[np.nonzero(sf.classification)]
        except TypeError:
            sf.logger.info('Song %s does not have a classification, cannot '
                    'find motifs', sf.name)
            return []
        
        in_motif = False
        
        for i, t in enumerate(times):
            if not in_motif:
                start_time = t
                in_motif = True
                sf.logger.debug('Motif for %s start at %0.4f', sf.name, start_time)
            
            if in_motif and (i==len(times)-1 or times[i+1]-t > smooth_gap):
                data = sf.data[sf.time_to_idx(start_time):sf.time_to_idx(t)]
                #name and Fs are the same
                new_motif = SongFile(data, sf.Fs, name=sf.name, start=start_time)
                
                motifs.append(new_motif)
                in_motif = False
                sf.logger.debug('Motif for %s end at %0.4f', sf.name, t)

        #Check that lengths satisfy the requirements
        return [m for m in motifs if min_dur<=m.length<=max_dur]
    
    def time_to_idx(self, t):
        return int(t * self.Fs)
    
    def __str__(self):
        return '{:s}_{:03d}_{:03d}'.format(
                self.name, int(self.start), int(self.length+self.start))

    def export(self, destination, filename=None):
        """Exports data in WAV format
        
        Not useful for SongFiles you just loaded, but possibly quite useful for
        generated SongFiles.
        """
        
        if filename is None:
            filename = str(self) + '.wav'
        elif os.path.splitext(filename)[1] is not '.wav':
            filename = filename + '.wav'
        
        fullpath = os.path.join(destination, filename)
        
        scipy.io.wavfile.write(fullpath, int(self.Fs), self.data)
        
    def pickle(self, destination, filename=None):
        if filename is None:
            filename = str(self) + '.pkl'
        elif os.path.splitext(filename)[1] is not '.pkl':
            filename = filename + '.pkl'
        
        fullpath = os.path.join(destination, filename)
        self.logger.info('Pickling to %s', fullpath)
        with open(fullpath, 'wb') as picklefile:
            pickle.dump(self, picklefile)
            
        self.logger.info('Done pickling!')
        
    @classmethod
    def unpickle(cls, filename):
        with open(filename, 'rb') as picklefile:
            sf = pickle.load(picklefile)
            
        try:
            assert isinstance(sf, cls)
        except AssertionError:
            logging.getLogger('SongFile.Unpickler').error('Cannot unpickle the '
            'file %s, not a valid instance of SongFile', filename)
        else:
            return sf
    
       
        