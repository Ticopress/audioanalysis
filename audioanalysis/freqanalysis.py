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
from scipy.io import wavfile 
import numpy as np
from functools import partial

class AudioAnalyzer():
    """AudioAnalyzer docstring goes here TODO
    
    """

    
    def __init__(self):
        """Constructor docstrings goes here TODO
        """
        
        #Spectrogram parameters
        self.window = ('hamming')
        self.fft_width = 512
        self.overlap = 256
        self.detrend = 'constant'
        self.onesided = True
        self.scaling = 'density'
        #Spectrogram inputs
        self.data = None
        self.Fs = None
        #Spectrogram outputs
        self.t = None
        self.f = None
        self.Sxx = None
        #Maximum time domain
        self.domain = None
        
        self.selection = None
        #Location (index) of the audio marker for playback
        self.marker = None
        #Meshes for plotting
        self.tmat = None
        self.fmat = None
        #Critical statistics
        self.entropy = None
        self.amplitude = None
        #NeuralNet for classification
        self.neural_net = None
        #Integer array of class assigned to each point, whether by human or NN
        self.classification = None
        self.class_labels = []
        #ndarray of NN output, fuzzy one-hot encoded class predicted for each
        #time point
        self.nn_output = None
    
    def set_data(self, data, Fs):
        """Take an ndarray representing an audio signal and process it.
        
        This setter executes the body of the preprocessing of a signal.  It
        generates the STFT/spectrogram of the data, including both the time and
        frequency vectors and meshes.  It generates the entropy and amplitude
        vectors, and any other statistics of merit to be implemented in the
        future. It resets the marker location, the current selection, and a
        default classification of the data.
        """
        self.data = data
        self.Fs = Fs

        (self.f, self.t, self.Sxx) = signal.spectrogram(self.data, fs=self.Fs,
                window=self.window, nperseg=self.fft_width, 
                noverlap=self.overlap, detrend=self.detrend, 
                return_onesided=self.onesided, scaling=self.scaling)
        
        self.domain = (min(self.t), max(self.t))
        self.freq_range = (min(self.f), max(self.f))
        self.marker = min(self.t)
        self.selection = None
        
        self.tmat, self.fmat = np.meshgrid(self.t, self.f)
        
        self.entropy = np.zeros(self.t.size)
        self.amplitude = np.zeros(self.t.size)
        self.classification = np.zeros(self.t.size)
    
    def import_wavfile(self, filename, downsampling=None):
        """Import a .wav file with downsampling from a file and process it"""
        fs, data = wavfile.read(filename)
        
        if downsampling is not None:
            fs = fs/downsampling
            data = data[::downsampling]
            
        self.set_data(data,fs)
 
    def classification_to_NN_vectorized(self):
        """Convert the integer class values to a one-hot vector encoding
        
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
        num_classes = np.max(self.classification)
        return np.zeros((num_classes, self.classification.size))
    
    def NN_vectorized_to_classification(self, nn_vector_classifications):
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
        """
        pass
    
    def classification_to_motifs(self):
        """Take a vector of integer classifications and 
        """
        pass
    
class NeuralNetwork:
    """NN docstring goes here TODO
    
    
    """
    
    
    def __init__(self, sample_input, **kwargs):
        """Create an ANN based from a template
        
        This method uses a sample input array, the number of categories,
        and the size of each hidden layer to construct a set of zero-
        mean randomized weight arrays for each node of the neural
        network.
        
        The general forward-prop format is Node_next = Weights_last * node_last
        Note the left-multiplication of the weights matrix means that each
        weights matrix will have dimensions (next_height x previous_height)
        
        Keywork Arguments:
            'init_mode': method used for initializing weight arrays. Defaults
                to 'random', which is mean-zero random values
            'num_categories': number of output categories. Default 2
            'hidden_layer_sizes': Nx1 ndarray where each value is the size of
                the output of that layer.  Default yields a single-layer NN
        """
        
        #Parse kwargs
        init_mode = kwargs.get('init_mode', 'random')
        self.n_output = kwargs.get('num_categories', 2)
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', np.zeros(0))
        self.prop_function = kwargs.get('prop_func', 
                partial(self.sigmoid, deriv=False))
        self.deriv_function = kwargs.get('deriv_func', 
                partial(self.sigmoid, deriv=True))
        
        try:   
            assert (sample_input.size is not 0)
            
            self.input_size = sample_input.size
            
            assert self.n_output > 0
            
            
            assert all(hidden_layer_sizes > 0)
            size_list = [self.input_size]
            size_list.extend(hidden_layer_sizes)
            size_list.append(self.n_output)
            
            #Initialize the list of weight matrices by iterating pairwise
            #over the sizes of successive layers
            self.weights = []
            np.random.seed(1)
            for (index, size) in enumerate(size_list[:-1]): 
                current, next_ = size, size_list[index + 1]
                self.weights.append(self.init_weights(current,
                        next_, init_mode))

        except (AssertionError, AttributeError):
            print "Invalid inputs were provided for ANN initialization"
    
    def init_weights(self, size_layer_in, size_layer_out, mode):
        """Generate an initial weights matrix based on input and output size
        
        Allows for future variability in weights matrix initialization, but
        for the moment all matrices are initialized as zero-mean random weights
        """
        
        if mode=='random':
            return 2*np.random.random((size_layer_out, size_layer_in)) - 1   
        
    def train(self, data_in, classes_in, n_chunk, n_train_iter):
        """Take a set of inputs and the classification vector for each input
        and train the ANN
        
        
        """
        
        for _ in range(n_train_iter):
            #Break the data into 'small' chunks and process each chunk
            for j in range(data_in.shape[1]/n_chunk):
                data = data_in[:,j*n_chunk:(j+1)*n_chunk]
                estimated_classes = self.forward_propagate(data)
                #Back propagate
                correct_classes = classes_in[:,j*n_chunk:(j+1)*n_chunk]
                self.back_propagate(estimated_classes, correct_classes)
            
            #process the last few data points
            if (j+1) * n_chunk < data_in.shape[1]:
                data = data_in[:, (j+1)*n_chunk:]
                estimated_classes = self.forward_propagate(data)
                #Back propagate these few points as well
                correct_classes = classes_in[:, (j+1)*n_chunk:]
                self.back_propagate(estimated_classes, correct_classes)
                pass
            
            
        
    
    
    
    def classify(self, data_in, n_chunk):
        """Take a set of inputs and return a classification vector for each
        input.
        
        The input should be an ndarray where the columns represent each input
        sample.  The output is likewise an ndarray with each column being a
        classifcation vector.  The size of the classification vector is
        determined by self.n_output, and each element of the vector will
        represent the square root of the probability that the sample belongs
        to that class, normalized so that the sum of the squares of the
        vector elements is one.
        
        n_chunk is an integer representing how many of the input samples
        should be classified at a time.  This affects speed and memory usage,
        but not results.
        """
        
        return np.zeros(self.n_output)

    def forward_propagate(self, data_chunk):
        """Calculate the anticipated classification for a vector of inputs
        
        Note: hidden inputs are hidden, meaning they are never stored"""
        nodes = []
        nodes.append(data_chunk)
        for idx, w in enumerate(self.weights):
            nodes.append(self.prop_function(np.dot(w, nodes[idx])))
            
        return nodes
    
    def sigmoid(self,x,deriv=False):
        """Take an ndarray and calculate the sigmoid activation function for 
        each value
        
        This is the default propagation function, and is used unless another
        function is specified in the kwargs of the class __init__
        """
        y = 1/(1+np.exp(-x))

        if(deriv):
            return y*(1-y)
        
        return y
            
        
def main():
    net = NeuralNetwork(np.zeros(2), num_categories=3, 
            hidden_layer_sizes=np.array([]));
    random_data = np.random.random((2,5))
    print "My random data is:",random_data
    print "The fprop results are is:",net.forward_propagate(random_data)

if __name__ == '__main__':
    main()        
        