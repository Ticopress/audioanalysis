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
import numpy as np
from functools import partial
from scikits.audiolab import Sndfile
import time
from PyQt4.Qt import left

class AudioAnalyzer():
    """AudioAnalyzer docstring goes here TODO
    
    """

    
    def __init__(self):
        """Constructor docstrings goes here TODO
        """
        
        #Spectrogram parameters
        self.window = ('hamming')
        self.fft_width = 512
        self.nfft = 512
        self.time_step_ms = 2
        self.detrend = 'constant'
        self.onesided = True
        self.scaling = 'density'
        #Spectrogram inputs
        self.data = None
        self.Fs = None
        self.chunked = None
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
        self.tmesh = None
        self.fmesh = None
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
        
        This setter executes the bulk of the preprocessing of a signal.  It
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
                noverlap=(self.fft_width - self.Fs/1000 * self.time_step_ms), 
                detrend=self.detrend, return_onesided=self.onesided, 
                scaling=self.scaling, nfft=self.nfft)
        
        self.Sxx = 20*np.log10(self.Sxx)
        
        self.domain = (min(self.t), max(self.t))
        self.freq_range = (min(self.f), max(self.f))
        self.marker = min(self.t)
        self.selection = None
        
        self.tmesh, self.fmesh = np.meshgrid(self.t, self.f)
                
        self.entropy = np.zeros(self.t.size)
        self.amplitude = np.zeros(self.t.size)
        self.classification = np.zeros(self.t.size)
    
    def import_wavfile(self, filename, downsampling=None):
        """Import a .wav file with downsampling from a file and process it"""

        f = Sndfile(str(filename), mode='r')
        frames_to_read = f.nframes - (f.nframes % self.fft_width)
        data = f.read_frames(frames_to_read)
        fs = f.samplerate
        
        if data.ndim is not 1:
            data = data[:, 0]
        
        if downsampling is not None:
            fs = fs/downsampling
            data = data[::downsampling]    
        
        self.set_data(data, fs)
 
    def classification_to_NN_vectorized(self, int_classification):
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
        num_categories = np.max(int_classification) + 1
        
        nn_vectors = np.zeros((num_categories, int_classification.size))
        
        for idx, i in enumerate(int_classification):
            nn_vectors[i, idx] = 1
            
        return nn_vectors
    
    def NN_vectorized_to_classification(self, nn_vector_classifications, **kwargs):
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
            beta: an argument for window type 'kaiser', defaults to 14, with a
                valid range of 0<beta<infinity
            sigma: an argument for window type 'gaussian', defaults to 0.4,
                with a valid range 0<sigma<0.5
        """
        
        type = kwargs.get('window_type', 'hamming')
        N = kwargs.get('window_size', 40)

        length = nn_vector_classifications.shape[1]
        new_classification = np.zeros(length)

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
            windowed = self.apply_window(subset, type, side=side, **kwargs)
            windowed_average = np.mean(windowed, 1)
            new_classification[idx] = np.argmax(windowed_average)
        
        return new_classification
    
    def classification_to_motifs(self):
        """Take a vector of integer classifications and determine start and
        end times for motifs.
        """
        pass
    
    def apply_window(self, data, type, **kwargs):
        """Takes a window from a set of standard windows and applies it to a
        classification (in integer or vectorized format).
        
        Inputs:
            data: a numpy ndarray, either 1d or 2d
            window: a string representing the type of window.  Must be one of
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
        if type == 'gaussian':
            sigma = kwargs.get('sigma', 0.25)
            coeffs = np.exp(-0.5 * ((n - 0.5*(N-1))/(sigma * 0.5*(N-1)))**2)
        if type == 'blackman':
            coeffs = np.blackman(N)
        if type == 'hamming':
            coeffs = np.hamming(N)
        if type == 'hanning':
            coeffs = np.hanning(N)
        if type == 'kaiser':
            beta = kwargs.get('beta', 14)
            coeffs = np.kaiser(N, beta)
        
        
        if side=='left':
            coeffs = coeffs[:N/2]
        if side=='right':
            coeffs = coeffs[N/2:]
        
        print coeffs, data
        
        return np.multiply(coeffs, data)
        
        
        
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
        
        Keyword Arguments:
            'init_mode': method used for initializing weight arrays. Defaults
                to 'random', which is mean-zero random values
            'num_categories': number of output categories. Default 2, a binary
                classification system
            'hidden_layer_sizes': Nx1 ndarray where each value is the size of
                the output of that layer. Default, [], yields a single-layer NN
            'prop_func': the nonlinear activation function used in propagation.
                Must be Numpy ndarray compliant, defaults to a sigmoid function
            'deriv_func': the derivative of the activation function. Must be
                Numpy ndarray compliant
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
    
    def classify(self, data_in, n_chunk):
        """Take a set of inputs and return a classification vector for each
        input.
        
        The input should be an ndarray where the columns represent each input
        sample.  The output is likewise an ndarray with each column being a
        classification vector.  The size of the classification vector is
        determined by self.n_output, and each element of the vector will
        represent the square root of the probability that the sample belongs
        to that class, normalized so that the sum of the squares of the
        vector elements is one.
        
        n_chunk is an integer representing how many of the input samples
        should be classified at a time.  This affects speed and memory usage,
        but not results.
        """
        
        #Pre-allocate for small speed gains and for slicing
        estimated_classes = np.zeros((self.n_output, data_in.shape[1]))
        
        #Break the data into 'small' chunks and process each chunk
        for j in range(data_in.shape[1]/n_chunk):
            data = data_in[:,j*n_chunk:(j+1)*n_chunk]
            estimated_classes[:,j*n_chunk:(j+1)*n_chunk] = self.forward_propagate(data)

        #process the last few data points
        if (j+1) * n_chunk < data_in.shape[1]:
            data = data_in[:, (j+1)*n_chunk:]
            estimated_classes[:,j*n_chunk:(j+1)*n_chunk] = self.forward_propagate(data)

        return estimated_classes

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
    analyzer = AudioAnalyzer()

    nn_classification = np.random.random((5,100))
    #print nn_classification
    new = analyzer.NN_vectorized_to_classification(nn_classification, 
            window_size=10, window_type='gaussian', sigma=0.3)

    print new

    #analyzer.import_wavfile('/Users/new/Downloads/157536__juskiddink__woodland-birdsong-june.wav', 4)

    net = NeuralNetwork(np.zeros(2), num_categories=3, 
            hidden_layer_sizes=np.array([]));
    random_data = np.random.random((2,5))
    #print "My random data is:",random_data
    #print "The fprop results are is:",net.forward_propagate(random_data)

if __name__ == '__main__':
    main()        
        