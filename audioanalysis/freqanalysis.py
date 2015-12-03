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
from functools import partial
from scikits.audiolab import Sndfile
import time
import numpy as np


    

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
        
        self.Sxx = np.log10(self.Sxx)
        
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
 
    def class_integer_to_vectorized(self, int_classification):
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
        
        
        
class NeuralNetwork:
    """NN docstring goes here TODO
    
    
    """
    
    
    def __init__(self, sample_input, **kwargs):
        """Create an ANN based from a template
        
        This method uses a sample input array, the number of categories,
        and the size of each hidden layer to construct a set of weight arrays 
        for each node of the neural network.
        
        The general forward-prop format is Node_next = Weights_last * node_last
        Note the left-multiplication of the weights matrix means that each
        weights matrix will have dimensions (next_height x previous_height)
        
        Keyword Arguments:
            'init_mode': method used for initializing weight arrays. Defaults
                to 'random', which is mean-zero random values
            'num_categories': number of output categories. Default 1
            'hidden_layer_sizes': Nx1 ndarray where each value is the size of
                the output of that layer. Default, [], yields a single-layer NN
            'prop_func': the nonlinear activation function used in propagation.
                Must be Numpy ndarray compliant, defaults to a sigmoid function.
                Must have a kwarg deriv=T/F such that it outputs its own
                derivative if derif=True
            'learning_rate': a value on (0, 1) the affects the rate of training.
                Defaults to 0.5
        """
        
        #Parse kwargs
        init_mode = kwargs.get('init_mode', 'random')
        self.n_output = kwargs.get('num_categories', 1)
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', np.zeros(0))
        self.prop_function = kwargs.get('prop_func', self.sigmoid)

        self.rate = kwargs.get('learning_rate', 0.8)
        self.input_function = kwargs.get('input_func', 
                partial(self.linear, slope=1))
        self.output_function = kwargs.get('output_func', 
                self.sigmoid)
        
        try:   
            assert (sample_input.size is not 0)
            
            self.input_size = sample_input.size
            
            assert self.n_output > 0
            
            
            assert all(hidden_layer_sizes > 0)
            size_list = [self.input_size]
            size_list.extend(hidden_layer_sizes)

            #Initialize the list of weight matrices by iterating pairwise
            #over the sizes of successive layers
            self.hidden_weights = []
            np.random.seed(1)
            for index, size in enumerate(size_list[:-1]): 
                print 'First weight!!'
                current, next_ = size, size_list[index + 1]
                self.hidden_weights.append(self.init_weights(current,
                        next_, init_mode))
            
            self.output_weights = self.init_weights(size_list[-1], 
                    self.n_output, init_mode)   
            
        except (AssertionError, AttributeError):
            print "Invalid inputs were provided for ANN initialization"
    
    def init_weights(self, size_layer_in, size_layer_out, mode):
        """Generate an initial weights matrix based on input and output size
        
        Allows for future variability in weights matrix initialization, but
        for the moment all matrices are initialized as zero-mean random weights
        """
        
        if mode=='random':
            return 2*np.random.random((size_layer_out, size_layer_in)) - 1   
        
    def train(self, data_in, classes_in, **kwargs):
        """Take a set of inputs and the classification vector for each input
        and train the ANN
        
        Inputs:
            data_in: an ndarray with columns representing each sample input
            classes_in: an ndarray with columns representing the correct
                function value for the corresponding column of the input
            
        Keyword Arguments:
            chunk_size: an integer representing the number of inputs that
                should be processed at once.  Defaults to the number of columns
                of data_in (the entire dataset in one pass).  A tradeoff
                between speed and memory is used here, where processing the
                entire input is fastest but requires the most memory
            iterations: the number of times that the entire data set should be
                passed through the neural net and errors should be back-
                propagated during training.  Defaults to a 1, a single time
            
        Output:
            None
        """
        try:
            assert(data_in.shape[1] == classes_in.shape[1])
        except (AssertionError):
            print 'Number of inputs and number of outputs do not match'
            return
        
        #Process kwargs
        chunk_size = kwargs.get('chunk_size', data_in.shape[1])
        iterations = kwargs.get('iterations', 1)
        
        for i in range(iterations):
            #Break the data into 'small' chunks and process each chunk
            for j in range(data_in.shape[1]/chunk_size):
                #Grab a full chunk of data if possible
                if (j+1)*chunk_size < data_in.shape[1]:
                    data = data_in[:,j*chunk_size:(j+1)*chunk_size]
                    correct_output = classes_in[:,j*chunk_size:(j+1)*chunk_size]
                #if a full chunk is too large, just finish off the data
                else:
                    data = data_in[:,j*chunk_size:]
                    correct_output = classes_in[:,j*chunk_size:]

                #Forward-prop with storage of all node stages and slopes
                node_inputs = []
                node_outputs = []
                node_slopes = []

                #Process the input node
                current_input = data
                current_output = self.input_function(current_input)
                current_slope = self.input_function(current_input, deriv=True)
                
                #print "Input node in: ",current_input
                #print "Input node out: ",current_output
                #print "Input node slope: ",current_slope
                
                node_inputs.append(current_input)
                node_outputs.append(current_output)
                node_slopes.append(current_slope)
                
                #Process hidden nodes, if any
                for w in self.hidden_weights:
                    current_input = w.dot(current_output)
                    current_output = self.prop_function(current_input)
                    current_slope = self.prop_function(current_input, deriv=True)
                    node_inputs.append(current_input)
                    node_outputs.append(current_output)
                    node_slopes.append(current_slope)

                #Process the final (output) node
                current_input = self.output_weights.dot(current_output)
                current_output = self.output_function(current_input)
                current_slope = self.output_function(current_input, deriv=True)

                node_inputs.append(current_input)
                node_outputs.append(current_output)
                node_slopes.append(current_slope)
                
                #print "Output node in: ",current_input
                #print "Output node out: ",current_output
                #print "Output node slope: ",current_slope
                
                #Back-prop of error
                node_errors = []
                node_deltas = []
                
                current_error = current_output - correct_output
                current_delta = np.multiply(current_slope, current_error)
                
                node_errors.append(current_error)
                node_deltas.append(current_delta)
                
                #print "Output node error: ", current_error
                #print "Output node delta: ", current_delta
                
                #This is the error on either the input node or the last hidden
                #node, depending on if there are any hidden nodes
                current_error = self.output_weights.T.dot(current_delta)
                
                for idx, w in enumerate(reversed(self.hidden_weights)):
                    #Access the node slopes in reverse order; skip the output
                    #node slope (idx = -1) because it was already used
                    current_node_slope = node_slopes[-2-idx]
                    current_delta = np.multiply(current_node_slope, current_error)
                    
                    node_deltas.append(current_delta)
                    node_errors.append(current_error)
                    
                    current_error = w.T.dot(current_delta)
                    
                #For the last node, the input node - don't HAVE to do anything,
                #but for completeness and demonstration, show the following
                current_delta = np.multiply(node_slopes[0], current_error)
                
                node_deltas.append(current_delta)
                node_errors.append(current_error)
                
                #Put them in the initial order you total idiot
                node_deltas.reverse()
                node_errors.reverse()
                
                weights_update = []
                
                #Loop over all node outputs except the last
                for idx, node_output in enumerate(node_outputs[:-1]):
                    #print "Output used for weights update: ",node_output
                    next_delta = node_deltas[idx+1]
                    #print "Deltas used: ", next_delta
                    update = -self.rate * next_delta.dot(node_output.T)
                    weights_update.append(update)
                
                for idx, w in enumerate(self.hidden_weights):
                    self.hidden_weights[idx] = w + weights_update[idx]
                    
                #print "updating output weights by",weights_update[-1]
                #print "current output weights",self.output_weights
                self.output_weights += weights_update[-1]
                print "updated output weights",self.output_weights

                
            if i % np.ceil(iterations / 100.0) == 0:
                print i+1,' of ',iterations,' training iterations complete'
                print 'Current error norm: ',np.linalg.norm(node_errors[-1])
                
    
    def estimate(self, data_in, **kwargs):
        """Take a set of inputs and return the estimated value of the function
        that the neural net is attempting to imitate.  
        
        Inputs:
            data_in: an ndarray with columns representing each sample input.
                Note that the ndarray MUST be columnar - i.e., even if the
                inputs are single numbers, the shape of data_in must be a 
                single row of dimension (1,N) where N is the number of inputs, 
                not a single column of shape (N,), which is default.
            
        Keyword Arguments:
            chunk_size: an integer representing the number of inputs that
                should be processed at once.  Defaults to the number of columns
                of data_in (the entire dataset in one pass).  A tradeoff
                between speed and memory is used here, where processing the
                entire input is fastest but requires the most memory
                
        Output:
            An ndarray with each column being the function estimation vector
            for the corresponding input column.  The classification vector
            can be 1x1 in the case of estimating a 1D functions y = f(x), 
            or it can be a multidimensional output for multidimensional
            functions or for classification of a piece of data into
            categories where each vector element represents the similarity
            of the data element to the element's corresponding category
        """
        #Check for keyword arguments
        chunk_size = kwargs.get('chunk_size', data_in.shape[1])
        
        #Pre-allocate for small speed gains and for slicing
        estimated_classes = np.zeros((self.n_output, data_in.shape[1]))
        
        #Break the data into 'small' chunks and process each chunk
        for j in range(data_in.shape[1]/chunk_size):
            node_input = data_in[:,j*chunk_size:(j+1)*chunk_size]
            node_output = self.input_function(node_input)
            
            for w in self.hidden_weights:
                node_input = w.dot(node_output)
                node_output = self.prop_function(node_input)
            
            final_input = self.output_weights.dot(node_output)
            final_output = self.output_function(final_input)
            estimated_classes[:,j*chunk_size:(j+1)*chunk_size] = final_output

        return estimated_classes
    
    def sigmoid(self,x,deriv=False):
        """Take an ndarray and calculate the sigmoid activation function for 
        each value
        
        This is the default propagation function, and is used unless another
        function is specified in the kwargs of the class __init__
        """
        y = np.tanh(x)

        if(deriv):
            return 1 - y**2
        
        return y
    
    def linear(self, x, slope = 1, deriv=False):
        """The default linear input and output activation function"""
        y = slope*x

        if(deriv):
            return slope*np.ones(x.shape)
        
        return y
            
        
def main():
    
    import numpy as np
    import matplotlib.cm as cm
    import pylab as pl
    np.random.seed(1337) # for reproducibility
    
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import containers
    from keras.layers.core import Dense, AutoEncoder
    from keras.optimizers import RMSprop
    from keras.utils import np_utils
    
    batch_size = 64
    nb_epoch = 1
    
    nb_classes = 10

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    i = 4600
    pl.imshow(X_train[i, 0], interpolation='nearest', cmap=cm.binary)
    print("label : ", Y_train[i,:])
    pl.show()
    
    
    
if __name__ == '__main__':
    main()        
        