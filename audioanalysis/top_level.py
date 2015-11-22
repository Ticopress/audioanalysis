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

from freqanalysis import AudioAnalyzer
import numpy as np
from PyQt4.uic import loadUiType

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT)

from functools import partial 

from PyQt4 import QtGui

Ui_MainWindow, QMainWindow = loadUiType('main_toolbar.ui')

class AudioGUI(Ui_MainWindow, QMainWindow):

    def __init__(self,):
        #Initialization of GUI from Qt Designer
        super(AudioGUI, self).__init__()
        self.setupUi(self)

        # Initialize the basic plot area
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.plot_vl.addWidget(self.canvas)
        self.canvas.draw()
        
        self.toolbar = SpectrogramNavBar(self.canvas, self.plot_container, coordinates=False)
        self.plot_vl.addWidget(self.toolbar)
        
        self.analyzer = AudioAnalyzer()
        
        self.set_axis((0, 10, 0, 10))


    def redraw(self):
        # self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.show()

        
    def set_axis(self, limits):
        self.axes.axis(limits)
        self.current_view = limits
        
        self.redraw()
        
    def set_data(self, data, Fs):
        self.analyzer.set_data(data, Fs)
        self.set_axis(self.analyzer.max_window)
        self.toolbar.axis_limits = self.analyzer.max_window
        
    def show_data(self):
        if self.analyzer.Sxx is not None:
            #Put the spectrogram data on a graph
            self.axes.pcolormesh(self.analyzer.tmat, self.analyzer.fmat, self.analyzer.Sxx, cmap=plt.get_cmap('gray_r'))
            
            #Set the axis scale
            self.set_axis(self.current_view)
            
            #Add a cursor
            pass
            
            
            
            #Add the selection box
            pass
            
            
        else:
            #show a blank plot, but with correct bounds
            self.axes.axis(self.current_view)

        
        
class SpectrogramNavBar(NavigationToolbar2QT):
    
    def __init__(self, canvas_, parent_, *args, **kwargs):   
        self.axis_limits = None
        
        #only display buttons we need
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to  previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            (None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            (None, None, None, None),
            #('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        
        NavigationToolbar2QT.__init__(self, canvas_, parent_, *args, **kwargs)
    
    '''Consider redefining the init_toolbar method to let me customize the toolbar more
    def __init_toolbar__(self):
        pass
    '''
    
    #define the action of the forward button
    def forward(self, *args):
        pass
    
    #define the action of the back button
    def back(self, *args):
        pass
    
    def select(self, *args):
        #model after pan and zoom functions in backend_bases
        pass
    
    #model the following functions after press_pan, release_pan, and drag_pan functions in backend_bases
    def press_select(self, *args):
        pass
    
    def release_select(self, *args):
        pass
    
    def drag_select(self, *args):
        pass
    
    #Overwrite the release zoom function to allow for left-click zoom in, right-click zoom out
    #def release_zoom(self, *args):
    #   pass
    
    #Overwrite the drag_pan function to account for axis bounds
    def drag_pan(self, *args):
        pass
    
    def out_of_bounds(self, axes_to_check):
        if self.axis_limits is None:
            return False
        
            
            
    def constrain_to_bounds(self, axes_to_constrain):
        pass
    

def main():
    import sys

    
    fs = 44100
    N = 4e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    freq = np.linspace(1e3, 4e3, N)
    x = amp * np.sin(2*np.pi*freq*time)
    x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

    app = QtGui.QApplication(sys.argv)
    main = AudioGUI()
    #main.set_data(x, fs)
    #main.show_data()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
