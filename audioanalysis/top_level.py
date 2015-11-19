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
    FigureCanvasQTAgg as FigureCanvas)

Ui_MainWindow, QMainWindow = loadUiType('main.ui')

class AudioGUI(Ui_MainWindow, QMainWindow):

    def __init__(self,):
        #Initialization of GUI from Qt Designer
        super(AudioGUI, self).__init__()
        self.setupUi(self)
        
        # Linking buttons to functions
        self.y_zoom_in.clicked.connect(self.yzoom_in)
        self.y_zoom_out.clicked.connect(self.yzoom_out)
        self.x_zoom_in.clicked.connect(self.xzoom_in)
        self.x_zoom_out.clicked.connect(self.xzoom_out)
        self.spectr_vsb.valueChanged.connect(self.yslide)
        self.spectr_hsb.valueChanged.connect(self.xslide)

        # Initialize the basic plot area
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        # self.axes.plot(np.random.rand(5))
        self.canvas = FigureCanvas(self.fig)
        self.spectr_1.insertWidget(0, self.canvas)
        self.canvas.draw()
        
        self.analyzer = AudioAnalyzer()
        
        self.set_axis((0, 1, 0, 1))

    def redraw(self):
        # self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.show()

    def yzoom_out(self):
        yl = self.current_view[2]
        yh = self.current_view[3]
        dy = yh - yl
        yl -= 0.1 * dy
        yh += 0.1 * dy
        
        new_limits = (self.current_view[0], self.current_view[1], yl, yh)
        
        self.set_axis(new_limits) 
        
    def yzoom_in(self):
        yl = self.current_view[2]
        yh = self.current_view[3]
        dy = yh - yl
        yl += 0.1 * dy
        yh -= 0.1 * dy
        
        new_limits = (self.current_view[0], self.current_view[1], yl, yh)
        
        self.set_axis(new_limits)
        
        
    def xzoom_out(self):
        xl = self.current_view[0]
        xh = self.current_view[1]
        dx = xh - xl
        xl -= 0.1 * dx
        xh += 0.1 * dx
        
        new_limits = (xl, xh, self.current_view[2], self.current_view[3])
        
        self.set_axis(new_limits) 
               
    def xzoom_in(self):
        xl = self.current_view[0]
        xh = self.current_view[1]
        dx = xh - xl
        xl += 0.1 * dx
        xh -= 0.1 * dx
        
        new_limits = (xl, xh, self.current_view[2], self.current_view[3])
        
        self.set_axis(new_limits)
         
    def yslide(self):
        print self.spectr_vsb.sliderPosition()
      
    def xslide(self):
        print self.spectr_hsb.sliderPosition()  
        
    def set_axis(self, limits):
        self.axes.axis(limits)
        self.current_view = limits
        self.redraw()
        
    def set_data(self, data, Fs):
        self.analyzer.set_data(data, Fs)
        self.current_view = self.analyzer.max_window
        
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

        
        

def main():
    import sys
    from PyQt4 import QtGui
    
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
    main.set_data(x, fs)
    main.show_data()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
