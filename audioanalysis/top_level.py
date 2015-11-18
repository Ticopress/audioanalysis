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

import freqanalysis
import numpy as np
from PyQt4.uic import loadUiType
 
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.pyplot import specgram

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

    def redraw(self):
        # self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.show()

    def yzoom_out(self):
        plot_limits = self.axes.axis()
        yl = plot_limits[2]
        yh = plot_limits[3]
        dy = yh - yl
        yl -= 0.1 * dy
        yh += 0.1 * dy
        
        new_plot_limits = (plot_limits[0], plot_limits[1], yl, yh)
        
        self.axes.axis(new_plot_limits)
        self.redraw()
        
    def yzoom_in(self):
        plot_limits = self.axes.axis()
        yl = plot_limits[2]
        yh = plot_limits[3]
        dy = yh - yl
        yl += 0.1 * dy
        yh -= 0.1 * dy
        
        new_plot_limits = (plot_limits[0], plot_limits[1], yl, yh)
        
        self.axes.axis(new_plot_limits)
        self.redraw()
        
    def xzoom_out(self):
        plot_limits = self.axes.axis()
        xl = plot_limits[0]
        xh = plot_limits[1]
        dx = xh - xl
        xl -= 0.1 * dx
        xh += 0.1 * dx
        
        new_plot_limits = (xl, xh, plot_limits[2], plot_limits[3])
        
        self.axes.axis(new_plot_limits)
        self.redraw()
        
    def xzoom_in(self):
        plot_limits = self.axes.axis()
        xl = plot_limits[0]
        xh = plot_limits[1]
        dx = xh - xl
        xl += 0.1 * dx
        xh -= 0.1 * dx
        
        new_plot_limits = (xl, xh, plot_limits[2], plot_limits[3])
        
        self.axes.axis(new_plot_limits)
        self.redraw()
         
    def yslide(self):
        print self.spectr_vsb.sliderPosition()
      
    def xslide(self):
        print self.spectr_hsb.sliderPosition()  
        
    def set_data(self, data, Fs):
        data, freqs, t, im = specgram(data, NFFT=256, Fs=Fs, noverlap=128)
        self.axes.imshow(data, cmap='bone', extent=[-1,1,-1,1])
        #self.redraw()
        
        

def main():
    import sys
    from PyQt4 import QtGui
    
    data = np.random.rand(10000)
    print data
    Fs = 500

    app = QtGui.QApplication(sys.argv)
    main = AudioGUI()
    main.set_data(data, Fs)

    main.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
