'''
Created on Nov 16, 2015

@author: justinpalpant

Copyright 2015 Justin Palpant

This file is part of the Jarvis Lab Audio Analysis program.

Audio Analysis is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Audio Analysis is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Audio Analysis. If not, see http://www.gnu.org/licenses/.
'''

from freqanalysis import AudioAnalyzer
from PyQt4.uic import loadUiType

from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT)

from PyQt4 import QtGui

Ui_MainWindow, QMainWindow = loadUiType('main.ui')

class AudioGUI(Ui_MainWindow, QMainWindow):
    """AudioGUI docstring goes here TODO
    
    
    """
    

    def __init__(self,):
        """Initialization docstring goes here TODO
        
        
        """
        #Initialization of GUI from Qt Designer
        super(AudioGUI, self).__init__()
        self.setupUi(self)

        # Initialize the basic plot area
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.plot_vl.addWidget(self.canvas)
        self.canvas.draw()
        
        self.toolbar = SpectrogramNavBar(self.canvas, self.plot_container)
        self.plot_vl.addWidget(self.toolbar)
        
        self.analyzer = AudioAnalyzer()
        
        #Set up button callbacks
        self.open_file.clicked.connect(self.file_open_dialog)
        
        #Initialize the collection of assorted parameters
        #Not currently customizable, maybe will make interface later
        self.params = {'downsampling':1, 'time_downsample':2, 
                       'freq_downsample':1, 'display_threshold':-400}
        
        self.redraw()
    
    def file_open_dialog(self):
        """Provide a standard file open dialog to import .wav data into the 
        model classes"""        
        file_name = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 
                '/home', 'WAV files (*.wav)')
        
        if file_name is not None:
            self.file_name.setText(file_name)
            
        self.analyzer.import_wavfile(file_name, self.params['downsampling'])
        
        self.show_data()
        
    def redraw(self):
        """Basic refresh operation for all plots"""
        self.canvas.draw()
        self.show()
        
    def show_data(self):
        """Put all applicable data from the Model to the View
        
        This function assumes all preprocessing has been completed and merely
        looks at the state of the model and displays it
        """
        if self.analyzer.Sxx is not None:
            #Put the spectrogram data on a graph  
            if not self.toolbar.axis_dict:
                print self.toolbar.axis_dict
                self.toolbar.add_axis('spectrogram') 
              
            print self.toolbar.axis_dict
            
            t_step = self.params['time_downsample']
            f_step = self.params['freq_downsample']
            
            self.toolbar.axis_dict['spectrogram'].pcolormesh(
                    self.analyzer.tmesh[::t_step,::f_step], 
                    self.analyzer.fmesh[::t_step,::f_step],
                    self.analyzer.Sxx[::t_step,::f_step], cmap='gray_r')
            
            #cbar = ColorbarBase(self.toolbar.axis_dict['spectrogram'], cmap='gray_r')
            #cbar.set_clim(self.params['display_threshold'], 0)
              
            self.toolbar.set_domain_limits(self.analyzer.domain)        
            self.toolbar.set_domain(self.analyzer.domain)
            self.toolbar.set_range('spectrogram', self.analyzer.freq_range)
            
            self.redraw()
            
            #Add a cursor
            pass
            
            
            
            #Add the selection box
            pass
        
        
class SpectrogramNavBar(NavigationToolbar2QT):
    """SpectrogramNavBar docstring goes here TODO
    
    
    """
    
    
    def __init__(self, canvas_, parent_, *args, **kwargs):  
        """Initialization docstring goes here TODO
        
        
        """ 
        
        #A single, unified set of x-boundaries, never violable
        self.x_constraint = ()
        #Dictionary mapping str->axis object
        self.axis_dict = {}
        #Ordered list of axis names, in order added
        self.axis_names = []
            
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Scroll left', 'back', 'back'),
            ('Forward', 'Scroll right', 'forward', 'forward'),
            (None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            (None, None, None, None),
            #('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        
        NavigationToolbar2QT.__init__(self, canvas_, parent_, *args, **kwargs)
    
    '''Consider redefining the init_toolbar method to let me customize the 
    handling of self.toolitems more
    
    def __init_toolbar__(self):
        pass
    '''  
    def add_axis(self, name, init_x=(), init_y=()):
        if not self.axis_dict:
            self.axis_dict[name] = self.canvas.figure.add_subplot(111)
            self.axis_names.append(name)
            if init_x:
                self.set_domain(init_x)
            if init_y:
                self.set_range(name, init_y)
        else:
            self.axis_dict[name] = self.axis_dict[self.axis_names[0]].twinx()
            self.axis_names.append(name)
            if init_y:
                self.set_range(name, init_y)

    def set_domain_limits(self, x_constrain):
        self.x_constraint = x_constrain
        
    def set_domain(self, x_domain):
        if self.x_constraint:
            try:
                assert x_domain[0] > self.x_constraint[0] 
                assert x_domain[1] < self.x_constraint[1]
            except AssertionError:
                print "Assert failed: the sdomain command is out of bounds"
                return
            
        for ax in self.axis_dict.values():
            yrange = ax.get_ylim()
            new_bounds = (x_domain[0], x_domain[1], yrange[0], yrange[1])
            ax.axis(new_bounds)
    
    def set_range(self, name, yrange):
        ax = self.axis_dict[name]
        xbounds = self.get_domain()
        new_bounds = (xbounds[0], xbounds[1], yrange[0], yrange[1])
        ax.axis(new_bounds)
        
    
    def set_all_ranges(self, yranges):
        try:
            for idx, yrange in enumerate(yranges):
                self.set_range(self.axis_names[idx], yrange)
        except IndexError:
            print ('Too many yranges provided, there are only', 
                    len(self.axis_names)+1, 'axes available')
            
            

    def forward(self, *args):
        try:
            axes = self.axis_dict.values()
            ax = axes[0]
        except IndexError:
            print "Scrolling irrelevant, no plots at this time"
        else:
            xbounds = ax.get_xlim()
            width = xbounds[1] - xbounds[0]
            if xbounds[1]+0.25*width < self.x_constraint[1]:
                dx = 0.25*width
            else:
                dx = self.x_constraint[1] - xbounds[1]
                
            new_bounds = (xbounds[0]+dx, xbounds[1]+dx)
            
            self.set_domain(new_bounds)
     
    def back(self, *args):
        try:
            axes = self.axis_dict.values()
            ax = axes[0]
        except IndexError:
            print "Scrolling irrelevant, no plots at this time"
        else:
            xbounds = ax.get_xlim()
            width = xbounds[1] - xbounds[0]
            if xbounds[0]-0.25*width > self.x_constraint[0]:
                dx = 0.25*width
            else:
                dx = xbounds[0] - self.x_constraint[0]
                
            new_bounds = (xbounds[0]-dx, xbounds[1]-dx)
            
            self.set_domain(new_bounds)
    
    
    """
    def select(self, *args):
        pass
    
    def press_select(self, *args):
        pass
    
    def release_select(self, *args):
        pass
    
    def drag_select(self, *args):
        pass
    
    def release_zoom(self, *args):
       pass
    """
    def drag_pan(self, event):
        """Overwrites the original NavigationToolbar2 drag_pan callback
        
        Adjusted to make sure limits are maintained
        """
        
        for a, _ in self._xypress:
            #safer to use the recorded button at the press than current button:
            #multiple buttons can get pressed during motion...
            pre_drag = a.get_xlim()
            a.drag_pan(self._button_pressed, event.key, event.x, event.y)
            post_drag = a.get_xlim()
            
            correct_xlims = post_drag if self.validate(post_drag) else pre_drag
            
            self.set_domain(correct_xlims)
            
        self.dynamic_update()
    
    def validate(self, xlims):
        if not self.x_constraint:
            return True
        else:
            return xlims[0] > self.x_constraint[0] and xlims[1] < self.x_constraint[1]
        
           
def main():
    import sys

    app = QtGui.QApplication(sys.argv)
    main = AudioGUI()

    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
