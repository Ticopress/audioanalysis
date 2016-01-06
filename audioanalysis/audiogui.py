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

from freqanalysis import AudioAnalyzer, SongFile
from PyQt4.uic import loadUiType

from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT)
from matplotlib.backends.qt_compat import QtWidgets


from PyQt4 import QtGui, QtCore

import logging, sys, os

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
        
        #Initialize logging
        self.logger = logging.getLogger('AudioGUI.logger')
        
        #Initialize text output to GUI
        logstream = OutLog(self.console)
        logging.basicConfig(level=logging.DEBUG, stream=logstream)
        
        sys.stdout = OutLog(self.console, sys.stdout)
        sys.stderr = OutLog(self.console, sys.stderr, QtGui.QColor(255,0,0) )
        
        
        # Initialize the basic plot area
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.plot_vl.addWidget(self.canvas)
        
        self.toolbar = SpectrogramNavBar(self.canvas, self.plot_container)
        self.plot_vl.addWidget(self.toolbar)
        
        self.analyzer = AudioAnalyzer()
        
        #Set up button callbacks
        self.open_file.clicked.connect(self.file_open_dialog)
        
        #Initialize the collection of assorted parameters
        #Not currently customizable, maybe will make interface later
        self.params = {'load_downsampling':1, 'time_downsample_disp':1, 
                       'freq_downsample_disp':1, 'display_threshold':-400, 
                       'split':600}
        
        self.canvas.draw_idle()
        self.show()
        
        self.logger.info('Finished with initialization')
    
    def file_open_dialog(self):
        """Provide a standard file open dialog to import .wav data into the 
        model classes"""  
        self.logger.debug('Clicked the file select button')
              
        file_name = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 
                '/home', 'WAV files (*.wav)')
        
        
        if file_name:
            self.logger.debug('Selected the file %s', str(file_name))

            self.file_name.setText(file_name)
            
            newest_sf = len(self.analyzer.songs)
            self.analyzer.songs.extend(
                    SongFile.load(
                            str(file_name), 
                            split=None,
                            downsampling=self.params['load_downsampling']
                            )
                    )
            
            self.analyzer.set_active(newest_sf)
            
            self.show_data()

        else:
            self.logger.debug('Cancelled file select')
        
    def show_data(self):
        """Put all applicable data from the Model to the View
        
        This function assumes all preprocessing has been completed and merely
        looks at the state of the model and displays it
        """
        if self.analyzer.Sxx is not None:
            #Put the spectrogram data on a graph  
            if not self.toolbar.axis_dict:
                self.toolbar.add_axis('spectrogram') 
                          
            t_step = self.params['time_downsample_disp']
            f_step = self.params['freq_downsample_disp']
            
            self.toolbar.axis_dict['spectrogram'].pcolormesh(
                    self.analyzer.tmesh[::t_step,::f_step], 
                    self.analyzer.fmesh[::t_step,::f_step],
                    self.analyzer.Sxx[::t_step,::f_step], cmap='gray_r')
            
            #cbar = ColorbarBase(self.toolbar.axis_dict['spectrogram'], cmap='gray_r')
            #cbar.set_clim(self.params['display_threshold'], 0)
              
            self.toolbar.x_constraint = self.analyzer.domain       
            self.toolbar.set_domain(self.analyzer.domain)
            self.toolbar.set_range('spectrogram', self.analyzer.freq_range)
            
            self.canvas.draw_idle()

        
        
class SpectrogramNavBar(NavigationToolbar2QT):
    """SpectrogramNavBar docstring goes here TODO
    
    
    """
    
    
    def __init__(self, canvas_, parent_, *args, **kwargs):  
        """Initialization docstring goes here TODO
        
        
        """ 
        
        #initialize logging
        self.logger = logging.getLogger('NavBar.logger')
        
        #A single, unified set of x-boundaries, never violable
        self.x_constraint = ()
        #Dictionary mapping str->axis object
        self.axis_dict = {}
        #Ordered list of axis names, in order added
        self.axis_names = []
        
        self.current_selection = ()
        self.playback = 0
            
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Scroll left', 'back', 'back'),
            ('Forward', 'Scroll right', 'forward', 'forward'),
            (None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            (None, None, None, None),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        
        self.custom_toolitems = (
            (None, None, None, None),
            ('Select', 'Cursor with click, select with drag', 'select1', 'select'),
        )
                
        self.icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
        self.logger.debug('Icons directory %s', self.icons_dir)
        
        NavigationToolbar2QT.__init__(self, canvas_, parent_, coordinates=False)

        for a in self.findChildren(QtGui.QAction):
            if a.text() == 'Customize':
                self.removeAction(a)
                break
             
        for text, tooltip_text, image_file, callback in self.custom_toolitems:
            if text is None:
                self.addSeparator()
            else:
                a = self.addAction(self.local_icon(image_file + '.png'),
                        text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['select']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)
         
        self.addSeparator() 
         
        self.locLabel = QtWidgets.QLabel("", self)
        labelAction = self.addWidget(self.locLabel)
        labelAction.setVisible(True) 
        self.coordinates = True
                    
        self._idSelect = None

        
    def local_icon(self, name):
        imagefile = os.path.join(self.icons_dir, name)
        
        self.logger.debug('Icon image file %s', imagefile)
        return QtGui.QIcon(imagefile)
         
    def add_axis(self, name, init_x=(), init_y=()):
        """Add one axis to this plot, and associate it with name"""
        
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
        
    def set_domain(self, x_domain):
        """Set the domain for ALL plots (which must share an x-axis domain)"""
        
        try:
            assert self.validate(x_domain)
        except AssertionError:
            self.logger.warning("Assert failed: the domain command %s is" 
                    "out of bounds", str(x_domain))
            return
     
        self.logger.debug('Setting the plot domain to %s', str(x_domain))

        for name in self.axis_names:
            ax = self.axis_dict[name]
            self.logger.debug('Setting domain for axis %s', name)
            yrange = ax.get_ylim()
            new_bounds = (x_domain[0], x_domain[1], yrange[0], yrange[1])
            ax.axis(new_bounds)
            
        self.dynamic_update()
    
    def set_range(self, name, yrange):
        """Set the y-axis range of the plot associated with name"""
        
        try:
            ax = self.axis_dict[name]
        except KeyError:
            self.logger.warning('No such axis to set the range')
            return
        
        self.logger.debug('Setting the range of %s to %s', name, str(yrange))
        xbounds = ax.get_xlim()
        new_bounds = (xbounds[0], xbounds[1], yrange[0], yrange[1])
        ax.axis(new_bounds)
        
        self.dynamic_update()
    
    def set_all_ranges(self, yranges):
        """Set the range (y-axis) of all plots linked to this NavBar"""
        
        try:
            for idx, yrange in enumerate(yranges):
                self.set_range(self.axis_names[idx], yrange)
        except IndexError:
            self.logger.warning('Too many yranges provided, there are only %d' 
                    'axes available', len(self.axis_names)+1)
            return

    def forward(self, *args):
        """OVERRIDE the foward function in backend_bases.NavigationToolbar2
        
        Scrolls right
        """
        
        self.logger.debug('Clicked the forward button')
        
        try:
            ax = self.axis_dict[self.axis_names[0]]
        except IndexError:
            self.logger.warning("Scrolling irrelevant, no plots at this time")
        else:
            xbounds = ax.get_xlim()
            width = xbounds[1] - xbounds[0]
            if xbounds[1]+0.25*width <= self.x_constraint[1]:
                dx = 0.25*width
            else:
                dx = self.x_constraint[1] - xbounds[1]
                
            new_bounds = (xbounds[0]+dx, xbounds[1]+dx)
            
            self.set_domain(new_bounds)
     
    def back(self, *args):
        """OVERRIDE the back function in backend_bases.NavigationToolbar2
        
        Scrolls left
        """
        
        self.logger.debug('Clicked the back button')
        
        try:
            ax = self.axis_dict[self.axis_names[0]]
        except IndexError:
            self.logger.warning("Scrolling irrelevant, no plots at this time")
        else:
            xbounds = ax.get_xlim()
            width = xbounds[1] - xbounds[0]
            if xbounds[0]-0.25*width >= self.x_constraint[0]:
                dx = 0.25*width
            else:
                dx = xbounds[0] - self.x_constraint[0]
                
            new_bounds = (xbounds[0]-dx, xbounds[1]-dx)
            
            self.set_domain(new_bounds)
    
    
    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        self._actions['pan'].setChecked(self._active == 'PAN')
        self._actions['zoom'].setChecked(self._active == 'ZOOM')
        self._actions['select'].setChecked(self._active == 'SELECT')
    
    def select(self, *args):
        self.logger.debug('Clicked the select button')
        
        """Activate the pan/zoom tool. pan with left button, zoom with right"""
        # set the pointer icon and button press funcs to the
        # appropriate callbacks

        if self._active == 'SELECT':
            self._active = None
        else:
            self._active = 'SELECT'
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

        if self._active:
            self._idPress = self.canvas.mpl_connect(
                'button_press_event', self.press_select)
            self._idRelease = self.canvas.mpl_connect(
                'button_release_event', self.release_select)
            self.mode = 'select/set'
            self.canvas.widgetlock(self)
        else:
            self.canvas.widgetlock.release(self)

        #for a in self.canvas.figure.get_axes():
        #    a.set_navigate_mode(self._active)

        self.set_message(self.mode)
        
        self._update_buttons_checked()

        
    
    def press_select(self, event):
        """the press mouse button in select mode callback"""

        
        # If we're already in the middle of a zoom, pressing another
        # button works to "cancel"
        self.remove_rubberband()
        
        if self._idSelect:
            self.canvas.mpl_disconnect(self._idSelect)
            self.release(event)
            self.draw()
            self._xypress = None
            self._button_pressed = None
            return

        if event.button == 1:
            self._button_pressed = 1
            self.logger.debug('Pressed left mouse to select region in select mode')

        elif event.button == 3:
            self.logger.debug('Pressed right mouse button to place start mark in select mode')
            self._button_pressed = 3
        else:
            self._button_pressed = None
            return

        x, y = event.x, event.y

        # push the current view to define home if stack is empty
        if self._views.empty():
            self.push_current()

        self._xypress = []
        for i, a in enumerate(self.canvas.figure.get_axes()):
            if (x is not None and y is not None and a.in_axes(event)):
                self._xypress.append((x, y, a, i, a._get_view()))

        self._idSelect = self.canvas.mpl_connect('motion_notify_event', 
                self.drag_select)

        self.press(event)
    
    def release_select(self, event):
        self.logger.debug('Released mouse in select mode')
        
        self.canvas.mpl_disconnect(self._idSelect)
        self._idSelect = None

        if not self._xypress:
            return
        
        x, y = event.x, event.y
        lastx, lasty, a, ind, view = self._xypress[0]
    
        if self._button_pressed == 3:
            self.playback = x
            self.logger.debug('Playback marker set to %0.4f', self.playback)
        
        elif self._button_pressed == 1:
            # ignore singular clicks - 5 pixels is a threshold
            # allows the user to "cancel" a selection action
            # by selecting by less than 5 pixels
            if ((abs(x - lastx) < 5 and self._zoom_mode!="y") or
                    (abs(y - lasty) < 5 and self._zoom_mode!="x")):
                
                
                self.current_selection = ()
                
            else:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
                self.logger.debug('format_coord yields %s', s)
                
                if lastx > x:
                    self.current_selection = (x, lastx)
                else:
                    self.current_selection = (lastx, x)
                    
                self.logger.debug('Selection  set to %s', 
                        str(self.current_selection))


        self.draw()
        self._xypress = None
        self._button_pressed = None

        self.release(event)
    
    def drag_select(self, event):
        self.logger.debug('Dragged in select mode with button %d', 
                self._button_pressed)
        pass
    
    
    def release_zoom(self, event):
        """OVERRIDE the release_zoom method in backend_bases.NavigationToolbar2
        
        Identical function with domain checking added
        """
        
        self.logger.debug('Called release_zoom')
        
        for zoom_id in self._ids_zoom:
            self.canvas.mpl_disconnect(zoom_id)
        self._ids_zoom = []

        self.remove_rubberband()

        if not self._xypress:
            return

        last_a = []

        for cur_xypress in self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, ind, view = cur_xypress
            # ignore singular clicks - 5 pixels is a threshold
            # allows the user to "cancel" a zoom action
            # by zooming by less than 5 pixels
            if ((abs(x - lastx) < 5 and self._zoom_mode!="y") or
                    (abs(y - lasty) < 5 and self._zoom_mode!="x")):
                
                if self._button_pressed == 1:
                    direction = 'in'
                    
                elif self._button_pressed == 3:
                    direction = 'out'
                
                
                self._xypress = None
                self.release(event)
                self.draw()
                return

            # detect twinx,y axes and avoid double zooming
            twinx, twiny = False, False
            if last_a:
                for la in last_a:
                    if a.get_shared_x_axes().joined(a, la):
                        twinx = True
                    if a.get_shared_y_axes().joined(a, la):
                        twiny = True
            last_a.append(a)

            if self._button_pressed == 1:
                direction = 'in'
            elif self._button_pressed == 3:
                direction = 'out'
            else:
                continue

            a._set_view_from_bbox((lastx, lasty, x, y), direction,
                                  self._zoom_mode, twinx, twiny)
            
            if not self.validate(a.get_xlim()):
                self.set_domain(self.x_constraint)

        self.draw()
        self._xypress = None
        self._button_pressed = None

        self._zoom_mode = None

        self.push_current()
        self.release(event)

    def drag_pan(self, event):
        """OVERRIDE the drag_pan function in backend_bases.NavigationToolbar2
        
        Adjusted to make sure limits are maintained
        """
        
        self.logger.debug('Called drag_pan')

        for a, _ in self._xypress:
            #safer to use the recorded button at the press than current button:
            #multiple buttons can get pressed during motion...
            pre_drag = a.get_xlim()
            a.drag_pan(self._button_pressed, event.key, event.x, event.y)
            
            if not self.validate(a.get_xlim()):
                self.set_domain(pre_drag)
            
        self.dynamic_update()
    
    def validate(self, xlims):
        if not self.x_constraint:
            return True
        else:
            return xlims[0] >= self.x_constraint[0] and xlims[1] <= self.x_constraint[1]
        
 
class OutLog:
    '''OutLog pipes output from a stream to a QTextEdit widget
    
    This class is taken exactly from stackoverflow
    http://stackoverflow.com/questions/17132994/pyside-and-python-logging/17145093#17145093
    '''
    
    def __init__(self, edit, out=None, color=None):
        """(edit, out=None, color=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        color = alternate color (i.e. color stderr a different color)
        """
        self.edit = edit
        self.out = None
        self.color = color

    def write(self, m):
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QtGui.QTextCursor.End)
        self.edit.insertPlainText( m )

        if self.color:
            self.edit.setTextColor(tc)

        if self.out:
            self.out.write(m)
            
    def flush(self):
        pass
 
           
def main():    
    app = QtGui.QApplication(sys.argv)
    main = AudioGUI()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
