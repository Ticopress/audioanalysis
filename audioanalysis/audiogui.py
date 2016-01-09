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

from PyQt4.uic import loadUiType

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT)
from matplotlib.backend_bases import cursors

from PyQt4 import QtGui, QtCore

import logging, sys, os, pyaudio, time, fnmatch
import numpy as np

from freqanalysis import AudioAnalyzer, SongFile
from PyQt4.QtCore import pyqtSignal


Ui_MainWindow, QMainWindow = loadUiType('main.ui')

class AudioGUI(Ui_MainWindow, QMainWindow):
    """The GUI for automatically identifying motifs
    
    
    """
    

    def __init__(self):
        """Create a new AudioGUI
        
        
        """
        #Initialization of GUI from Qt Designer
        super(AudioGUI, self).__init__()
        self.setupUi(self)
        
        #Initialize logging
        self.logger = logging.getLogger('AudioGUI.logger')
        
        #Initialize text output to GUI
        #sys.stdout = OutLog(self.console, sys.stdout)
        #sys.stderr = OutLog(self.console, sys.stderr, QtGui.QColor(255,0,0) )
        
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        
        # Initialize the basic plot area
        canvas = SpectrogramCanvas(Figure())
        toolbar = SpectrogramNavBar(canvas, self)

        self.set_canvas(canvas, self.plot_vl)
        
        #Set up button callbacks
        self.play_button.clicked.connect(self.click_play_button)
        self.entropy_checkbox.clicked.connect(lambda: self.plot('entropy'))
        self.power_checkbox.clicked.connect(lambda: self.plot('power'))
        self.classes_checkbox.clicked.connect(lambda: self.plot('classification'))
        
        #Set up menu callbacks
        self.action_load_files.triggered.connect(self.select_wav_files)
        self.action_load_folder.triggered.connect(self.select_wav_folder)
        self.action_load_nn.triggered.connect(self.select_neural_net_file)
        
        self.action_new_nn.triggered.connect(self.create_new_neural_net)
        
        self.action_save_all_motifs.triggered.connect(lambda: self.save_motifs('all'))
        self.action_save_current_motif.triggered.connect(lambda: self.save_motifs('current'))
        self.action_save_nn.triggered.connect(self.save_neural_net)
        
        self.action_classify_all.triggered.connect(lambda: self.auto_classify('all'))
        self.action_classify_current.triggered.connect(lambda: self.auto_classify('current'))
        
        self.action_find_all_motifs.triggered.connect(lambda: self.find_motifs('all'))
        self.action_find_current_motifs.triggered.connect(lambda: self.find_motifs('current'))
        
        #Initialize the collection of assorted parameters
        #Not currently customizable, maybe will make interface later
        defaultlayers = [
                {'type':'Convolution2D', 'args':(16,3,1,), 'kwargs':{'border_mode':'same'}},
                {'type':'Activation', 'args':('relu',)},
                {'type':'Convolution2D', 'args':(16,3,1,)},
                {'type':'Activation', 'args':('relu',)},
                {'type':'MaxPooling2D', 'kwargs':{'pool_size':(2,1,)}},
                {'type':'Dropout', 'args':(0.25,)},
                {'type':'Flatten'},
                {'type':'Dense', 'args':(128,)},
                {'type':'Activation', 'args':('relu',)},
                {'type':'Dropout', 'args':(0.5,)},
                ]
        
        
        self.params = {'load_downsampling':1, 'time_downsample_disp':1, 
                       'freq_downsample_disp':1, 'display_threshold':-400, 
                       'split':600, 'vmin':-90, 'vmax':-20, 'nfft':512, 
                       'fft_time_window_ms':10, 'fft_time_step_ms':2, 
                       'process_chunk_s':15, 'layers':defaultlayers, 
                       'loss':'categorical_crossentropy', 'optimizer':'adadelta',
                       }
            
        self.analyzer = AudioAnalyzer(**self.params)
        self.player = pyaudio.PyAudio()
        
        self.canvas.draw_idle()
        self.show()
        
        self.logger.info('Finished with initialization')
    
    
    def set_canvas(self, canvas, loc):
        """Set the SpectrogramCanvas for this GUI
        
        Assigns the given canvas to be this GUI's canvas, connects any
        relevant slots, and places the canvas and its tools into the loc 
        container
        """
        loc.addWidget(canvas)
        
        for t in canvas.tools.values():
            loc.addWidget(t)
            
        self.canvas = canvas
        
        #Connect any slots coming from canvas here
        #-----slots-----
    
    def select_wav_files(self):
        """Load one or more wav files as SongFiles"""
        self.logger.debug('Clicked the file select button')
              
        file_names = QtGui.QFileDialog.getOpenFileNames(self, 'Select file(s)', 
                '', 'WAV files (*.wav)')
        if file_names:
            self.load_wav_files(file_names)
        else:
            self.logger.debug('Cancelled file select')
            
    def select_wav_folder(self):
        """Load all .wav files in a folder and all its subfolders as SongFiles"""
        
        self.logger.debug('Clicked the folder select button')
              
        folder_name = QtGui.QFileDialog.getExistingDirectory(self, 'Select folder')
        self.logger.info('selected %s', str(folder_name))
        if folder_name:
            pass
            self.load_wav_files(self.find_files(str(folder_name), '*.wav'))
        else:
            self.logger.debug('Cancelled file select')
    
    def load_wav_files(self, file_names):
        """Load a list of wave files as SongFiles"""
        
        for f in file_names:
            self.logger.debug('Loading the file %s', str(f))
            
            new_songs = SongFile.load(
                            str(f),
                            downsampling=self.params['load_downsampling']
                            )
            
            self.logger.info('Loaded %s as %d SongFiles', str(f), 
                    len(new_songs))
            
            self.analyzer.songs.extend(new_songs)
            self.update_table('songs')
                    
    
    def update_table(self, name):
        """Display information on loaded SongFiles in the table"""
        if name=='songs':
            pass
        elif name=='motifs':
            pass
        else:
            self.logger.warning('No table %s, cannot update', name)
    
    def select_neural_net_file(self):
        """Load one or more wav files as SongFiles"""
        self.logger.debug('Clicked the file select button')
              
        file_names = QtGui.QFileDialog.getOpenFileNames(self, 'Open file', 
                '/home', 'Neural Net files (*.nn)')
        if file_names:
            self.load_wav_files(file_names)
        else:
            self.logger.debug('Cancelled file select')
            
    def create_new_neural_net(self):
        """Uses the Analyzer's active_song to construct and train a neural net"""
        self.analyzer.nn = self.analyzer.build_neural_net(**self.analyzer.params)
        
        #then, train it
        
    def save_neural_net(self):
        """Save the analyzer's current neural net to a file to avoid training"""
        pass
    
    def save_motifs(self, mode):
        if mode == 'all':
            #Get a folder, put them there
            pass
        elif mode == 'current':
            #Get a filename, put it there
            pass
        else:
            #Get a filename, put it there
            try:
                self.analyzer.motifs[mode].export()
            except TypeError:
                self.logger.error('Unknown motif export mode, cannot export')
            
    def click_play_button(self):
        """Callback for clicking the GUI button"""
        try:
            if self.play_button.isChecked():
                self.start_playback()
                self.logger.debug('Playback started')
            else:
                self.logger.debug('Ending playback')  
                self.stop_playback()    
        except AttributeError:
            self.logger.error('Could not execute playback, no song prepared')
            self.play_button.setChecked(not self.play_button.isChecked())
              
        
    def start_playback(self):
        """Open and start a PyAudio stream
        
        Raises AttributeError if self.analyzer.active_song is not set
        """
        self.stream = self.player.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=int(self.analyzer.active_song.Fs),
                output=True,
                stream_callback=self.play_audio_callback,
                frames_per_buffer=4096)
        
        self.stream.start_stream()
 
    
    def stop_playback(self):
        """Stop and close a PyAudio stream
        
        Raises AttributeError if self.stream is not set
        """
        if self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()

    def play_audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for separate thread that plays audio
        
        This is responsible for moving the marker on the canvas, and since the
        stream auto-dies when it reaches the end of the data, no handling is
        needed to make sure the marker stops moving
        """
        
        index = int(self.canvas.marker * self.analyzer.active_song.Fs)
        data = self.analyzer.active_song.data[index:index+frame_count]
        
        self.canvas.set_marker((index+frame_count)/self.analyzer.active_song.Fs)
        
        return (data, pyaudio.paContinue)    
    
    def select_song(self):
        """Put all applicable data from the Model to the View
        
        This function assumes all preprocessing has been completed and merely
        looks at the state of the model and displays it
        """
        
        self.analyzer.set_active(self.analyzer.songs[0])
        
        for p in ['spectrogram', 'classification', 'entropy', 'power']:
            self.plot(p)
        
        self.canvas.set_marker(0)
        self.canvas.set_selection(())
        
    def keyPressEvent(self, e):
        """Listens for a keypress
        
        If the keypress is a number and a region has been selected with the
        select tool, the keypress will assign the value of that number to be
        the classification of the selected region
        """
        if (e.text() in [str(i) for i in range(10)] and 
                self.canvas.current_selection):
            indices = np.searchsorted(self.analyzer.active_song.time, 
                    np.asarray(self.canvas.current_selection))
            
            self.analyzer.active_song.classification[indices[0]:indices[1]] = int(e.text())
            
            self.logger.debug('Updating class from %s to be %d', 
                    str(self.analyzer.active_song.time[indices]), int(e.text()))
        
            self.plot('classification')
            self.canvas.set_selection(())  
                 
    def plot(self, plot_type):
        """Show active song data on the plot
        
        This method relies on the state of the GUI checkboxes to know whether or
        not a type of data should be shown or hidden.
        
        Inputs:
            plot_type: string specifying which set of data to display
        """
        t_step = self.params['time_downsample_disp']
        f_step = self.params['freq_downsample_disp']
        
        try:   
            time = self.analyzer.active_song.time[::t_step]
            freq = self.analyzer.active_song.freq[::f_step]
            classification = self.analyzer.active_song.classification[::t_step]
            entropy = self.analyzer.active_song.entropy[::t_step]
            power = self.analyzer.active_song.power[::t_step]
            disp_Sxx = np.flipud(self.analyzer.Sxx[::t_step, ::f_step])
        except AttributeError:
            self.logger.warning('No active song, cannot display plot %s', plot_type)
            return
        
        if plot_type == 'spectrogram':
            pass
        elif plot_type == 'classification':
            pass
        elif plot_type == 'entropy':
            pass
        elif plot_type == 'power':
            pass
        else:
            self.logger.warning('Unknown plot type %s, cannot plot', plot_type)
    
    def auto_classify(self, mode):
        pass
    
    def find_motifs(self, mode):
        pass
       
    def find_files(self, directory, pattern):
        """Recursively walk a directory and return filenames matching pattern"""
        
        files_out = []
        for root, _, files in os.walk(directory):
            self.logger.debug('Looking in %s', str(root))
            for basename in files:
                self.logger.debug('Found basename %s', str(basename))
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    
                    files_out.append(filename)
                    
        return files_out
    
        
class SpectrogramCanvas(FigureCanvas, QtCore.QObject):
    """Subclasses the FigureCanvas to provide features for spectrogram display
    
    Of primary interest is that this class keeps record of all the axes on the
    plot by name, with independent y scales but a uniform x-scale.  The primary
    intent of this class is to bound the x-scale so that it is impossible to
    scroll, pan, or zoom outside of the domain set in the public instance
    variable x_constraint.  Anything that moves that plot that is not contained
    within this class should emit a signal which should be bound to this class's
    'navigate' method.  navigate corrects for any navigation not done by the 
    SpectrogramCanvas and then redraws.
    """
    
    def __init__(self, figure_):
        self.logger = logging.getLogger('SpectrogramCanvas.logger')
        
        FigureCanvas.__init__(self, figure_)
        QtCore.QObject.__init__(self)
        
        #Extent of the spectrogram data, for homing and as domain bound
        self.extent = ()
        #Dictionary mapping str->axis object
        self.axis_dict = {}
        #Ordered list of axis names, in order added
        self.axis_names = []
        
        self.tools = {}
        
        self.image = None
        
        self.last_gui_refresh_time = time.time()
        self.marker = 0
        self.current_selection = ()
        
    def add_tool(self, tool):
        """Add the tool to the tools dictionary and connect any known signals"""
        
        self.tools[type(tool).__name__] = tool
        
        if isinstance(tool, SpectrogramNavBar):
            tool.navigate.connect(self.navigate)
            tool.set_selection.connect(self.set_selection)
            tool.set_marker.connect(self.set_marker)
    
    @QtCore.pyqtSlot(dict)
    def navigate(self, kwargs):
        """Receive all navigation signals that this canvas must deal with"""
        
        try:
            navtype = kwargs['type']
        except KeyError:
            self.error('Invalid navigation signal or command %s, no type'
                    ' provided', str(kwargs))
            return
        
        if navtype == 'drag_pan':
            try:
                ax = kwargs['axis']
                button = kwargs['button']
                key = kwargs['key']
                x = kwargs['x']
                y = kwargs['y']
                pre_drag_x = kwargs['pre_drag_x']
                pre_drag_y = kwargs['pre_drag_y']
            except KeyError:
                self.logger.error('Invalid drag command or signal, cannot execute')
                return
            
            ax.drag_pan(button, key, x, y)
            
            if not self.valid(ax.get_xlim()):
                self.set_domain(pre_drag_x)
                
            if ax is not self.axis_dict['spectrogram']:
                ax.set_ylim(pre_drag_y) 
                
        elif navtype == 'release_zoom':
            try:
                ax = kwargs['axis']
                key = kwargs['key']
                tup = kwargs['zoom_tuple']
                mode = kwargs['mode']
            except KeyError:
                self.logger.error('Invalid drag command or signal, cannot execute')
                return
            
            if key == 1:
                direction = 'in'
            elif key == 3:
                direction = 'out'
                
            ax._set_view_from_bbox(tup, direction, mode, False, False)   
            
            if direction=='out' and not self.valid(ax.get_xlim()):
                self.set_domain(ax.get_xlim())
                   
        elif navtype == 'forward':
            #Get any plot, read the plot domain, move left by 25% of domain
            try:
                ax = self.axis_dict[self.axis_names[0]]
            except IndexError:
                self.logger.warning("Scrolling irrelevant, no plots at this time")
            else:
                xbounds = ax.get_xlim()
                width = xbounds[1] - xbounds[0]
                if xbounds[1]+0.25*width <= self.extent[1]:
                    dx = 0.25*width
                else:
                    dx = self.extent[1] - xbounds[1]
                    
                new_bounds = (xbounds[0]+dx, xbounds[1]+dx)
                
                self.set_domain(new_bounds)  
                  
        elif navtype == 'back':
            #Get any plot, read the plot domain, move left by 25% of domain
            try:
                ax = self.axis_dict[self.axis_names[0]]
            except IndexError:
                self.logger.warning("Scrolling irrelevant, no plots at this time")
            else:
                xbounds = ax.get_xlim()
                width = xbounds[1] - xbounds[0]
                if xbounds[0]-0.25*width >= self.extent[0]:
                    dx = 0.25*width
                else:
                    dx = xbounds[0] - self.extent[0]
                    
                new_bounds = (xbounds[0]-dx, xbounds[1]-dx)
                
                self.set_domain(new_bounds)  
                      
        elif navtype == 'home':
            self.set_domain(self.extent[0:2])
            self.set_range('spectrogram', self.extent[2:4])

        else:
            self.logger.warning('Unknown navigate type %s', navtype)
        
        self.draw_idle()
        
    @QtCore.pyqtSlot(tuple)
    def set_selection(self, sel):
        if not sel:
            self.current_selection = ()
            self.drawRectangle(None)
        else:
            self.current_selection = sel
            
        self.logger.debug('Selection set to %s', str(self.current_selection))
    
    @QtCore.pyqtSlot(float)
    def set_marker(self, marker):
        """Sets the marker, but does not update the GUI unless the change in
        values from the last GUI value is significant
        """
        
        self.marker = marker
        t = time.time()
        
        if t - self.last_gui_refresh_time > 0.05:
            try:
                ax = self.axis_dict['marker']
            except KeyError:
                self.add_axis('marker')
                ax = self.axis_dict['marker']
            
            if ax.lines:
                ax.lines.remove(ax.lines[0])
            
            ax.plot([self.marker, self.marker],[0,1], 'k--', linewidth=2)
            self.set_range('marker', (0,1))
            self.draw_idle()
            self.last_gui_refresh_time = t
                
    def add_axis(self, name):
        """Add one axis to this plot, and associate it with name"""
        
        if not self.axis_dict:
            self.axis_dict[name] = self.canvas.figure.add_subplot(111)
            self.axis_names.append(name)
        else:
            self.axis_dict[name] = self.axis_dict[self.axis_names[0]].twinx()
            self.axis_names.append(name)
                
        if name in ['marker', 'classification', 'power', 'entropy']:
            self.axis_dict[name].yaxis.set_visible(False) 
            
    def set_domain(self, x_domain):
        """Set the domain for ALL plots (which must share an x-axis domain)"""
        
        try:
            assert self.valid(x_domain)
        except AssertionError:
            self.logger.warning("Assert failed: the domain command %s is" 
                    "out of bounds", str(x_domain))
            return
     
        self.logger.debug('Setting the plot domain to %s', str(x_domain))

        for name in self.axis_names:
            ax = self.axis_dict[name]
            self.logger.debug('Setting domain for axis %s to %s', name, str(x_domain))
            ax.set_xlim(x_domain)
            
        self.draw_idle()
        
    def set_range(self, name, y_range):
        """Set the y-axis range of the plot associated with name"""
        
        try:
            ax = self.axis_dict[name]
        except KeyError:
            self.logger.warning('No such axis %s to set the range', name)
            return
        
        self.logger.debug('Setting the range of %s to %s', name, str(y_range))
        ax.set_ylim(y_range)
        
        self.draw_idle()
        
    def set_all_ranges(self, yranges):
        """Set the range (y-axis) of all plots linked to this NavBar"""
        
        try:
            for idx, yrange in enumerate(yranges):
                self.set_range(self.axis_names[idx], yrange)
        except IndexError:
            self.logger.warning('Too many yranges provided, there are only %d' 
                    'axes available', len(self.axis_names)+1)
            return
        
    def valid(self, xlims):
        if not xlims:
            return False
        if not self.extent:
            return True
        else:
            return xlims[0] >= self.extent[0] and xlims[1] <= self.extent[1]
        
    def validate(self, xlims):
        if self.valid(xlims):
            return xlims
        else:
            return (max(xlims[0], self.extent[0]), min(xlims[1], self.extent[1]))
         
            
    def display_spectrogram(self, t, f, Sxx):
        """Fetches spectrogram data from analyzer and plots it
        
        NO display method affects the domain in any way.  That must be done
        external to the display method
        """   
          
        try:
            ax = self.axis_dict['spectrogram']
        except KeyError:
            self.add_axis('spectrogram') 
            ax = self.axis_dict['spectrogram']
        
        halfbin_time = (t[1] - t[0]) / 2.0
        halfbin_freq = (f[1] - f[0]) / 2.0
        
        # this method is much much faster!
        # center bin
        self.extent = (t[0] - halfbin_time, t[-1] + halfbin_time,
                  f[0] - halfbin_freq, f[-1] + halfbin_freq)
        
        self.image = ax.imshow(Sxx, 
                interpolation="nearest", 
                extent=self.extent,
                cmap='gray_r',
                vmin=self.params['vmin'],
                vmax=self.params['vmax']
                )
        
        ax.axis('tight')

        self.set_domain(self.extent[0:2])
        self.set_range('spectrogram', self.extent[2:4])
  
        self.draw_idle() 
        
    def display_classification(self, t, classes, show=True):
        """Fetches classification data from analyzer and plots it
        
        NO display method affects the domain in any way.  That must be done
        external to the display method
        """
        
        try:
            ax = self.axis_dict['classification']
        except KeyError:
            self.add_axis('classification')
            ax = self.axis_dict['classification']
        
        if ax.lines:
            ax.lines.remove(ax.lines[0])
            
        l, = ax.plot(t, classes, 'b-')
        
        self.set_range('classification', (0, max(classes)+1))
        
        l.set_visible(show)

        self.canvas.draw_idle() 
        
        
    def display_entropy(self, t, entropy, show=True):

        try:
            ax = self.axis_dict['entropy']
        except KeyError:
            self.add_axis('entropy')
            ax = self.axis_dict['entropy']
             
        
        if ax.lines:
            ax.lines.remove(ax.lines[0])
            
        l, = ax.plot(t, entropy, 'g-')
           
        self.set_range('entropy', (min(entropy), max(entropy)))

        l.set_visible(show)

        self.draw_idle()  
        
    def display_power(self, t, power, show=True):
        try:
            ax = self.axis_dict['power']
        except KeyError:
            self.add_axis('power')
            ax = self.axis_dict['power']
        
        if ax.lines:
            ax.lines.remove(ax.lines[0])
            
        l, = ax.plot(t, power, 'r-')

        self.set_range('power', (min(power), max(power)))
        
        l.set_visible(show)
        
        self.draw_idle()        
         
class SpectrogramNavBar(NavigationToolbar2QT):
    """Provides a navigation bar specially configured for spectrogram interaction
    
    This class overrides a number of the methods of the standard 
    NavigationToolbar2QT and NavigationToolbar2, and adds some convenience
    methods.

    The navigation bar tools have been updated.  The left and right buttons 
    scroll left and right, rather than moving between views.  The pan and zoom 
    buttons are identical, but with the horizontal bound enforced.  Several 
    buttons are removed, and a 'select' button has been added which is used for 
    manual classification of data.
    
    Emits a signal 'navigate' which is called any time an action is take which
    would change the plot bounds.
    """
    
    #List of signals for emitted by this class
    navigate = pyqtSignal(dict)
    set_marker = pyqtSignal(int)
    set_selection = pyqtSignal(tuple)
    
    def __init__(self, canvas_, parent_, *args, **kwargs):  
        """Creates a SpectrogramNavigationBar instance
        
        Requires that canvas_ be a SpectrogramCanvas
        """ 
        
        #initialize logging
        self.logger = logging.getLogger('SpectrogramNavBar.logger')
            
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
        
        NavigationToolbar2QT.__init__(self, canvas_, parent_, coordinates=False)
                
        self.custom_toolitems = (
            (None, None, None, None),
            ('Select', 'Cursor with click, select with drag', 'select1', 'select'),
        )
                
        self.icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
        self.logger.debug('Icons directory %s', self.icons_dir)

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
                if callback in ['select', 'scale']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)
         
        self.addSeparator() 
         
        self.locLabel = QtGui.QLabel("", self)
        labelAction = self.addWidget(self.locLabel)
        labelAction.setVisible(True) 
        self.coordinates = True
                    
        self._idSelect = None
        
        try:
            canvas_.add_tool(self)
        except AttributeError as e:
            self.logger.error('Cannot initialize SpectrogramNavBar - canvas '
                    'type not valid')
            raise e
        
    def local_icon(self, name):
        """Load a file in the /icons folder as a QIcon"""
        
        imagefile = os.path.join(self.icons_dir, name)
        
        self.logger.debug('Icon image file %s', imagefile)
        return QtGui.QIcon(imagefile)


    def forward(self, *args):
        """OVERRIDE the foward function in backend_bases.NavigationToolbar2
        
        Emits a signal causing the SpectrogramCanvas to scroll right
        """
        
        self.logger.debug('Clicked the forward button')
        
        self.set_selection.emit(())
        self.navigate.emit({'type':'forward'})
     
    def back(self, *args):
        """OVERRIDE the back function in backend_bases.NavigationToolbar2
        
        Emits a signal causing the SpectrogramCanvas to scroll left
        """
        self.logger.debug('Clicked the back button')
        
        self.set_selection.emit(())
        self.navigate.emit({'type':'back'})
     
    def home(self, *args):
        """Override the home method of backend_bases.NavigationToolbar2"""
        self.navigate.emit({'type':'home'})
            
    def drag_pan(self, event):
        """OVERRIDE the drag_pan function in backend_bases.NavigationToolbar2
        
        Adjusted to make sure limits are maintained
        """
        
        for a, _ in self._xypress:
            #safer to use the recorded button at the press than current button:
            #multiple buttons can get pressed during motion...

            drag_data = {
                    'type':'drag_pan',
                    'axis':a, 
                    'button':self._button_pressed, 
                    'key':event.key,
                    'x':event.x,
                    'y':event.y,
                    'pre_drag_x':a.get_xlim(),
                    'pre_drag_y':a.get_ylim()}

            self.navigate.emit(drag_data)
            
    def pan(self, *args):
        self.set_selection.emit(())
        super(SpectrogramNavBar, self).pan(*args)
        
    def zoom(self, *args):
        self.set_selection.emit(())
        super(SpectrogramNavBar, self).zoom(*args)
        
    def release_zoom(self, event):
        """OVERRIDE the release_zoom method in backend_bases.NavigationToolbar2
        
        Emits a signal containing the data necessary to zoom the plot
        """
        
        self.logger.debug('Called release_zoom')
        
        for zoom_id in self._ids_zoom:
            self.canvas.mpl_disconnect(zoom_id)
        self._ids_zoom = []

        self.remove_rubberband()
        
        for cur_press in self._xypress:
            x, y = event.x, event.y
            lastx, lasty, ax, _, _ = cur_press
            
            if abs(x - lastx) < 5 and abs(y - lasty) < 5:
                self._xypress = None
                self.release(event)
                continue

            data = {
                    'type':'release_zoom',
                    'axis':ax,
                    'key':self._button_pressed,
                    'zoom_tuple':(lastx, lasty, x, y),
                    'mode':self._zoom_mode
                    }
            
            self.navigate.emit(data)
        
        self._xypress = None
        self._button_pressed = None

        self._zoom_mode = None

        self.push_current()
        self.release(event) 
            
    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        super(SpectrogramNavBar, self)._update_buttons_checked()
        self._actions['select'].setChecked(self._active == 'SELECT')
    
    def select(self, *args):
        self.logger.debug('Clicked the select button on the toolbar')
        
        """Activate the select tool. select with left button, set cursor with right"""
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
            self.logger.debug('Pressed left mouse to select region in '
                    'select mode')

        elif event.button == 3:
            self.logger.debug('Pressed right mouse button to place start mark'
                    ' in select mode')
            self._button_pressed = 3
        else:
            self._button_pressed = None
            return

        x, y = event.x, event.y
        self._select_start = event.xdata

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

        lastx, lasty, _, _, _ = self._xypress[0]
    
        if self._button_pressed == 3:
            self.set_marker.emit(event.xdata)
    
        elif self._button_pressed == 1:
            # ignore singular clicks - 5 pixels is a threshold
            # allows the user to "cancel" a selection action
            # by selecting by less than 5 pixels
            if (abs(event.x - lastx) < 5 and abs(event.y - lasty) < 5):
                
                self.set_selection.emit(())
                
            else:
                if event.xdata > self._select_start:
                    sel = (self._select_start, event.xdata)
                else:
                    sel = (event.xdata, self._select_start)
                
                self.set_marker.emit(sel[0])
                self.set_selection.emit(sel)

        self._xypress = None
        self._button_pressed = None

        self.release(event)
    
    def drag_select(self, event):
        
        if self._button_pressed == 3:
            self.set_marker.emit(event.xdata)
             
        elif self._button_pressed == 1:
            if self._xypress:
                x, y = event.x, event.y
                lastx, lasty, a, _, _ = self._xypress[0]
    
                # adjust x, last, y, last
                x1, y1, x2, y2 = a.bbox.extents
                x, lastx = max(min(x, lastx), x1), min(max(x, lastx), x2)
                y, lasty = max(min(y, lasty), y1), min(max(y, lasty), y2)
    
                self.draw_rubberband(event, x, y, lastx, lasty)
        
    def _set_cursor(self, event):
        """OVERRIDE the _set_cursor method in backend_bases.NavigationToolbar2"""
        
        if not event.inaxes or not self._active:
            if self._lastCursor != cursors.POINTER:
                self.set_cursor(cursors.POINTER)
                self._lastCursor = cursors.POINTER
        else:
            if self._active == 'ZOOM':
                if self._lastCursor != cursors.SELECT_REGION:
                    self.set_cursor(cursors.SELECT_REGION)
                    self._lastCursor = cursors.SELECT_REGION
            elif (self._active == 'PAN' and
                  self._lastCursor != cursors.MOVE):
                self.set_cursor(cursors.MOVE)

                self._lastCursor = cursors.MOVE   
            elif (self._active == 'SELECT' and
                    self._lastCursor != cursors.SELECT_REGION):
                self.set_cursor(cursors.SELECT_REGION)
    
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
