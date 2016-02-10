'''
Created on Feb 9, 2016

@author: justinpalpant
'''
from PyQt4.QtCore import QThread, QObject, QTimer
from PyQt4.QtCore import pyqtSignal, pyqtSlot

class BGThread(QThread):
    '''
    BGThread defines a background QThread that will run a single function
    
    The function, self.function, must be parameterless, because it is called as
    part of start() without arguments.  If there are parameters for the
    function, use a lambda to make it a parameterless lambda before initializing
    the BGThread
    '''
        
    def __init__(self, fn, name=''):
        '''Create a BGThread that will run fn() when started'''
        
        super(BGThread, self).__init__()
        self._function = fn
        self.name = name

    def __del__(self):
        '''Safe thread deletion... maybe bad if a thread is non-terminating'''
        self.wait()

    def run(self):
        '''The overriden run function just calls this thread's function'''
        self._function()

class SignalStream(QObject):
    '''SignalStream is a file-like object that emits a text signal on writing
    
    This class is used to provide threadsafe communication of data to the GUI.
    A SignalStream can be used in place of sys.stdout and the instance's 
    write_signal can be connected to a slot that processes the text to where it
    ought to go.  Since signals and slots are threadsafe, this lets you pass
    text from anywhere to anywhere reasonably safely
    
    SignalStream uses some intelligent buffering to prevent the signalstorm that
    happened the first time I used it.  Signal emit only happens when flush()
    is called - so an application can force a flush - but in order to make sure
    that happens reasonable often SignalStream can be initialized with a QTimer
    on an interval (default: 100ms) and the QTimer will make sure to call flush()
    every 100ms.
    '''
    
    write_signal = pyqtSignal(str)
    
    def __init__(self, interval_ms=100):
        '''Create a SignalStream that emits text at least every interval_ms'''
        
        super(SignalStream, self).__init__()
        self.data = []
        
        self.timer = QTimer()
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.flush)
        self.timer.start()
        
    def write(self, m):
        '''Add the message in m to this stream's cache'''
        self.data.append(m)
        
    @pyqtSlot()
    def flush(self):
        '''Write all data in the stream and clear the stream's cache'''
        if self.data:
            self.write_signal.emit(''.join(self.data))
            self.data = []
            
    def set_interval(self, interval_ms):
        '''Alter the timer period'''
        self.timer.setInteval(interval_ms)