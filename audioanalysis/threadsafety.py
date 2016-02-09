'''
Created on Feb 9, 2016

@author: justinpalpant
'''
from contextlib import contextmanager
import sys
from PyQt4 import QtCore

@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout

class BGThread(QtCore.QThread):
    #say = QtCore.pyqtSignal(str)
    #QThread also defines started and finished as void-type signals
    
    def __init__(self, fn, *args, **kwargs):
        super(BGThread, self).__init__()
        self.function = fn
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        self.wait()

    def run(self):
        self.function(*self.args, **self.kwargs)

    @property
    def name(self):
        return self.function.func_name

class SignalStream(QtCore.QObject):
    write_signal = QtCore.pyqtSignal(str)
    
    def __init__(self):
        super(SignalStream, self).__init__()
        
    def write(self, m):
        self.write_signal.emit(m)
        
    def flush(self):
        pass