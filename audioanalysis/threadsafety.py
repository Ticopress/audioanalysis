'''
Created on Feb 9, 2016

@author: justinpalpant
'''
from PyQt4 import QtCore


class BGThread(QtCore.QThread):
    #QThread also defines started and finished as void-type signals
    
    def __init__(self, fn, name=''):
        super(BGThread, self).__init__()
        self.function = fn
        self.name = name

    def __del__(self):
        self.wait()

    def run(self):
        self.function()

class SignalStream(QtCore.QObject):
    write_signal = QtCore.pyqtSignal(str)
    
    def __init__(self):
        super(SignalStream, self).__init__()
        
    def write(self, m):
        self.write_signal.emit(m)
        
    def flush(self):
        pass