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
from PyQt4.uic import loadUiType
 
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas)

Ui_MainWindow, QMainWindow = loadUiType('main.ui')

class AudioGUI(Ui_MainWindow, QMainWindow):

    def __init__(self, ):
        super(AudioGUI, self).__init__()
        self.setupUi(self)




def main():
    import sys
    from PyQt4 import QtGui, QtCore
     
    app = QtGui.QApplication(sys.argv)
    main = AudioGUI()
    main.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()