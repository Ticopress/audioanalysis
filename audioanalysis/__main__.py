'''
Created on Dec 13, 2015

@author: justinpalpant
'''
from audioanalysis.audiogui import AudioGUI
import sys
from PyQt4 import QtGui

def main():
    app = QtGui.QApplication(sys.argv)
    main = AudioGUI()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()