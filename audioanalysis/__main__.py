'''
Created on Dec 13, 2015

@author: justinpalpant
'''
<<<<<<< HEAD
import audiogui
import sys
from PyQt4 import QtGui

def main():
    app = QtGui.QApplication(sys.argv)
    main = audiogui.AudioGUI()
=======
from audioanalysis.audiogui import AudioGUI

def main():
    main = AudioGUI()
>>>>>>> build_exe_and_wheel
    
if __name__ == '__main__':
    main()