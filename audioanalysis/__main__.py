'''
Created on Dec 13, 2015

@author: justinpalpant
'''
import audiogui
import sys
from PyQt4 import QtGui

def main():
    app = QtGui.QApplication(sys.argv)
    main = audiogui.AudioGUI()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()