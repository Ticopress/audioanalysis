'''
Created on Nov 16, 2015

@author: justinpalpant
'''

import Tkinter, tkFileDialog
import matplotlib
import freqanalysis
from logging import root

class AudioGUI(Tkinter.Frame):

    def __init__(self, root):
        #Add some widgets
        root.minsize(width=500, height=200)
        
        Tkinter.Frame.__init__(self, root, width=500, height=200)
        
        #Define the specifications for opening files
        self.file_opt = options = {}
        options['defaultextension'] = '.wav'
        options['filetypes'] = [('WAV files', '.wav')]
        options['initialdir'] = '/Users/new/Documents/JarvisLab/audioanalysis/'
        options['parent'] = root
        options['title'] = 'Select an audio file'
        self.grid()
        self.createWidgets()
    
    def createWidgets(self):
        self.file_button = Tkinter.Button(self, text='Select File', command=self.browsefilename)
        self.file_button.grid(row=0, column=0, padx=5)
        self.filename_box = Tkinter.Text(self, height=1, highlightbackground="black", highlightthickness=1)
        self.filename_box.grid(row=0, column=1, padx=5)
        
    def browsefilename(self):
        filename = tkFileDialog.askopenfilename(**self.file_opt)
        print filename
        self.filename_box.insert("1.end", filename)
    
        
def main():
    root = Tkinter.Tk()
    AudioGUI(root)
    root.title("Audio Analysis")
    root.attributes("-topmost", True)
    root.mainloop()
           

    pass


if __name__ == '__main__':
    main()
