'''
Created on Dec 13, 2015

@author: justinpalpant
'''
import subprocess
import platform

def run_as_package():
    """This method is called when you use python -m {package}
    
    It requires that all the important packages (including Qt, et al.) be
    available in the Python path, correctly installed.
    """
    print 'Please select the module to run:'
    print '1. Main Audio GUI'
    print '0: Quit'
    
    key = int(input('Make a selection: '))
    
    if key == 1:
        import audiogui

def run_as_executable():
    """This method is the one put on the PATH by pip install
    
    It assumes that pip install completes successfully, but does not need Qt
    to be installed, or any other package, because it just finds the built
    executable and calls that.  It selects the correct executable for the
    platform and calls it, nothing else.
    
    Coincidentally, that executable is just a built version of this __main__.py,
    run as __name__==__main__ and executing the run_as_package() function above
    
    Funny how things work out, eh?
    """
    print 'Trying to run the executable, I see?'
    sys = platform.system()
    
    if sys == 'Windows':
        print 'You have Windows!  Using .exe'
    elif sys == 'Darwin':
        print 'You have Mac!  Using executable'
    else:
        print "Idk what you have but I can't run it, sorry"
   

if __name__ == '__main__':
    run_as_package()