What Is It?
-----------

This project will provide tools for bird song motif discovery and analysis to the Jarvis Lab at Duke University

Erich Jarvis Lab - Neurobiology of Vocal Communication
Website: http://jarvislab.net/


Current State
-------------

The software is in the initial stages of development.

- I have selected a GUI framework for the project - it will use PyQt4, until further notice.
- I have designed the first GUI using Qt Designer.
- I have figured out how to plot spectrograms in the main window efficiently
- I have separated Model and View functions into several classes
- I have linked some of the UI controls to their functions and tested those functions
- I have subclassed the NavigationToolbar2 and customized several navigation functions to make them convenient for spectrogram navigation
- I have implemented Model subclasses for storing song data
- I have implemented efficient chunking of large datasets and efficient data display
- I have implemented all menu callbacks in the final GUI, including import and export of motifs and trained neural nets

Short-Term Goals
----------------

- Implement training of the neural net from a single manually classified songfile
- Figure out how to build the application with all dependencies in a way that actually will run and not yield annoying ImportErrors
- Implement classification of a group of songfiles with a trained neural net
- Extra cleanup/feature-adds: right-click delete songfiles or motifs from the list, a 'deactivate/clear' function in SpectrogramCanvas, ability to manually split motifs, more intelligent motif splitting/detection

Copyright and License
---------------------

For the complete copyright and licensing information see LICENSE


----------------------------------------------
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