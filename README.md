What Is It?
-----------

This project will provide tools for bird song motif discovery and analysis to the Jarvis Lab at Duke University

Erich Jarvis Lab - Neurobiology of Vocal Communication
Website: http://jarvislab.net/


Current State
-------------

The software is in the alpha stage of development.

- A nearly complete, multithreaded GUI is implemented using PyQt4
- A model for storing and processing audio data is implemented
- Classification of audio using Keras neural nets is implemented, including training
- Manual classification of audio for training or testing
- Automated clipping of classified song bouts
- A first alpha version has been released and is being tests

Changelog
-------------
**Version 0.1.1**
- Added export and import of parameters as text files
- Added ability to simultaneously serialize all files to disk
- Improved WAV file loading system so that no files would be loaded with small slices split off of them.  Previously, loading a 300.01 second-long WAV file with a file split of 300 would result
in a 300 second file and a 0.01 second file.  Now, if the final split is less than one second, the split rule will be broken and that section will be merged into the last SongFile.

**Version 0.1.0**
- Initial alpha release

Short-Term Goals
----------------
- Expose parameters for configuration in some sort of modal pop-up
- Create a list of workable issues on GitHub based on first test case

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