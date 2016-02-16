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

Short-Term Goals
----------------

- Build and deliver the alpha version of the software for trial use
- Clean up text output during training and classification with Keras
- Expose parameters for configuration in some sort of modal pop-up

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