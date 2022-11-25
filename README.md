# Preserving Complex Object-Centric Graph Structures to Improve Machine Learning Tasks in Process Mining

### Jan Niklas Adams, Gyunam Park, Wil van der Aalst

The experiments are based on the Python library [ocpa](https://ocpa.readthedocs.io)

_____________

### Instructions

Use anaconda prompt to create the environment.

``conda env create --file environment.yml``

then unzip the event log and put it in the same directory.

Activate the environment in anaconda prompt

``conda activate eaai``

Go into the repository directory and run 

``python main.py``

This will run all the experiments and produce the output files of measurements.

Note: To use the DGL library, you additionally have to place a config file in 

``~\.dgl\config.json``

containing only the line 
``{"backend":"tensorflow"}``

There might be an error if the environment was installed with a wrong version of a package.
https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly

run 
``pip install protobuf==3.20.*``
in that case.
