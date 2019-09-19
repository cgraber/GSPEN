# GSPEN
The reference codebase for the paper "Graph Structured Prediction Energy Networks," appeaing in NeurIPS 2019.

## Dependencies
This code was developed/run using the following versions of the following libraries; in some cases, newer versions may be acceptable. The primary exception is PyTorch - some of the syntax used here changed in newer versions.
* PyTorch 0.4.1
* numpy 1.15.4
* scikit-image 0.13.1
* tensorflow 1.12.0 (This is for training visualization purposes)

Additionally, this code contains a module written in C++; thus, a C++ compiler needs to be installed as well. To make use of LP/ILP inference, you will need a valid Gurobi installation; make sure the GUROBI_HOME environment variable is set to the root dir of this installation.

## Instructions
Make sure to install the library using pip before running (this compiles the C++ code) by running the following command from the root directory:
```
pip install ./
```
If you plan on making your own changes, make sure to include the `-e` flag. Run all scripts from the root directory.

