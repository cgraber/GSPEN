# GSPEN
The reference codebase for the paper "Graph Structured Prediction Energy Networks," appeaing in NeurIPS 2019.

## Dependencies
This code was developed/run using the following versions of the following libraries; in some cases, newer versions may be acceptable. The primary exception is PyTorch - some of the syntax used here changed in newer versions.
* PyTorch 0.4.1
* numpy 1.15.4
* scikit-image 0.13.1
* tensorflow 1.12.0 (This is for training visualization purposes)
* arff (for reading bibtex/bookmarks data files)

Additionally, this code contains a module written in C++; thus, a C++ compiler needs to be installed as well. To make use of LP/ILP inference, you will need a valid Gurobi installation; make sure the GUROBI_HOME environment variable is set to the root dir of this installation.

## Instructions
Make sure to install the library using pip before running (this compiles the C++ code) by running the following command from the root directory:
```
pip install ./
```
If you plan on making your own changes, make sure to include the `-e` flag. Run all scripts from the root directory.

## Data
The (compressed) synthetic words datasets are included in the `data/` directory. The bibtex/bookmarks datasets can be downloaded [here](http://mulan.sourceforge.net/datasets-mlc.html). The script `data/arff2tensor.py` converts the raw data files into the form used by the training code; see `scripts/run_bibtex.sh` to see how to run this script.

