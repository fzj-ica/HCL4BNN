# SiPM BNN (GA) Python package

A Python package for training BNNs with Genetic Algorithm.


## Installation

Clone the repository and install dependencies:

```bash
git clone git@icagit.zel.kfa-juelich.de:nc_fpga/bnn/sipm_bnn-python-package.git
cd sipm_bnn-python-package
pip install -e .
```

## Project Structure
``` bash
.
├── docs/                # Sphinx Documentation
├── src/                 # Main source code of the project (Python package)  
├── notebooks/           # Jupyter notebooks for demos and analysis  
├── .gitignore           # Git ignore rules for untracked files and directories  
├── pyproject.toml       # Project metadata, dependencies, and build configuration  
└── README.md            # Project overview and usage instructions  
```

## Usage

There is an example for genearting waveforms in `notebooks/sipm_signals.ipynb`


## Documentation (in progress)

To build local docs:

``` bash
cd docs
make html
```