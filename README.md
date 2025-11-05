# Hardware-Constrained Learning for BNNs - Python package

A Python package for training BNNs with Genetic Algorithm.


## Installation

Clone the repository and install dependencies:

```bash
git clone ...
cd HCL4BNN
pip install -e .

# or (if used for development)

pip install -e .[dev] 
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

## Source Strutcture
```bash
each with __init__.py

src/
├── datasets             #
│   ├── base_dataset.py  # Abstract Class
│   ├── sipm_dataset.py  # SiPMDataset Class, load_data()
│   └── utils.py         #
├── genetic_algorithm    #
│   ├── algorithm.py     # GeneticAlgorithm Class, .run() for optimization 
│   ├── toolbox_utils.py # for deap Toolbox(), rand_indi_segmented() 
│   ├── statistics.py    # for deap Stats() 
│   └── utils.py         # confusion_matrix() 
└── neural_network       #
    ├── base_nn.py       # Abstract Class
    ├── binary_nn.py     # NN Class: .run_nn() for inference, .write_VHDL() for conversion
    ├── utils.py         # calc_...() evaluation-related functions
    └── BNN.vhd          # VHDL package for BNN_entity.vhd

```

## Usage

### `notebooks/example_run.ipynb`
This is an example for a full very coarse training with monitoring and result plots.

### `notebooks/example_run.py`
This is truncated version of above as an non-interactive example for e.g. cluster running.

### `notebooks/sipm_signals.ipynb`
This is a demo of the used simulated SiPM signals.


# TWEPP25
This code was used in a study presented at TWEPP25: [FPGA-Based Real-Time Waveform Classification and Reduction in Particle Detectors](https://indico.cern.ch/event/1502285/contributions/6554519/)


