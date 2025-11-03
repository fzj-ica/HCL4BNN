#!/usr/bin/env python3
import sys
from pathlib import Path

from neural_network.binary_nn import NN
from datasets.sipm_dataset import SiPMDataset

import datetime
import pickle

def process_file(filename: Path):
    print(f"Processing {filename}")
    log = pickle.load(open(filename,'rb'))
    indi = log.genlog[-1]

    n_samples = 128
    n_frames = 200

    inp = SiPMDataset(n_samples = n_samples, n_frames = n_frames)

    nn = NN((inp.n_samples, 16, 128, len(inp.LABLES)), individual=indi, input=inp, description="Good_vs_Ugly")
    nn.write_VHDL()
    

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file1> [file2 ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.exists() and path.is_file():
            process_file(path)
        else:
            print(f"Skipping invalid file: {path}")

if __name__ == "__main__":
    main()

