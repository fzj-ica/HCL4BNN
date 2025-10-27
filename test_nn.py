#!/usr/bin/env python

test_2n_frames , test_output_size = 50, 2

import neural_network as nn
import datasets.sipm_dataset as sipmds

sipm = sipmds.SiPMDataset(n_frames = test_2n_frames)

SiPM_clsfr = nn.NN(
    layers=(128, 16, 128, test_output_size), input=sipm, description="SiPM_Good_vs_Ugly_Classifier"
)


wf, lb = sipm.gen_Data_Labled()


try:
    assert SiPM_clsfr.run_nn(wf).shape == (test_2n_frames*2, test_output_size)
    print("Pass")
except:
    print(f"got {SiPM_clsfr.run_nn(wf).shape}, expected {(test_2n_frames*2, test_output_size)}")
