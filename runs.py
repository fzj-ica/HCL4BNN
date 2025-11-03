#!/usr/bin/env python3
from neural_network.binary_nn import NN
from datasets.sipm_dataset import SiPMDataset
from genetic_algorithm.algorithm import GeneticAlgorithm

import datetime
import pickle

if __name__ == "__main__":
    n_samples = 128
    n_frames = 200

    # GA parameters
    pop_size = 1000  
    ngen = 50 
    mutation_nbit = 5
    tourn_size = int(0.01*pop_size)
    cxpb = 0.008
    cxpb_bit = 0.5
    elite_size = 2

    # machine dependent
    pool_nproc = 90

    inp = SiPMDataset(n_samples = n_samples, n_frames = n_frames)

    nn = NN((inp.n_samples, 16, 128, len(inp.LABLES)), input=inp, description="Good_vs_Ugly")
    
    print(str(NN))

    ga = GeneticAlgorithm(
        nn = nn, 
        pop_size = pop_size,
        ngen = ngen,
        elite_size = elite_size,
        tourn_size = tourn_size,
        cxpb = cxpb,
        cxpb_bit = cxpb_bit,
        nmutbit = mutation_nbit,
        pool_nproc = pool_nproc
    )

    for run in range(10):
        print(f"Run Nr {run}")
        pop, log, hof = ga.run()

        print("Best individual size:", ga.nn.eval_size(hof[0]))
        print("Best fitness:", ga.nn.evaluate(hof[0])[0])
        print("GA run complete.\n")
        # Example run for the HCL4BNN package

        now_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        NN_descr = "-".join( [str(i) for i in nn.NN] ) + f"__bitlen_inp-neur-wght_-{nn.inp_len}-{nn.neur_len}-{nn.wght_len}"
        filename = f"{now_timestamp_str}_log__run{run}__{nn.description}__{NN_descr}.pkl"

        log.hof = hof

        with open(filename, 'wb') as file:
            pickle.dump(log, file)
            print(f"{filename} written: \n*log* obj, containing log.hof (HallOfFame) and log.genlog (Best per Generation)")


