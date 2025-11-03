from neural_network.binary_nn import NN
from datasets.sipm_dataset import SiPMDataset
from genetic_algorithm.algorithm import GeneticAlgorithm


if __name__ == "__main__":
    # GA parameters
    pop_size = 10       # small population for demo
    ngen = 5            # few generations
    n_samples = 128

    # nn = NN((genome_length, 64, 128, 2), input=SiPMDataset(n_samples=genome_length))
    nn = NN((n_samples, 16, 128, 2), input=SiPMDataset(n_samples=n_samples))

    ga = GeneticAlgorithm(
        nn=nn, 
        pop_size=pop_size,
        ngen=ngen,
        elite_size=2
    )

    pop, log, hof = ga.run()

    print("Best individual size:", int(ga.nn.eval_size(hof[0])*len(hof[0])) )
    print("Best accuracy:", ga.nn.fitness(hof[0])[1])
    print("GA run complete.\n")
    # Example run for the HCL4BNN package


