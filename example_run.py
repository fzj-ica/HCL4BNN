from neural_network.binary_nn import NN
from datasets.sipm_dataset import SiPMDataset
from genetic_algorithm.algorithm import GeneticAlgorithm

def required_genome_length(self):
    total = 0
    for i in range(len(self.NN) - 1):
        total += self.NN[i] * self.NN[i + 1] * 2  # 2 bits pro Gewicht
    return total


if __name__ == "__main__":
    # GA parameters
    genome_length = 128  # smaller genome for quick test
    mutation_prob = 0.05
    pop_size = 10       # small population for demo
    ngen = 5            # few generations

    nn = NN((genome_length, 64, 128, 2), input=SiPMDataset(n_samples=genome_length))

    ga = GeneticAlgorithm(
        nn=nn, 
        genome_length=genome_length,
        mutation_prob=mutation_prob,
        pop_size=pop_size,
        ngen=ngen,
        elite_size=2
    )

    pop, log, hof = ga.run()

    print("Best individual:", hof[0])
    print("Best fitness:", ga.nn.fitness(hof[0]))
    print("GA run complete.\n")
    # Example run for the HCL4BNN package
    print(f"log: \n{log}")
    print(f"pop: \n{pop}")


