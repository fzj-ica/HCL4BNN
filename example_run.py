import numpy as np
from neural_network.binary_nn import NN
from src.genetic_algorithm.algorithm import GeneticAlgorithm




def main():
    # GA parameters
    genome_length = 64  # smaller genome for quick test
    mutation_prob = 0.05
    pop_size = 10       # small population for demo
    ngen = 5            # few generations

    ga = GeneticAlgorithm(
        nn=NN() ,
        genome_length=genome_length,
        mutation_prob=mutation_prob,
        pop_size=pop_size,
        ngen=ngen,
        elite_size=2
    )

    pop, log, hof = ga.run()

    print("\n--- Best Individual ---")
    print("Genome:", hof[0])  
    print("Fitness:", hof[0].fitness.values[0])

if __name__ == "__main__":
    main()
