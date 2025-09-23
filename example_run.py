import numpy as np
from sipm_signals.signals import sipm_therm, nois_therm
from sipm_signals.nn_cam import NN
from genetic_algorithm.algorithm import GeneticAlgorithm

def fitness_function(individual: np.ndarray) -> float:
    """
    Example fitness function using NN with SiPM signals.
    Higher score if NN correctly distinguishes signal vs noise.
    """
    nn = NN(NN=(2, 8, 2))
    nnwgth = nn.conv_from_indi_to_wght(individual)
    nnsummap = nn.conv_from_indi_to_summap(individual)

    Train_D_good = np.array([sipm_therm() for _ in range(2)], dtype=np.uint8)
    Train_D_bad = np.array([nois_therm() for _ in range(2)], dtype=np.uint8)

    correct_good = np.sum([np.all(nn.run_nn(x, (nnwgth, nnsummap)) == 1) for x in Train_D_good])
    correct_bad = np.sum([np.all(nn.run_nn(x, (nnwgth, nnsummap)) == 0) for x in Train_D_bad])

    # Combine scores
    score = correct_good + correct_bad
    return float(score)

def main():
    # GA parameters
    genome_length = 64  # smaller genome for quick test
    mutation_prob = 0.05
    pop_size = 10       # small population for demo
    ngen = 5            # few generations

    ga = GeneticAlgorithm(
        fitness_function=fitness_function,
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
