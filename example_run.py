from typing import List
import numpy as np
from sipm_signals.signals import nois_therm_2bit, sipm_therm, nois_therm, sipm_therm_2bit
from sipm_signals.nn_cam import NN
from genetic_algorithm.algorithm import GeneticAlgorithm

def fitness_function(individual: np.ndarray) -> float:
    """
    Example fitness function using NN with SiPM signals.
    Higher score if NN correctly distinguishes signal vs noise.
    """
    nn = NN(NN=(2, 8, 2))

    Train_D_good = np.array([sipm_therm_2bit() for _ in range(2)], dtype=np.uint8)
    Train_D_bad = np.array([nois_therm_2bit() for _ in range(2)], dtype=np.uint8)

    correct_good = np.sum([np.all(nn.run_nn(x) == 1) for x in Train_D_good])
    correct_bad = np.sum([np.all(nn.run_nn(x) == 0) for x in Train_D_bad])

    # Combine scores
    score = correct_good + correct_bad
    return float(score)

# def fitness(nn: NN, indi):
#     # Zulässige Werte: hohe Fitness = gutes Individuum
#     weights = nn.conv_from_indi_to_wght(indi)
#     summap = nn.conv_from_indi_to_summap(indi)
    
#     Train_D_good = np.array([sipm_therm_2bit() for _ in range(2)], dtype=np.uint8)
#     Train_D_bad = np.array([nois_therm_2bit() for _ in range(2)], dtype=np.uint8)
    
#     # Klassifikations-Outputs für Good/Bad
#     res_g = np.apply_along_axis(nn.run_nn, 1, Train_D_good, weights, summap)
#     res_b = np.apply_along_axis(nn.run_nn, 1, Train_D_bad, weights, summap)

#     # Ziel: möglichst viele Good/Bad korrekt klassifiziert
#     correct_g = np.sum(on_target(res_g, [1, 0]))
#     correct_b = np.sum(on_target(res_b, [0, 1]))

#     # Normalisiere Score
#     score = (correct_g / len(Train_D_good)) + (correct_b / len(Train_D_bad))

#     # Optional: Schreibe zur Kontrolle die Errors aus
#     # errors_g = len(Train_D_good) - correct_g
#     # errors_b = len(Train_D_bad) - correct_b

#     return score


def main():
    # GA parameters
    genome_length = 64  # smaller genome for quick test
    mutation_prob = 0.05
    pop_size = 10       # small population for demo
    ngen = 5            # few generations

    ga = GeneticAlgorithm(
        fitness_function=fitness_function, # type: ignore
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
