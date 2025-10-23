from deap import base, creator, tools
import numpy as np

def create_toolbox(genome_length, mutation_prob, eval_func, pool=None):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: 1 if np.random.rand() < 0.8 else 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=genome_length) # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_prob)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("map", pool.map if pool else map)
    return toolbox
