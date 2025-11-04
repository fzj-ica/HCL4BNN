from deap import tools
from .utils import time_elapsed
import time
import numpy as np

def create_stats():
    time_start = time.time()
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max",  lambda x: round(max(x), 4))#max)
    stats.register("avg",  lambda x: round(np.mean(x), 3))
    stats.register("min",  lambda x: round(min(x), 4))#min)
    stats.register("diversity", lambda x: round(diversity(x), 3)) 
    stats.register("time", lambda _: time_elapsed(time_start))
    return stats

def create_multi_stats():
    stats = create_stats()

    # For second objective
    stats2 = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats2.register("max",  lambda x: round(max(x), 3))
    
    # For third objective
    stats3 = tools.Statistics(lambda ind: ind.fitness.values[2])
    stats3.register("max",  lambda x: round(max(x), 3))
    
    # Combine
    mstats = tools.MultiStatistics(match=stats, accuracy=stats2, size=stats3)
    return mstats

def diversity(pop):
    """Return the fraction of unique fitnesses in the population."""
    unique = len({ind for ind in pop})
    return unique / len(pop)
