from deap import tools
from .utils import time_elapsed, diversity
import time

def create_stats():
    time_start = time.time()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(x[0]) / len(x[0]))
    stats.register("min", min)
    stats.register("max", max)
    stats.register("diversity", diversity)
    stats.register("time", lambda _: time_elapsed(time_start))
    return stats
