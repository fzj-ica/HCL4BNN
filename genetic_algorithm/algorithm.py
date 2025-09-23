from typing import Callable, List, Optional, Tuple
from deap import base, creator, tools, algorithms
import time
import numpy as np
from .utils import time_elapsed, diversity

class GeneticAlgorithm:
    """
    DEAP-based genetic algorithm with elitism, custom statistics and logging.

    Attributes
    ----------
    fitness_function : Callable[[List[int]], float]
        Function to evaluate individual fitness.
    genome_length : int
        Number of genes in each individual.
    mutation_prob : float
        Mutation probability per bit.
    pop_size : int
        Population size.
    cxpb : float
        Crossover probability.
    ngen : int
        Number of generations.
    elite_size : int
        Number of individuals in Hall of Fame.
    """

    def __init__(self, fitness_function: Callable[[List[int]], float], 
                 genome_length: int, 
                 mutation_prob: float,
                 pop_size:int = 200, 
                 cxpb: float = 0.8, 
                 ngen: int = 10, 
                 elite_size: int = 2, 
                 pool: Optional[object] = None):
        self.fitness_function = fitness_function
        self.genome_length = genome_length
        self.mutation_prob = mutation_prob
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.ngen = ngen
        self.elite_size = elite_size
        self.pool = pool  # Placeholder for multiprocessing pool if needed

    def _create_toolbox_and_creator(self) -> base.Toolbox:
        """Create DEAP toolbox and creator objects."""
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax) # type: ignore

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", lambda: 1 if np.random.rand() < 0.8 else 0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, # type: ignore
                         toolbox.attr_bool, self.genome_length) # type: ignore
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
        toolbox.register("evaluate", lambda ind: (self.fitness_function(ind),))
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_prob)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("map", self.pool.map if self.pool else map) # type: ignore

        return toolbox

    
    def run(self) -> Tuple[List, object, object]:
        """Run the genetic algorithm and return population, logbook, Hall of Fame."""
        toolbox = self._create_toolbox_and_creator()

        print("Create init population...")
        time_start = time.time()

        pop = toolbox.population(n=self.pop_size) # type: ignore
        hof = tools.HallOfFame(self.elite_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", lambda x: sum(x)/len(x))
        stats.register("min", min)
        stats.register("max", max)
        stats.register("diversity", diversity)
        stats.register("time", lambda _: time_elapsed(time_start))
        

        def ea_simple_with_elitism(population, toolbox):
            log = tools.Logbook()
            log.header = ['gen','nevals'] + stats.fields # type: ignore
            for gen in range(self.ngen+1):
                invalid = [ind for ind in population if not ind.fitness.valid]
                for ind, fit in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
                    ind.fitness.values = fit
                if gen < self.ngen:
                    offspring = algorithms.varAnd(population, toolbox, self.cxpb, self.mutation_prob)
                    offspring = toolbox.select(offspring, len(population))
                    population[:] = offspring
                log.record(gen=gen, nevals=len(invalid), **stats.compile(population))
                print(log.stream)
            return population, log
        
        pop, log = ea_simple_with_elitism(pop, toolbox)
        hof.update(pop)
        return pop, log, hof

    # @staticmethod
    # def sel_tournament_wiht_fit_bracket(individuals, k, tournsize, max_fitness=None, min_fitness=None) -> List:
    #     """Select via tournament, but filter by min/max fitness if set."""
    #     pop = tools.selTournament(individuals, k, tournsize)
    #     # 2) Replace low-fitness winners
    #     sub_pop = [ind for ind in pop 
    #                if (min_fitness and ind.fitness.values[0] < min_fitness) 
    #                or (max_fitness and ind.fitness.values[0] > max_fitness)]
    #     return sub_pop








