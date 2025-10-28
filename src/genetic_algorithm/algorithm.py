from typing import Callable, List, Optional, Tuple
from deap import base, creator, tools, algorithms
import time
import numpy as np

from genetic_algorithm.evaluation import NNEvaluator
from .utils import time_elapsed, diversity
from .toolbox_utils import create_toolbox
from .statistics import create_stats

class GeneticAlgorithm:
    """
    DEAP-based genetic algorithm with elitism, custom statistics and logging.

    Attributes
    ----------
    fitness_function : Callable[[List[int]], float]
        Function to evaluate individual fitness.
    genome_length : int
        Number of genes in each individual.
    nmutbit : int
        Mutated bits per genome.
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

    def __init__(self, 
                 nn, 
                 nmutbit: int = 3,
                 pop_size: int = 200, 
                 cxpb: float = 0.8, 
                 ngen: int = 10, 
                 elite_size: int = 2, 
                 pool: Optional[object] = None):
        self.nn = nn
        self.genome_length = nn.segm[-1]
        self.mutation_prob = nmutbit / self.genome_length
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.ngen = ngen
        self.elite_size = elite_size
        self.pool = pool  # Placeholder for multiprocessing pool if needed

    def evaluate(self, indi):
        acc, div = self.nn.evaluate()
        return acc ,
        

    def _ea_simple_with_elitism(self, population, toolbox, stats=None, 
                            halloffame=None, verbose=True) -> Tuple[List, tools.Logbook]:
        """
        Simple evolutionary algorithm with elitism.

        Args:
            population: initial population
            toolbox: DEAP toolbox with operators registered
            cxpb: crossover probability
            mutpb: mutation probability
            ngen: number of generations
            stats: optional statistics object (e.g., tools.Statistics)
            halloffame: optional HallOfFame object
            elitism_size: number of best individuals to preserve each generation
            verbose: whether to print log output

        Returns:
            (final_population, logbook)
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) # type: ignore

        # Evaluate initial population
        invalid = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid), **record)
        if verbose:
            print(logbook.stream)

        # Evolutionary loop
        for gen in range(1, self.ngen + 1):
            # Select and clone offspring
            offspring = toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            offspring = algorithms.varAnd(offspring, toolbox, self.cxpb, self.mutation_prob)

            # Evaluate invalid offspring
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid) # convert to NN here
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # Add elites back
            elites = tools.selBest(population, self.elite_size)
            offspring.extend(map(toolbox.clone, elites))

            # Update hall of fame
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace population
            population[:] = offspring

            # Record stats
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook


    
    def run(self) -> Tuple[List, tools.Logbook, tools.HallOfFame]:
        """Run the genetic algorithm and return population, logbook, Hall of Fame."""
        # eval_func = NNEvaluator(self.nn)
        toolbox = create_toolbox(self.genome_length, self.mutation_prob, self.evaluate, self.pool)

        print("Create init population...")
        time_start = time.time()

        pop = toolbox.population(n=self.pop_size) # type: ignore

        # Stats & Hall of Fame
        hof = tools.HallOfFame(self.elite_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: sum(x[0])/len(x[0]))
        stats.register("min", min)
        stats.register("max", max)
        stats.register("diversity", diversity)
        stats.register("time", lambda _: time_elapsed(time_start))
        

        print("Start evolution...")
        pop, log = self._ea_simple_with_elitism(pop, toolbox, stats=stats, halloffame=hof)
        print("Evolution finished.")

        return pop, log, hof

    