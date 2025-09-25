from typing import Callable, List, Optional, Tuple
from deap import base, creator, tools, algorithms
import time
import numpy as np
from pkg_resources import invalid_marker
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
                         toolbox.attr_bool, n=self.genome_length) # type: ignore
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
        toolbox.register("evaluate", lambda ind: (self.fitness_function(ind),))
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_prob)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("map", self.pool.map if self.pool else map) # type: ignore

        return toolbox
    
    def _test_toolbox(self, toolbox: base.Toolbox):
        """Test the toolbox by creating and evaluating a sample individual."""
        ind = toolbox.individual() # type: ignore
        print("Sample Individual:", ind)
        fitness = toolbox.evaluate(ind) # type: ignore
        print("Sample Fitness:", fitness)

    def ea_simple_with_elitism(self, population, toolbox, stats=None, 
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
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
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
        toolbox = self._create_toolbox_and_creator()
        self._test_toolbox(toolbox)

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
        pop, log = self.ea_simple_with_elitism(pop, toolbox, stats=stats, halloffame=hof)
        print("Evolution finished.")

        return pop, log, hof

    