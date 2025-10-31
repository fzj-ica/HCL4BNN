from typing import List, Optional, Tuple
from deap import tools, algorithms
import time

from .toolbox_utils import create_toolbox
from .statistics import create_multi_stats

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
        fit, div, eval_size = self.nn.evaluate(indi)
        return fit * div, fit, eval_size
        

    def _ea_simple_with_elitism(self, population, toolbox, stats=None, 
                            halloffame=None, elites=0, verbose=True) -> Tuple[List, tools.Logbook]:
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
        logbook.genlog = [] # type: ignore

        # Evaluate initial population
        invalid = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)
            if len(halloffame) < elites:
                elites = len(halloffame)
        

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid), **record)
        if verbose:
            print(logbook.stream)

        # Evolutionary loop
        for gen in range(1, self.ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, int(len(population)*1.01))
            # offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            offspring = algorithms.varAnd(offspring, toolbox, self.cxpb, self.mutation_prob)

            # Vary the pool of individuals
            if len(population) > len(offspring):
                fillers = algorithms.varAnd(offspring[:], toolbox, self.cxpb, self.mutation_prob)[:len(population) - len(offspring)]
                offspring.extend(fillers)
            if elites > 0:
                old_best_idx = tools.selBest(population, elites)
                for best_i in old_best_idx:
                    del best_i.fitness.values
                offspring.extend(old_best_idx)

            # Evaluate invalid offspring
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid) # convert to NN here
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)


            # Replace the current population by the offspring, remove worst if too long e.g. due to elitism
            if len(offspring) > len(population):
                worst_idx = tools.selWorst(offspring, len(offspring)-len(population))
                for worst_one in worst_idx: 
                    offspring.remove(worst_one)

            population[:] = offspring[:]

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid), **record)
            logbook.genlog.append(tools.selBest(population, 1)[0]) # type: ignore
            if verbose:
                print(logbook.stream)

        return population, logbook


    
    def run(self) -> Tuple[List, tools.Logbook, tools.HallOfFame]:
        """Run the genetic algorithm and return population, logbook, Hall of Fame."""
        # eval_func = NNEvaluator(self.nn)
        toolbox = create_toolbox(self.mutation_prob, self.evaluate, self.nn, pool=self.pool)

        print("Create init population...")
        time_start = time.time()

        pop = toolbox.population(n=self.pop_size) # type: ignore

        # Stats & Hall of Fame
        hof = tools.HallOfFame(self.elite_size)

        mstats = create_multi_stats()
        

        print("Start evolution...")
        pop, log = self._ea_simple_with_elitism(pop, toolbox, stats=mstats, halloffame=hof)
        print("Evolution finished.")

        return pop, log, hof

    