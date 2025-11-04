from typing import List, Optional, Tuple
from deap import tools, algorithms
import multiprocessing
import signal
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
    pop_size : int
        Population size.
    nmutbit : int
        Mutated bits per genome.
    mutation_prob : float
        Mutation probability per bit.
    mutation_prob_indi : float
        Mutation probability per individual.
    cxpb : float
        Crossover probability. (TODO in pop vs. per bit)
    ngen : int
        Number of generations.
    elite_size : int
        Number of individuals in Hall of Fame.
    """

    def __init__(self, 
                 nn, 
                 nmutbit: int = 3,
                 pop_size: int = 200, 
                 tourn_size: int = 3, 
                 cxpb: float = 0.8, 
                 cxpb_bit: float = 0.5, 
                 ngen: int = 10, 
                 elite_size: int = 2, 
                 pool_nproc: int = None
                 ):
        self.nn = nn
        self.genome_length = nn.segm[-1]
        self.mutation_prob = nmutbit / self.genome_length
        self.mutation_prob_indi = 1.0
        self.cxpb = cxpb
        self.cxpb_bit = cxpb_bit
        self.tourn_size = tourn_size
        self.pop_size = pop_size
        self.ngen = ngen
        self.elite_size = elite_size
        self.pool_nproc = pool_nproc 
    

    def evaluate(self, indi):
        acc, div, mat, eval_size = self.nn.evaluate(indi)
        return mat, acc, eval_size
        # return acc, mat * div, eval_size # blocked by broken-clock
        

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
        # logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) # type: ignore
        logbook.header = (stats.fields if stats else []) # type: ignore
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
            try:
                # Select the next generation individuals
                offspring = toolbox.select(population, int(len(population)*1.01))
                # offspring = list(map(toolbox.clone, offspring))
    
                # Apply crossover and mutation
                offspring = algorithms.varAnd(offspring, toolbox, self.cxpb, self.mutation_prob_indi)
    
    
                # Vary the pool of individuals
                if len(population) > len(offspring):
                    fillers = algorithms.varAnd(offspring[:], toolbox, self.cxpb, self.mutation_prob_indi)[:len(population) - len(offspring)]
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
            except KeyboardInterrupt:
                print(f"\nInterrupted by user at gen: {gen}")
                if toolbox.pool:
                    toolbox.pool.terminate()
                    toolbox.pool.join()
                break

        return population, logbook


    
    def run(self) -> Tuple[List, tools.Logbook, tools.HallOfFame]:
        """Run the genetic algorithm and return population, logbook, Hall of Fame."""
        # eval_func = NNEvaluator(self.nn)
        if self.pool_nproc and self.pool_nproc > 1:
            pool = multiprocessing.Pool(self.pool_nproc, initializer=init_worker)
        else:
            pool = None
        print(f"Start GA, run for {self.ngen} generations ...")
        print(f"    pop={self.pop_size}, cxpb={self.cxpb}, cxpb_bit={self.cxpb_bit}, mutpb/genome={self.mutation_prob*self.genome_length}, tourn_size={self.tourn_size}, elite_size={self.elite_size}, pool={self.pool_nproc} ")
        
        toolbox = create_toolbox(self.mutation_prob, self.cxpb_bit, self.tourn_size, self.evaluate, self.nn, pool=pool)

        print("Create init population...")
        time_start = time.time()

        pop = toolbox.population(n=self.pop_size) # type: ignore

        # Stats & Hall of Fame
        hof = tools.HallOfFame(max(2,int(0.01*self.pop_size*self.ngen)))

        mstats = create_multi_stats()
        

        print("Start evolution...")
        pop, log = self._ea_simple_with_elitism(pop, toolbox, stats=mstats, halloffame=hof)
        print("Evolution finished.")

        return pop, log, hof

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

