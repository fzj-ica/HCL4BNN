from deap import base, creator, tools, algorithms
import time
import numpy as np
from dateutil.relativedelta import relativedelta

class GeneticAlgorithm:
    # --- Constants ---
    POP_SIZE = 200
    CXPB = 0.8  # Crossover probability
    NGEN = 10  # Number of generations
    ELITE_SIZE = 2

    def __init__(self, fitness_function, genome_length, mutation_prob):
        self.fitness_function = fitness_function
        self.GENOME_LENGTH = genome_length
        self.MUTPB = mutation_prob

        self.pool = None  # Placeholder for multiprocessing pool if needed




    def selTournamentWithFitBracket(self, individuals, k, tournsize, max_fitness = None, min_fitness = None):
        pop = tools.selTournament(individuals, k, tournsize)
        # 2) Replace low-fitness winners
        sub_pop = [ind for ind in pop if (min_fitness and ind.fitness.values[0] < min_fitness) or (max_fitness and ind.fitness.values[0] > max_fitness)]
        return sub_pop


    def eaSimpleWithElitism(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                halloffame=None, verbose=__debug__): # elites = 0,

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            if len(population)>len(offspring):
                fillers = algorithms.varAnd(offspring[:], toolbox, cxpb, mutpb)[:len(population)-len(offspring)]
                offspring.extend(fillers)

                

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
                offspring = offspring[:-len(halloffame)]
                offspring.extend(halloffame.items)
            
            # Replace the current population by the offspring
            population[:] = offspring
            
            # if elites > 0 and elites <= len(population) and elites <= len(hof):
            #     population[-elites:] = halloffame[:]

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook
    
    def _create_toolbox_and_creator(self):
        # --- Fitness and Individual ---
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # --- Toolbox Setup ---
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.binomial, 1, 0.8)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.GENOME_LENGTH)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("map", self.pool.map)


        toolbox.register("evaluate", evaluate)

        toolbox.register("mate", tools.cxTwoPoint)

        toolbox.register("mutate", tools.mutFlipBit, indpb=self.MUTPB)
        toolbox.register("select", tools.selTournament, tournsize=3) 

        return toolbox


    def run(self):
        POP_SIZE = self.POP_SIZE
        CXPB = self.CXPB
        NGEN = self.NGEN
        ELITE_SIZE = self.ELITE_SIZE
        GENOME_LENGTH = self.GENOME_LENGTH
        MUTPB = self.MUTPB
        fitness = self.fitness_function
        pool = self.pool if self.pool else None

        toolbox = self._create_toolbox_and_creator()

        print("Create init population...")
        time_start = time.time()

        pop = toolbox.population(n=POP_SIZE)
        
        hof = tools.HallOfFame(ELITE_SIZE)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", lambda x: sum(x)/len(x))
        stats.register("min", min)
        stats.register("max", max)
        stats.register("diversity", diversity)
        # stats.register("first_indi", first_indi)
        stats.register("time", time_elapsed)

        

        print("Start Algorithm...")
        pop, log = self.eaSimpleWithElitism( #eaSimple( #eaSimpleWithElitism(#algorithms.eaSimple(
            pop, toolbox,
            cxpb=CXPB, 
            mutpb=1.0,  # mutpb=1.0 means each individual is mutated with MUTPB per bit
            ngen=NGEN,
            stats=stats,
            halloffame=hof,
            verbose=True
        )



# --- Evaluation Function ---
def evaluate(fitness, individual):
    return (fitness(individual),)


def time_elapsed(pop, time_start):
    time_diff = time.time() - time_start # seconds
    rd = relativedelta(seconds=time_diff)
    years = f'{int(rd.years)} y, ' if rd.years > 0 else ''
    months = f' {int(rd.months)} mon, ' if rd.months > 0 else ''
    days = f' {int(rd.days)} d' if rd.days > 0 else ''
    hours = f' {int(rd.hours)} h' if rd.hours > 0 else ''
    mins = f' {int(rd.minutes)} m' if rd.minutes > 0 else ''
    secs = f' {int(rd.seconds)} s' if rd.seconds > 0 else ''
    return f'{years}{months}{days}{hours}{mins}{secs}'

def diversity(pop):
    """Return the fraction of unique genotypes in the population."""
    unique = len({ind for ind in pop})
    return unique / len(pop)