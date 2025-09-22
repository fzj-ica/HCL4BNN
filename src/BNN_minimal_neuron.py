# Define Neuron behaviour
import numpy as np
import random
import hashlib
from deap import base, creator, tools, algorithms




def BNN_plus(a,b):
    # lookup table
    plus_d = {
        frozenset({0} )    :  0,
        frozenset({1} )    :  1,
        frozenset({2} )    :  2,
        frozenset({3} )    :  3,
        frozenset({0,1} )  :  0,
        frozenset({0,2} )  :  1,
        frozenset({0,3} )  :  1,
        frozenset({1,2} )  :  2,
        frozenset({1,3} )  :  2,
        frozenset({2,3} )  :  3,    
    }
    return plus_d[frozenset({a,b} )]


# print ( 1 if random.random() > 0.2 else 0 );
def skw():
    return 1 if random.random() > 0.2 else 0


def hash_array(arr):
    # Convert the array to bytes
    arr_bytes = np.array(arr).tobytes()
    
    # Create a new SHA-256 hash object
    hash_object = hashlib.sha256()
    
    # Update the hash object with the array bytes
    hash_object.update(arr_bytes)
    
    # Get the hexadecimal representation of the hash value
    hash_value = hash_object.hexdigest()
    
    return hash_value

# generating random individuals
def rand_indi(npSegm):
    return np.random.binomial(1, 0.65, size=npSegm[-1])
def zero_indi(npSegm):
    return np.zeros(npSegm[-1])

# # Define NN
# TODO: Better structure
class NN_class:
    # in bit
    neur_len = 2
    bias_len = 2
    wght_len = 2


    NN = (2048, 64, 32, 2)

    keep_l = [None]*len(NN)


    # # Make Indi
    npNN = np.array(NN)
    npSegm = np.cumsum( np.concatenate( [[0], npNN[:-1]* npNN[1:] * wght_len ]) )
    # TODO: ?

    # generating random individuals
    def rand_indi(self):
        return np.random.binomial(1, 0.65, size=self.npSegm[-1])
    def zero_indi(self):
        return np.zeros(self.npSegm[-1])


    def conv_from_indi_to_wght(self, indi):
        arr =  np.array( indi ).astype(np.int8)
        wghtlist = [] 
        for i,s in enumerate( zip( self.npSegm [:len(self.npNN)+1],  self.npSegm [1:len(self.npNN)] ) ):
            iwght = arr[slice(*s)].reshape([self.npNN[i+1], self.npNN[i],2])
            iwght_2bit = (iwght[:,:,0]) | (iwght[:,:,1] << 1)
            wghtlist.append(iwght_2bit)
        return wghtlist



    def conv_from_indi_to_bias(self, indi):
        arr =  np.array( indi ).astype(np.int8) 
        biaslist = [] 
        for i,s in enumerate( zip( self.npSegm [len(self.npNN)-1:-1],  self.npSegm [len(self.npNN):] ) ):
            biasi = arr[slice(*s)].reshape(self.npNN[i+1],2)
            biasi = (biasi[:,0]) | (biasi[:,1] << 1)
            biaslist.append( biasi )
        return biaslist


    def conv_from_indi_to_summap(self, indi):
        summap = []
        for i in range(0,len(self.NN)-1):
            nonzwght = (self.npNN[i] - np.uint8(self.conv_from_indi_to_wght(indi)[i] == 0).sum(axis = 1))
            summap.append( np.uint16( nonzwght[:,np.newaxis] * [0.5, 1.5, 2.5]) )
        return summap


# # Generate NN parameters
    def generate_parameters(self):
        Ws = [self.NN[i]*self.NN[i+1] for i in range(len(self.NN)-1)]
        Bs = [self.NN[i] for i in range(1,len(self.NN))] # Max 0..3 * N_inputs

        NNbits = sum(Ws) * self.wght_len + sum(Bs) * self.bias_len

    # NNwgth = self.conv_from_indi_to_wght(indi)
    # NNsummap = self.conv_from_indi_to_summap(indi)


    verbose = True






    # ----------------------------------------------------------------------
    # _CAM_LUT = np.array([ # with input 0, 1, 2 ,3
    #     3, 2, 1, 0,   # bias = 0, n = 0..3
    #     0, 0, 1, 2,   # bias = 1, n = 0..3
    #     0, 1, 2, 3,   # bias = 2, n = 0..3
    #     1, 2, 3, 3    # bias = 3, n = 0..3
    # ], dtype=np.uint8)          # shape (16,)

    _CAM_LUT = np.array([ # with input 0, 1, 2 ,3
        0, 0, 0, 0,   # bias = 0, n = 0..3
        0, 1, 2, 3,   # bias = 1, n = 0..3
        1, 2, 3, 3,   # bias = 2, n = 0..3
        3, 2, 1, 0,   # bias = 3, n = 0..3
    ], dtype=np.uint8)          # shape (16,)



    def CAM_neur(self, neur: np.ndarray, wght: np.ndarray) -> np.ndarray: 
        # The index into the 4‑bit LUT is   idx = a*4 + b  == (a << 2) | b
        idx = (wght << 2) | neur          # still uint8, range 0‑15
        return self._CAM_LUT[idx]            # vectorised lookup, result shape = (n_next,)


    def layer_inp(self, input_i):
        return np.uint8(input_i+1)

    def output_o(self, layer_nL):
        return (layer_nL>=2).astype(np.uint8)


    def rand_inp(self, N):
        return np.array(list(np.binary_repr(random.getrandbits( N ), width=N)), dtype=np.uint8) 

    # digitize_func = np.vectorize(lambda x, y: np.digitize(x, y), signature='(n),(n,m)->(n)')

    # --------------------------------------------------------------
    # Vectorised digitise  (equivalent to np.digitize(..., side='right'))
    # --------------------------------------------------------------
    def digitise_rows(self, neuron_sum: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        # Expand dimensions so broadcasting works:
        #   neuron_sum[..., None]  → (R, C, 1)
        #   bin_edges[:, None, :]  → (R, 1, B)
        # Comparison is then performed row‑wise.
        greater_equal = neuron_sum[..., None] >= bin_edges[:, None, :]
        # print(greater_equal.shape)

        # For each value we count how many edges are ≤ that value.
        # The count is exactly the digitise index when side='right'.
        idx = greater_equal.sum(axis=2)          # shape (R, C)
        # print(idx.shape)

        return idx

    def calc_layer(
        self,
        layer_pre: np.ndarray,
        layer_pre_idx: int,
        NNwgth: list[np.ndarray],
        NNsummap: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute the activations of the next layer in a simple feed-forward network.

        Returns
        -------
        np.ndarray, dtype uint8, shape (n_next,)
            The digitised activations of the next layer.
        """
        if verbose:
            print("* Input:")
            print(layer_pre.shape)
            time.sleep(0.001)
        # ------------------------------------------------------------------ #
        weights = NNwgth[layer_pre_idx]
        weighted = self.CAM_neur(layer_pre, weights)
        if verbose:
            print("* weighted:")
            print(weighted.shape)
            time.sleep(0.001)

        # ------------------------------------------------------------------ #
        neuron_sum = weighted.sum(axis=1)               # shape (n_next,)
        if verbose:
            print("* neuron_sum:")
            print(neuron_sum.shape)

        # ------------------------------------------------------------------ #
        bin_edges = NNsummap[layer_pre_idx]            # shape (n_next, n_bins)
        if verbose:
            print("* bin_edges:")
            print(bin_edges.shape)

        # ReLu_2bit = digitize_func(neuron_sum, bin_edges)
        # ReLu_2bit = np.apply_along_axis(lambda a, b: np.digitize(a, b), 1, neuron_sum, bin_edges)
        ReLu_2bit = np.array( [ np.digitize(neuron_sum[i], b) for i,b in enumerate(bin_edges) ] )
        # ReLu_2bit = (neuron_sum[..., None] >= bin_edges[:, None, :]).sum(axis=2)
        # ReLu_2bit = digitise_rows(neuron_sum, bin_edges)

        if verbose:
            print("* ReLu_2bit:")
            print(list(bin_edges)[:10], ReLu_2bit.shape)

        neurons_next = ReLu_2bit

        return neurons_next


    bin_edges = np.array([[ 28,  84, 140],
            [ 28,  85, 142],
            [ 31,  102, 155],
            [ 28,  85, 142],
            [ 26,  79, 132]])
    neuron_sum = np.array( [[30,10,100],[20,10,100],[20,10,100],[40,10,100],[1,10,100]] )


    # ### Run NN: recursive layer calc
    def run_NN(self, inp, pars):
        # NN, NNwgth, NNbias, NNsummap = pars
        NNwgth,NNsummap  = pars
        layer_ni = self.layer_inp(inp)
        for i in range(0,len(self.NN)-1):
            layer_ni = self.calc_layer( layer_ni, i, NNwgth,  NNsummap)
        return self.output_o(layer_ni)
    
    def get_train_d(self):
        Train_D = np.empty((10000, self.NN[0]), dtype=np.uint8)
        for i, row in enumerate(SiPM_Therm()):
            Train_D[i, :] = row




# y = SiPM_Therm()
# for _ in range(10):
#     NNwgth = rand_NNwgth()
#     NNbias = rand_NNbias()
#     print( run_NN(y, pars=( NNwgth, NNbias)) )
    

    # # Calc fitness
    def get_train_d_good(self):
        Train_D_good = np.empty((2, self.NN[0]), dtype=np.uint8)
        for i in range(len(Train_D_good)):
            Train_D_good[i,:] = SiPM_Therm()
        return Train_D_good
    
    def get_train_d_bad(self):
        Train_D_bad = np.empty((2, self.NN[0]), dtype=np.uint8)
        # for i, row in enumerate(Nois_Therm()):
        #     Train_D_bad[i, :] = row
        for i in range(len(Train_D_bad)):
            Train_D_bad[i,:] = Nois_Therm()
        return Train_D_bad
    
    
    
    def fitness(self, indi):
        Train_D_good = self.get_train_d_good()
        Train_D_bad = self.get_train_d_bad()
        # lower = better
        Ngood =np.float32(len(Train_D_good))
        Nbad =np.float32(len(Train_D_bad))
        res_g = np.apply_along_axis(func1d=self.run_NN, axis=1, arr=Train_D_good, pars=(self.conv_from_indi_to_wght(indi),  self.conv_from_indi_to_summap(indi)))

        how_good = on_target(res_g,[1,0])
        res_b = np.apply_along_axis(func1d=self.run_NN, axis=1, arr=Train_D_bad, pars=(self.conv_from_indi_to_wght(indi), self.conv_from_indi_to_summap(indi) ))
       
        how_bad = on_target(res_b,[0,1])

        return how_good + how_bad # + np.sum(np.int8(res_g==res_b))


par_names = ['tfast', 'tslow','rfast', 'rslow']

def IsG(t, par):
    amp, tfast, tslow = par
    tslow = max(tfast*1.1, tslow)
    Islow = (np.exp(-t/tslow))
    Ifast = (1-np.exp(-t/tfast))
    # IsG = Ifast
    IsG = amp*(Islow + Ifast-1)
    return np.where(t < 0, 0, IsG) #IsG if t>0 else 0 

ADC_bits = 12
ADC_smpls = 128


ADC_MAX  = 2**ADC_bits - 1
ADC_ZERO = 2**(ADC_bits-1) - 0
ADC_MIN = 0

trace0 = [ int(ADC_ZERO) ]* ADC_smpls


def SiPM_wf():
    chrg = random.uniform(0.1,1)
    amp =  chrg * (ADC_MAX-ADC_ZERO) * 1.3
    par = [amp,4,14*chrg*random.uniform(1.0,1.4)]
    Dx = np.linspace(-1,ADC_smpls-1,500)
    Dy = IsG(Dx,par) + ADC_ZERO
    Dx = Dx - Dx[0]
    return Dx, Dy

# TODO: is declarated in the bottom too
# def SiPM_ADC():
#     chrg = random.uniform(0.1,1)
#     amp =  chrg * (ADC_MAX-ADC_ZERO) * 1.3
#     par = [amp,4,14*chrg*random.uniform(1.0,1.4)]
#     Dx = np.linspace(-1,ADC_smpls-2,ADC_smpls)
#     Dy = IsG(Dx,par) + ADC_ZERO
#     Dx = Dx - Dx[0]
#     Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
#     return Dx, Dy


def uint12_to_therm(values, num_bins = 16):
    values = np.asarray(values, dtype=np.uint16)
    thresholds = np.arange(0,2**11,2**11/num_bins) # 2**11+1 for endpoint
    thermometer = (values[:, None] > thresholds).astype(np.uint8)
    return thermometer#, thresholds, values

def SiPM_Therm():
    x,y = SiPM_ADC()
    return np.ravel( uint12_to_therm( y + 128 - ADC_ZERO ) )


def Nois(t, par):
    amp, tfast = par
    Ifast = (np.cos(-t/tfast))
    v = amp*(Ifast)
    return np.where(t < 0, 0, v) 


# TODO: Is declarated in the bottom too
# def Nois_ADC():
#     chrg = random.uniform(0.8,1)
#     amp =  chrg * (ADC_MAX*0.01)
#     par = [amp,0.001*chrg*random.uniform(1.0,1.4)]
#     Dx = np.linspace(-1,ADC_smpls-2,ADC_smpls)
#     Dy = Nois(Dx,par) + ADC_ZERO
#     Dx = Dx - Dx[0]
#     Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
#     return Dx, Dy


def Nois_Therm():
    x,y = Nois_ADC()
    return np.ravel( uint12_to_therm( y + 128 - ADC_ZERO ) )




def SiPM_ADC(): # Debug
    Dx = np.linspace(-1,ADC_smpls-2,ADC_smpls)
    Dy = np.zeros_like(Dx)
    Dy[Dx >= 64] = ADC_ZERO
    Dy[Dx < 64] = ADC_MAX
    # Dx = Dx - Dx[0]
    Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
    return Dx, Dy

def Nois_ADC(): # Debug
    Dx = np.linspace(-1,ADC_smpls-2,ADC_smpls)
    Dy = np.zeros_like(Dx)
    Dy[Dx >= 64] = ADC_MAX
    Dy[Dx < 64] = ADC_ZERO
    # Dx = Dx - Dx[0]
    Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
    return Dx, Dy


    



    # wird überschrieben
    # def softmax(z):
    #     e = np.exp(z - np.max(z))
    #     return e / e.sum()

def softmax(z):
    # e = np.exp(z - np.max(z))
    return z / z.sum()


def on_target(probe, target):
    p = probe # / np.sum(logits)   #softmax(logits)
    t = np.array(target) # / np.sum(target)
    return np.sum(np.int8(p==t))


# l = np.array( [1,0] ,dtype=np.float32)
# t0= np.array( [1,0] ,dtype=np.float32)
# t1= np.array( [0,1] ,dtype=np.float32)
# t2= np.array( [0,0] ,dtype=np.float32)
# t3= np.array( [1,0] ,dtype=np.float32)

# # fitness_from_logits(l,0), fitness_from_logits(l,1), fitness_from_logits(l,2)



def ravel_ragged(arrays):
    """Flatten a heterogeneous list of 2-D arrays without copying."""
    # Ensure each sub-array is C-contiguous (copy only if needed)
    parts = [a.ravel() if a.flags['C_CONTIGUOUS'] else np.ascontiguousarray(a).ravel() for a in arrays]
    flat = np.concatenate(parts)               # 1-D view of all data
    shapes = [a.shape for a in arrays]
    offsets = np.concatenate(([0], np.cumsum([r*c for r, c in shapes])[:-1]))
    return flat, shapes, offsets

def unravel_ragged(flat, shapes, offsets):
    """Recover the original list of 2-D views (no copy)."""
    out = []
    for (r, c), off in zip(shapes, offsets):
        out.append(flat[off:off + r*c].reshape(r, c))
    return out


def selTournamentWithFitBracket(individuals, k, tournsize, max_fitness = None, min_fitness = None):
    pop = tools.selTournament(individuals, k, tournsize)
    # 2) Replace low-fitness winners
    sub_pop = [ind for ind in pop if (min_fitness and ind.fitness.values[0] < min_fitness) or (max_fitness and ind.fitness.values[0] > max_fitness)]
    return sub_pop


# In[108]:


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
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


# In[ ]:





# In[112]:


import time
from dateutil.relativedelta import relativedelta
time.time()

import multiprocessing
pool = multiprocessing.Pool(12)


# --- Constants ---
GENOME_LENGTH = npSegm[-1]
POP_SIZE = 200
CXPB = 0.8  # Crossover probability
MUTPB = 10 / GENOME_LENGTH  # Mutation probability per bit
NGEN = 10  # Number of generations
ELITE_SIZE = 2

# --- Evaluation Function ---
def evaluate(individual):
    return (fitness(individual),)


time_start = time.time()
def time_elapsed(pop):
    time_diff = time.time() - time_start # seconds
    rd = relativedelta(seconds=time_diff)
    years = f'{int(rd.years)} y, ' if rd.years > 0 else ''
    months = f' {int(rd.months)} mon, ' if rd.months > 0 else ''
    days = f' {int(rd.days)} d' if rd.days > 0 else ''
    hours = f' {int(rd.hours)} h' if rd.hours > 0 else ''
    mins = f' {int(rd.minutes)} m' if rd.minutes > 0 else ''
    secs = f' {int(rd.seconds)} s' if rd.seconds > 0 else ''
    return f'{years}{months}{days}{hours}{mins}{secs}'
    # return repr(rd)
    # return time_diff

def diversity(pop):
    """Return the fraction of unique genotypes in the population."""
    unique = len({ind for ind in pop})
    return unique / len(pop)

# def first_indi(pop):
#     for i in pop:
#         return "".join(str(i))

# --- Fitness and Individual ---
if "creator" in globals():
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Toolbox Setup ---
toolbox = base.Toolbox()
# toolbox.register("attr_bool", random.randint, 0, 1)
# toolbox.register("attr_bool", np.random.binomial, 1, 0.67)
toolbox.register("attr_bool", np.random.binomial, 1, 0.8)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, GENOME_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("map", pool.map)


toolbox.register("evaluate", evaluate)

# toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=MUTPB)
toolbox.register("select", tools.selTournament, tournsize=3) 
# toolbox.register("select", selTournamentWithFitBracket, tournsize=3, toolbox=toolbox, max_fitness = 0.99) 
# toolbox.register("select", selTournamentWithFitBracket, tournsize=3, max_fitness = 0.99) 

# --- Main GA Loop ---
def main():
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
    pop, log = eaSimpleWithElitism( #eaSimple( #eaSimpleWithElitism(#algorithms.eaSimple(
        pop, toolbox,
        cxpb=CXPB, 
        mutpb=1.0,  # mutpb=1.0 means each individual is mutated with MUTPB per bit
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    # algorithms.eaGenerateUpdate(toolbox, ngen=150, stats=stats, halloffame=hof)
    # pop, log = algorithms.eaGenerateUpdate(
    #     pop, toolbox,
    #     ngen=NGEN, 
    #     # evaluate=toolbox.parallel_evaluate,
    #     stats=stats, 
    #     halloffame=hof, 
    #     cxpb=CXPB, 
    #     mutpb=MUTPB,
    #     verbose=True
    # )


    return pop, log, hof
# if __name__ == "__main__":
    # main()


# In[113]:


verbose=False


# In[114]:


if "pop" in globals():
    del pop
if "log" in globals():
    del log
if "hof" in globals():
    del hof

pop, log, hof = main()


# In[ ]:


fitness(hof[0])


# In[ ]:


import datetime
import pickle

now_timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

NN_descr = "-".join( [str(i) for i in NN] ) + f"__bitlen_neur-wght-bias_{neur_len}-{wght_len}-{bias_len}"
filename = f"{now_timestamp_str}_hof__{NN_descr}.pkl"

with open(filename, 'wb') as file:
    pickle.dump(hof, file)


# In[ ]:


get_ipython().system('ls -lart *.pkl | tail -n3')


# # Observe result

# In[117]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.optimize
from scipy import signal
get_ipython().run_line_magic('matplotlib', 'inline')


def to_hex(x, pos):
    return '0x%x' % int(x)

fmt = ticker.FuncFormatter(to_hex)


# In[ ]:


if NN[0] == 2048:
    plt.imshow(Train_D_good[1].reshape(128,-1).T, cmap='gray_r', interpolation='nearest', origin='lower')
    # plt.axis('off')  # Hide axes
    # plt.title('Black and White Image from Binary Array')
    plt.show()

    plt.imshow(Train_D_bad[1].reshape(128,-1).T, cmap='gray_r', interpolation='nearest', origin='lower')
    # plt.axis('off')  # Hide axes
    # plt.title('Black and White Image from Binary Array')
    plt.show()


# In[ ]:


len(indi)
print(list(indi[:2048]))


# In[ ]:


NN, npSegm


# In[ ]:


# indi = np.zeros_like( rand_indi_custom(0.1) )
# # indi[int(npSegm[1]*3/4):int(npSegm[1]*4/4)] = 1
# # indi[int(npSegm[1]*0/4):int(npSegm[1]*1/4)] = 1


# In[ ]:


def rand_indi_custom(p):
    return np.random.binomial(1, p, size=npSegm[-1])

indi = hof[0]
indi = rand_indi_custom(0.1)

indi = np.zeros_like( rand_indi_custom(0.1) )
indi[1+int(npSegm[2]):int(npSegm[-1]):2] = 1 # setting bias '2' (n_in=n_out) flat
indi[int(npSegm[1]*0/4):int(npSegm[1]*1/4)] = 1 # more weight to on pixels in front
indi[int(npSegm[1]*3/4):int(npSegm[1]*4/4)] = 1 # more weight to on pixels in back

# indi[int(npSegm[1]*3/4):int(npSegm[1]*4/4)] = 1
indi[npSegm[1]] = 1
# indi[npSegm[3]] = 1
# indi[npSegm[3]+3] = 0

verbose = True
fitness( indi )
verbose = False
# In[ ]:


conv_from_indi_to_bias(indi)

def fitness_v(indi):
    # lower = better
    res_g = np.apply_along_axis(func1d=run_NN, axis=1, arr=Train_D_good, pars=(conv_from_indi_to_wght(indi), conv_from_indi_to_bias(indi)))
    # X=(res[:,0]+res[:,1])/6
    # Y=(res[:,2]+res[:,3])/6
    # print(np.float32(np.sum( res_g )) / len(Train_D_good))
    print(np.float32(np.sum( res_g[:20] , axis=1))  )
    failed_good = np.abs( 4*3. - np.float32(np.sum( res_g )) / len(Train_D_good) )
    # good = np.sum( np.sqrt((X-1)**2+Y**2) )/len(Train_D_good)
    # print(failed_good)

    res_b = np.apply_along_axis(func1d=run_NN, axis=1, arr=Train_D_bad, pars=(conv_from_indi_to_wght(indi), conv_from_indi_to_bias(indi)))
    # X=(res[:,0]+res[:,1])/6
    # Y=(res[:,2]+res[:,3])/6
    # bad = np.sum( np.sqrt((Y-1)**2+X**2) )/len(Train_D_bad)
    # print(np.float32(np.sum( res_b )) / len(Train_D_bad))    
    print(np.float32(np.sum( res_b[:20] , axis=1))  )
    failed_bad = np.abs(np.float32(np.sum( res_b )) / len(Train_D_bad)  - 0)
    # print(failed_bad)

    # return (good + bad)
    return failed_good + failed_bad
    
fitness_v(indi)
# In[ ]:


sum(indi)


# In[ ]:


# def fitness_view(indi):
indi = hof[0]
if indi is not None:
    res = np.apply_along_axis(func1d=run_NN, axis=1, arr=Train_D_good, pars=(conv_from_indi_to_wght(indi), ))
    # plt.hist( (np.float32(np.sum( res , axis=1))) , alpha=0.5,   label="Good")
    plt.hist( on_target(res,[1,0]) , alpha=0.5,   label="Good")

    res = np.apply_along_axis(func1d=run_NN, axis=1, arr=Train_D_bad, pars=(conv_from_indi_to_wght(indi), ))
    # X=(res[:,0]+res[:,1])
    # Y=(res[:,2]+res[:,3])
    # plt.hist(( np.sqrt((Y-1)**2+X**2) ), label="Bad")
    # plt.scatter(X,Y, alpha=0.2, marker='*')
    # bad = np.sum( np.sqrt((Y-1)**2+X**2) )/len(Train_D_bad)
    # plt.hist( np.float32(np.sum(res, axis=1)) , alpha=0.5,  label="Bad")
    plt.hist( on_target(res,[0,1]) , alpha=0.5,   label="Bad")


plt.legend()
    # return good + bad


# In[ ]:


if False:
    # lower = better
    Ngood =np.float32(len(Train_D_good))
    Nbad =np.float32(len(Train_D_bad))
    res_g = np.apply_along_axis(func1d=run_NN, axis=1, arr=Train_D_good, pars=(conv_from_indi_to_wght(indi),  conv_from_indi_to_summap(indi)))
    # if verbose:
    #     print( res_g )
    how_good = on_target(res_g,[1,0])
    # failed_good = 1. - np.sum( np.sum( res_g, axis=1 ).astype(np.float32)/((2**neur_len-1)*(NN[-1])) )/Ngood
    # good = np.sum( np.sqrt((X-1)**2+Y**2) )/len(Train_D_good)
    # print(failed_good)

    res_b = np.apply_along_axis(func1d=run_NN, axis=1, arr=Train_D_bad, pars=(conv_from_indi_to_wght(indi), conv_from_indi_to_summap(indi) ))
    # X=(res[:,0]+res[:,1])/6
    # Y=(res[:,2]+res[:,3])/6
    # bad = np.sum( np.sqrt((Y-1)**2+X**2) )/len(Train_D_bad)
    # print(np.sum( res ) / len(Train_D_bad))    
    # print( np.sum( np.sum( res_b, axis=1 ).astype(np.float32)/(3*4.) )/Nbad )
    # failed_bad = np.sum( np.sum( res_b, axis=1 ).astype(np.float32)/((2**neur_len-1)*(NN[-1])) )/Nbad
    # print(failed_bad)
    how_bad = on_target(res_b,[0,1])

    # return (good + bad)
    # return failed_good + failed_bad
    return how_good + how_bad # + np.sum(np.int8(res_g==res_b))


# In[124]:


def fitness_fast(indi):
    # higher = better

    res_g = np.apply_along_axis(func1d=run_NN, axis=1, arr=Train_D_good[:10], pars=(conv_from_indi_to_wght(indi), conv_from_indi_to_bias(indi)))
    res_g_i = np.sum( res_g, axis=1 ).astype(np.float32)/(3*4.) 
    failed_good = np.std( res_g_i )

    res_b = np.apply_along_axis(func1d=run_NN, axis=1, arr=Train_D_bad[:10], pars=(conv_from_indi_to_wght(indi), conv_from_indi_to_bias(indi)))
    res_b_i = np.sum( res_b, axis=1 ).astype(np.float32)/(3*4.)
    failed_bad = np.std( res_b_i )

    # return (good + bad)
    return failed_good + failed_bad, res_g_i, res_b_i
    # return failed_bad


def rand_indi_custom(p):
    return np.random.binomial(1, p, size=npSegm[-1])

prob_tst=[np.linspace(0.0,0.999,11),[],[]]
for p in prob_tst[0]:
    print(p, ", ", end='' )
    p_nontriv = 0
    p_nontriv_std = 0
    for i in range(1000):
        indi = rand_indi_custom(p)
        # f,res_g,res_b = fitness_fast(indi)
        f = fitness(indi)
        # print(p, f , end=None)
        if (f) != 4:
            p_nontriv += 1
            p_nontriv_std += f
            # plt.hist(res_g)
            # plt.scatter(res_g, res_b)
    prob_tst[1].append(p_nontriv)
    prob_tst[2].append(p_nontriv_std)



# In[118]:


plt.plot(prob_tst[0],prob_tst[1], label="N > 4e-7")
plt.plot(prob_tst[0],prob_tst[2], label="std")
plt.xlabel("Binomial Prob. of '1' vs. '0' in genome")
plt.ylabel("Non-trivial evaluations")


# # Write to VHDL file

# In[ ]:


if "hof" in globals():
    indi = hof[0]
else:
    indi = rand_indi()

NNwgth = conv_from_indi_to_wght(indi)
NNbias = conv_from_indi_to_bias(indi)


# In[ ]:


VHD=open("./BNN_inst.vhd", "w")


# In[ ]:


VHD_HEAD=f"""
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

library work;
use work.BNN_pack.all;


entity BNN_forward is
    Port (
    clk : in std_logic ; 
    input_i : in std_logic_vector({NN[0]}-1 downto 0);
    output_o : out std_logic_vector({NN[-1]*2}-1 downto 0)
    );
end BNN_forward;

architecture arch_imp of BNN_forward is
"""

VHD.write(VHD_HEAD)
print(VHD_HEAD)
# VHD.close()
# !cat BNN_inst.vhd


# In[ ]:





# In[ ]:


T=f"""
type t_NN_layout  is array (0 to {len(NN)}-1) of integer;
-- max index = width-1 
constant NN_width: t_NN_layout := ({", ".join([f"{i}-1" for i in NN])});
"""

VHD.write(T)
print(T)
# type t_sum_n3 is array (0 to NN_width(3) ) of neursum;
# signal sum_n3: t_sum_n3:= (others => (others => '0'));


# for L in list(range(len(NN)-1))+["f"]:
T=""
for L in range(len(NN)):
    T+=f"type t_sum_n{L} is array (0 to NN_width({L}) ) of neursum;"
    T+="\n"
    if False:
    # if L = 0:
        T+=f"type t_layer_n{L} is array (0 to NN_width({L}) ) of std_logic;"
        T+="\n"
        T+=f"signal layer_n{L}: t_layer_n{L} := (others => '0');"
    else:
        T+=f"signal sum_n{L}: t_sum_n{L} := (others => (others => '0'));"
        T+="\n"
        T+=f"type t_layer_n{L} is array (0 to NN_width({L}) ) of neur;"
    T+="\n"
    T+=f"signal layer_n{L}: t_layer_n{L} := (others => (others => '0'));"
    T+="\n"
    T+=f"type t_bias_n{L} is array (0 to NN_width({L}) ) of bias;"
    T+="\n"
VHD.write(T)
print(T)

T="\n\n"
VHD.write(T)

for L in range(len(NN)-1):
    T+=f"type t_weght_n{L}_n{L+1} is array (0 to NN_width({L}) , 0 to NN_width({L+1}) ) of wght;"
    T+="\n"
VHD.write(T)
print(T)

T="\n\n"
VHD.write(T)




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


T=""
for L in range(1,len(NN)):
    T += f"constant bias_n{L}: t_bias_n{L} := (\n"
    ll = L

    for i in range(NN[L]):
        # T += f"b\"{NNbias[L][i*2]}{NNbias[L][i*2+1]}\""  +  ("," if i < NN[L]-1 else "\n")
        T += f"b\"{np.binary_repr(NNbias[L-1][i],2)}\""  +  (", " if i < NN[L]-1 else "\n")

    T += ");\n" 
    # T += "\n"
print(T)

T+="\n\n"
VHD.write(T)



# In[ ]:


T=""
for L in range(len(NN)-1):
    T = f"constant weight_n{L}_n{L+1}: t_weght_n{L}_n{L+1} := (\n"
    VHD.write(T)
    print(T)
    ll = L

    for i in range(NN[ll]):
        T = "("
        for o in range(NN[ll+1]):
            T += f"'{(NNwgth[L].ravel()[i*(NN[ll+1]) +o]+1)//2}'"  +  ("," if o < NN[ll+1]-1 else ")\n")

        T += ("," if i < NN[ll]-1 else ");") + "\n" 
        VHD.write(T)
        print(T)

    T = "\n"
    print(T)
    VHD.write(T)




T="\n\n"
VHD.write(T)
print(T)



# In[ ]:


weight_n0_n1 = NNwgth[0]
NNwgth[0][0]


# In[ ]:


T=""
for L in range(0,len(NN)-1):
    T += f"constant summap_n{L+1}: neursum_map := ( "
    # T += f"{-int(NN[L]/2)} , {int(NN[L]/2)} , {int(NN[L]*3/2)}"       
    T += f"{int(NN[L]*3*1/4)} , {int(NN[L]*3*1/2)} , {int(NN[L]*3*3/4)}"       
    T += ");\n" 
print(T)

T+="\n\n"
VHD.write(T)

"""
constant summap_n1 : neursum_map := ( -128/2, 128/2 , 128*3/2 );
constant summap_n2 : neursum_map := ( -60/2, 60/2 , 60*3/2 );
constant summap_n3 : neursum_map := ( -24/2, 24/2 , 24*3/2 );
"""
1

for L in range(len(NN)-1):
    print(f"constant weight_n{L}_n{L+1}: t_weght_n{L}_n{L+1} := (")
    ll = L
    
    for i in range(NN[ll]+1):
        print("(", end="")
        for o in range(NN[ll+1]+1):
            print(f"'{skw():1d}'"  +  ("," if o < NN[ll+1] else ")"), end="")
        
        print("\n"  +  ("," if i < NN[ll] else ");") )



# In[ ]:


T="\nattribute KEEP : string;\n"
T += f"{"" if keep_l[0] else "-- "}attribute KEEP of layer_n{0} : signal is \"TRUE\";\n"
for L in range(1,len(NN)-1):
    T += f"{"" if keep_l[L] else "-- "}attribute KEEP of layer_n{L} : signal is \"TRUE\";\n"
T += f"{"" if keep_l[len(NN)-1] else "-- "}attribute KEEP of layer_n{len(NN)-1} : signal is \"TRUE\";\n"

print(T)

T+="\n\n"
VHD.write(T)



# In[ ]:


T="\n\nbegin\n\n"
VHD.write(T)


# In[ ]:





# In[ ]:


T=""
for n in range(NN[0]):
    if False:
        T += f"layer_n0({n}) <= input_i({n});-- & not input_i({n});\n"
    else:
        T += f"layer_n0({n}) <= input_i({n}) & not input_i({n});\n"

T+="\n\n"

for n in range(NN[-1]):
    T += f"output_o({n*2}) <= layer_n{len(NN)-1}({n})(0);\n"
    T += f"output_o({n*2+1}) <= layer_n{len(NN)-1}({n})(1);\n"

print(T)

T+="\n\n"
VHD.write(T)


# In[ ]:


# D=[2,0]
D[0]


# ### Main layer calc

# In[ ]:


for L in range(0, len(NN)-1):
    T=""
    T += f"calc_layer_n{L+1}:   for n{L+1} in 0 to NN_width({L+1}) generate\n"
    T += f"sum_n{L+1}(n{L+1}) <= \n"
    n2 = int( 2**np.ceil(np.log2(NN[L])) )
    S=["" for i in range(n2+1)]
    for i in range(n2+1):
        S[i] += "+"


    for B in range( 1, int(np.log2(n2))+1 ):
        for i in range( int((n2+1)/(2**B))+1 ):
            S[i*(2**B)] = ")" + S[i*(2**B)] + "("

    # print (S)

    S[0] = S[0].split('+')[1]
    S[-1] = S[-1].split('+')[0]
    # print (S)


    # for i in range(NN[L]):
    #     T += f" {S[i]} \n {i} \n"
    # for i in range(NN[L],n2):
    #     T += f" {S[i]} \n {i}*0 \n"
    # T += f" {S[-1]}\n"
    # print(T)

    # break

    for i in range(NN[L]):
        T += f" {S[i]} \n neur_ws(layer_n{L}({i}),weight_n{L}_n{L+1}({i},n{L+1})) \n"
    for i in range(NN[L],n2):
        T += f" {S[i]} \n 0 \n"
    T += f" {S[-1]};\n"
    ### nested sum with brackets


    T += f"layer_n{L+1}(n{L+1}) <= neur_b( neur_act( sum_n{L+1}(n{L+1}) , summap_n{L+1}), bias_n{L+1}(n{L+1}) );\nend generate;\n" 
    T += "\n"
    print(T)
    VHD.write(T)

T="\n\nend arch_imp;\n\n"
print(T)


VHD.write(T)




# In[ ]:


# neur_ws(layer_n0(0),weight_n0_n1(0,n1)) 


# In[ ]:


VHD.close()


# ## Finished VHDL

# In[ ]:





# In[ ]:





# In[ ]:


len( np.ones(NN[0]) ), len(  np.array(list(np.binary_repr(random.getrandbits( NN[0] ), width=NN[0])), dtype=np.uint8)    ) 


# In[ ]:





# # End

# In[ ]:




