from deap import base, creator, tools
import numpy as np

def create_toolbox(mutation_prob, cxpb, tourn_size, eval_func, nn, pool=None):
    # if not hasattr(creator, "FitnessMax"):
    #     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "FitnessMaxSmall"):
        creator.create("FitnessMaxSmall", base.Fitness, weights=(10.0, 1.0, 0.1))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMaxSmall)  # type: ignore

    toolbox = base.Toolbox()
    toolbox.register("attr_bool_segmented", lambda: rand_indi_segmented([0.57, 0.85], len(nn.NN), nn.segm))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool_segmented) # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=cxpb)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_prob)
    toolbox.register("select", tools.selTournament, tournsize=tourn_size)
    if pool:
        toolbox.register("map", pool.map)
        toolbox.pool = pool  # attach pool object for later access
    return toolbox


def rand_indi_segmented(ps, n_layer, segm):
    arr = np.asarray(ps)
    if len(arr.shape) == 0:
        arr = np.asarray([ps])
    current_length = len(arr)
    target_length = n_layer
    if current_length >= target_length:
        ps = arr[:target_length]
    else:
        last_element = arr[-1]
        extension = np.full(target_length - current_length, last_element)
        ps = np.concatenate([arr, extension])
    return np.concatenate( [  np.random.binomial(1, ps[s], size=(segm[s+1] - segm[s])) for s in range(n_layer-1)  ] )

