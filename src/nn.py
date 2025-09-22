import imp
import numpy as np
import random
import time
import sim_adc_input as adc
from deap import base, creator, tools, algorithms



class NN:
    _CAM_LUT = np.array([ # with input 0, 1, 2 ,3
    0, 0, 0, 0,   # bias = 0, n = 0..3
    0, 1, 2, 3,   # bias = 1, n = 0..3
    1, 2, 3, 3,   # bias = 2, n = 0..3
    3, 2, 1, 0,   # bias = 3, n = 0..3
    ], dtype=np.uint8)  
    
    def __init__(self, NN=(2048, 64, 32, 2), neur_len=2, bias_len=2, wght_len=2):
        self.NN = NN
        self.neur_len = neur_len
        self.bias_len = bias_len
        self.wght_len = wght_len
        self.keep_l = [None]*len(NN)

        self.npNN = np.array(NN)
        self.npSegm = np.cumsum( np.concatenate( [[0], self.npNN[:-1]* self.npNN[1:] * wght_len ]) )

        self.weights = [NN[i]*NN[i+1] for i in range(len(NN)-1)]
        self.biases = [NN[i] for i in range(1,len(NN))] # Max 0..3 * N_inputs


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

    def conv_from_indi_to_summap(self, indi):
        summap = []
        for i in range(0,len(self.NN)-1):
            nonzwght = (self.npNN[i] - np.uint8(self.conv_from_indi_to_wght(indi)[i] == 0).sum(axis = 1))
            summap.append( np.uint16( nonzwght[:,np.newaxis] * [0.5, 1.5, 2.5]) )
        return summap
    

    def forwared(self):
        verbose = True

        indi = np.zeros(self.npSegm[-1])
        indi[0] = 1
        wght = self.conv_from_indi_to_wght(indi)[0]
        

    def CAM_neur(self, neur: np.ndarray, wght: np.ndarray) -> np.ndarray: 
        # The index into the 4‑bit LUT is   idx = a*4 + b  == (a << 2) | b
        idx = (wght << 2) | neur          # still uint8, range 0‑15
        return self._CAM_LUT[idx]  
    

    def calc_layer(
    self,
    layer_pre: np.ndarray,
    layer_pre_idx: int,
    NNwgth: list[np.ndarray],
    NNsummap: list[np.ndarray],
    verbose: bool=False,
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
        

        ReLu_2bit = np.array( [ np.digitize(neuron_sum[i], b) for i,b in enumerate(bin_edges) ] )

        if verbose:
            print("* ReLu_2bit:")
            print(list(bin_edges)[:10], ReLu_2bit.shape)

        neurons_next = ReLu_2bit
        
        return neurons_next
    
    def run_NN(self, inp, pars):
        # NN, NNwgth, NNbias, NNsummap = pars
        NNwgth,NNsummap  = pars
        layer_ni = layer_inp(inp)
        for i in range(0,len(self.NN)-1):
            layer_ni = self.calc_layer( layer_ni, i, NNwgth,  NNsummap)
        return output_o(layer_ni)
    

    def calc_fitness(self):
        Train_D_good = np.empty((2, self.NN[0]), dtype=np.uint8)
        for i in range(len(Train_D_good)):
            Train_D_good[i,:] = adc.SiPM_Therm()

        Train_D_bad = np.empty((2, self.NN[0]), dtype=np.uint8)
        # for i, row in enumerate(Nois_Therm()):
        #     Train_D_bad[i, :] = row
        for i in range(len(Train_D_bad)):
            Train_D_bad[i,:] = adc.Nois_Therm()

        return Train_D_good, Train_D_bad

    def fitness(self, indi):
        # lower = better
        Train_D_good, Train_D_bad = self.calc_fitness()
        Ngood =np.float32(len(Train_D_good))
        Nbad =np.float32(len(Train_D_bad))
        res_g = np.apply_along_axis(func1d=self.run_NN, axis=1, arr=Train_D_good, pars=(self.conv_from_indi_to_wght(indi),  self.conv_from_indi_to_summap(indi)))
        how_good = on_target(res_g,[1,0])

        res_b = np.apply_along_axis(func1d=self.run_NN, axis=1, arr=Train_D_bad, pars=(self.conv_from_indi_to_wght(indi), self.conv_from_indi_to_summap(indi) ))
        how_bad = on_target(res_b,[0,1])

        return how_good + how_bad + np.sum(np.int8(res_g==res_b))
    

def layer_inp(input_i):
    return np.uint8(input_i+1)

def output_o(layer_nL):
    return (layer_nL>=2).astype(np.uint8)


def rand_inp(N):
    return np.array(list(np.binary_repr(random.getrandbits( N ), width=N)), dtype=np.uint8) 


# --------------------------------------------------------------
# Vectorised digitise  (equivalent to np.digitize(..., side='right'))
# --------------------------------------------------------------
def digitise_rows(neuron_sum: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
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

  
def softmax(z):
    # e = np.exp(z - np.max(z))
    return z / z.sum()

def on_target(probe, target):
    p = probe # / np.sum(logits)   #softmax(logits)
    t = np.array(target) # / np.sum(target)
    return np.sum(np.int8(p==t))