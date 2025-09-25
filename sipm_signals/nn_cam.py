from abc import abstractmethod
import numpy as np
import time
from typing import Callable, List, Tuple, Optional
import sipm_signals.signals as adc

class NN:
    """
    Simple neural network simulator with LUT-based activation and custom quantised weights (2-bit).

    Attributes
    ----------
    NN : Tuple[int, ...]
        Number of neurons in each layer.
    neur_len, bias_len, wght_len : int
        Bit lengths for neurons, biases, and weights.
    """
    _CAM_LUT = np.array([ # with input 0, 1, 2 ,3
    0, 0, 0, 0,   # bias = 0, n = 0..3
    0, 1, 2, 3,   # bias = 1, n = 0..3
    1, 2, 3, 3,   # bias = 2, n = 0..3
    3, 2, 1, 0,   # bias = 3, n = 0..3
    ], dtype=np.uint8)  
    
    def __init__(self, NN: Tuple[int, ...] = (2048, 64, 32, 2), 
                 neur_len: int = 2, bias_len: int = 2, wght_len: int = 2,
                 individuals: Optional[np.ndarray] = None):
        self.NN = NN
        self.neur_len = neur_len
        self.bias_len = bias_len
        self.wght_len = wght_len
        self.keep_l = [None]*len(NN)

        self.npNN = np.array(NN)
        self.npSegm = np.cumsum( np.concatenate( [[0], self.npNN[:-1]* self.npNN[1:] * wght_len ]) )

        if not individuals:
            individuals = self._rand_indi()
        self.weight = self.conv_from_indi_to_wght(individuals)
        self.summap = self.conv_from_indi_to_summap(individuals)


    
    # ============================
    # Individuals
    # ============================
    def _rand_indi(self) -> np.ndarray:
        """Generate a random individual (binary array)."""
        return np.random.binomial(1, 0.65, size=self.npSegm[-1])


    # ============================
    # Conversions
    # ============================
    def conv_from_indi_to_wght(self, indi: np.ndarray) -> List[np.ndarray]:
        """Convert binary individual to 2-bit weight matrices per layer."""
        arr =  np.array(indi, dtype=np.int8)
        wghtlist: List[np.ndarray] = [] 
        for i,s in enumerate( zip( self.npSegm [:len(self.npNN)+1],  self.npSegm [1:len(self.npNN)] ) ):
            iwght = arr[slice(*s)].reshape([self.npNN[i+1], self.npNN[i],2])
            iwght_2bit = (iwght[:,:,0]) | (iwght[:,:,1] << 1)
            wghtlist.append(iwght_2bit)
        return wghtlist

    def conv_from_indi_to_summap(self, indi: np.ndarray) -> List[np.ndarray]:
        """Compute sum maps for ReLU digitisation."""
        summap: List[np.ndarray] = []
        for i in range(0,len(self.NN)-1):
            nonzero = (self.npNN[i] - np.uint8(self.conv_from_indi_to_wght(indi)[i] == 0).sum(axis = 1))
            summap.append(np.array(nonzero[:, np.newaxis] * [0.5, 1.5, 2.5], dtype=np.uint16))
        return summap
    
    # ============================
    # Forward / Layer
    # ============================
    # aenderbar
    def cam_neur(self, neur: np.ndarray, wght: np.ndarray) -> np.ndarray: 
        """
        CAM LUT lookup.
        
        Computes index = (wght << 2) | neur
        which is equivalent to index = wght * 4 + neur,
        giving values from 0–15 for 2-bit inputs.
        """
        # Ensure inputs are uint8
        neur = neur.astype(np.uint8, copy=False)
        wght = wght.astype(np.uint8, copy=False)
        print("wght shape:", wght.shape)
        print("neur shape:", neur.shape)


        # Compute 4-bit LUT index
        idx: np.ndarray = (wght << 2) | neur # range 0-15
        return self._CAM_LUT[idx]  
    
    def calc_layer(
    self,
    # TODO: cam_input 
    layer_pre: np.ndarray,
    layer_pre_idx: int,
    NNwgth: list[np.ndarray],
    NNsummap: list[np.ndarray],
    verbose: bool=True,
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
        weighted = self.cam_neur(layer_pre, weights)
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
    
    # aenderbar
    def run_nn(self, inp: np.ndarray) -> np.ndarray:
        """Forward pass for input vector."""
        layer = inp + 1
        for i in range(len(self.NN)-1):
            layer = self.calc_layer(layer, i, self.weight, self.summap)
        return (layer>=2).astype(np.uint8)
    

    def calc_fitness(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return SiPM (good) and noise (bad) training data."""
        Train_D_good = np.array([adc.sipm_therm() for _ in range(2)], dtype=np.uint8)
        Train_D_bad  = np.array([adc.nois_therm() for _ in range(2)], dtype=np.uint8)
        return Train_D_good, Train_D_bad

    def fitness(self, indi: np.ndarray) -> int:
        Train_D_good, Train_D_bad = self.calc_fitness()

        res_good = np.apply_along_axis(func1d=self.run_nn, axis=1, arr=Train_D_good)
        res_bad = np.apply_along_axis(func1d=self.run_nn, axis=1, arr=Train_D_bad)

        return np.sum(res_good == 1) + np.sum(res_bad == 0) + np.sum(res_good == res_bad)
    

