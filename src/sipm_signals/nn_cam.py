import numpy as np
import time
from typing import Tuple, Optional
from .individuals import rand_indi, conv_from_indi_to_wght, conv_from_indi_to_summap
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
    
    def __init__(self, NN: Tuple[int, ...] = (128, 16, 128, 2), 
                 neur_len: int = 2, inp_len: int = 7, bias_len: int = 2, wght_len: int = 2,
                 individuals: Optional[np.ndarray] = None, description: Optional[str] = None):
        self.NN = np.array(NN)
        self.neur_len = neur_len
        self.bias_len = bias_len
        self.wght_len = wght_len
        self.keep_l = [None]*len(NN)
        self.inp_len = inp_len
        self.inp_max = np.uint16((1 << inp_len) - 1)  # max value for inp_len bits 

        self.segm = np.cumsum( np.concatenate( [[0], self.NN[:-1] * self.NN[1:] * wght_len ]) )

        if not individuals:
            individuals = rand_indi(size=self.segm[-1])

        self.description = description if description else "NN"

        # TODO: has to be done for every new individual
        self.weights = conv_from_indi_to_wght(self, individuals)
        self.summap = conv_from_indi_to_summap(self, individuals)


    def __str__(self) -> str:
        return f"{self.description} Net: {self.NN}, Weights: {self.wght_len}-bit, Neurons: {self.neur_len}-bit, Bias: {self.bias_len}-bit"
    
    
    
    # ============================
    # Forward / Layer
    # ============================
    # aenderbar
    def cam_neur(self, neur: np.ndarray, wght: np.ndarray) -> np.ndarray: 
        """
        CAM LUT lookup.
        
        Computes index = (wght << 2) | neur
        which is equivalent to index = wght * 4 + neur,
        giving values from 0-15 for 2-bit inputs.
        """
        # Ensure inputs are uint8
        neur = neur.astype(np.uint8, copy=False)
        wght = wght.astype(np.uint8, copy=False)
        print("wght shape:", wght.shape)
        print("neur shape:", neur.shape)

        # Compute 4-bit LUT index
        idx: np.ndarray = (wght << 2) | neur # range 0-15
        return self._CAM_LUT[idx]  
    
    
    def cam_inp(self, inp: np.ndarray, wght: np.ndarray) -> np.ndarray:
        """
        Compute the CAM (Content-Addressable Memory) input transformation based on
        the given input values and weight control signals.

        This function applies one of four operations to each element in `inp`, 
        depending on the corresponding value in `wght`. The available operations are:

        - **0 → blk**: Returns a zero array of the same shape as `inp`.
        - **1 → pas**: Pass-through (returns `inp` unchanged).
        - **2 → inc**: Increments the input by performing a left bit shift (×2),
        saturated to the maximum value allowed by `inp_len`.
        - **3 → neg**: Bitwise negation (~x), masked with `inp_max` to stay within
        the input bit width.

        Parameters
        ----------
        inp : np.ndarray
            Input array representing the input signals to the CAM block.
        wght : np.ndarray
            Array of the same shape as `inp` containing control codes (0–3).
            Each code selects one of the four transformation functions.

        Returns
        -------
        np.ndarray
            The transformed array, where each element of `inp` has been modified
            according to its corresponding control signal in `wght`.

        Notes
        -----
        - The function uses `np.vectorize` internally, so it supports elementwise
        operations across the entire input array.
        - The increment operation (`inc`) performs saturation arithmetic, ensuring
        that results do not exceed the representable range given by `inp_len`.
        - The negation operation (`neg`) performs a bitwise NOT and applies a mask
        defined by `inp_max` to stay within valid bit width limits.

        Examples
        --------
        >>> inp = np.array([1, 2, 3, 4], dtype=np.uint16)
        >>> wght = np.array([0, 1, 2, 3], dtype=np.uint8)
        >>> cam.cam_inp(inp, wght=wght)
        array([  0,   2,   6, 251], dtype=uint16)
        """

        def blk(inp):
            return np.zeros_like(inp)

        def pas(inp):
            return np.array(inp)

        def inc(inp):
            x = np.array(inp)
            # left shift by 1, saturate at max value
            return np.minimum(x << 1, self.inp_max)

        def neg(inp):
            x = np.array(inp)
            tmp = np.bitwise_not(x) # ~x  
            result = np.bitwise_and(tmp, self.inp_max)   # (& mask)
            return result
        
        # mapping of weights to functions
        case4 = {0: blk, 1: pas, 2: inc, 3: neg}

        def CAM_inp_scalar(inpt, wght):
            return case4[wght](inpt)

        CAM_inp = np.vectorize(CAM_inp_scalar)

        return CAM_inp(inp, wght)

    def calc_layer(
    self,
    layer_pre: np.ndarray,
    layer_pre_idx: int,
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
        layer_weights = self.weights[layer_pre_idx]

        if layer_pre_idx == 0:
            weighted = self.cam_inp(layer_pre, layer_weights)
        else:
            weighted = self.cam_neur(layer_pre, layer_weights)

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
        bin_edges = self.summap[layer_pre_idx]            # shape (n_next, n_bins)
        if verbose:
            print("* bin_edges:")
            print(bin_edges.shape)
        

        # Digitise using ReLU-like activation (TODO: change for input layer)
        ReLu_2bit = np.fromiter(    (np.digitize(x, b) for x, b in zip(neuron_sum, bin_edges)),    dtype=np.uint8)
        # ReLu_2bit = np.array( [ np.digitize(neuron_sum[i], b) for i,b in enumerate(bin_edges) ] )

        if verbose:
            print("* ReLu_2bit:")
            print(list(bin_edges)[:10], ReLu_2bit.shape)

        neurons_next = ReLu_2bit
        
        return neurons_next
    
    
    # aenderbar (je nach input und output layer)
    # eher separates pre- und post-processing der Daten
    # def run_nn(self, inp: np.ndarray) -> np.ndarray:
    #     """Forward pass for input vector."""
    #     # layer = layer_inp(inp)
    #     layer = inp + 1
    #     for i in range(len(self.NN)-1):
    #         layer = self.calc_layer(layer, i)
    #     return (layer >= 2).astype(np.uint8)

    def run_nn(self, inp: np.ndarray) -> np.ndarray:
        """Forward pass for input vector."""
        layer = inp
        for i in range(len(self.NN)-1):
            layer = self.calc_layer(layer, i)
        return layer
    
    def run_nn_from_indi(self, data, indi):
        """Run NN with weights from individual on given data."""
        old_weights = self.weights
        old_summap = self.summap
        self.weights = conv_from_indi_to_wght(self, indi)
        self.summap = conv_from_indi_to_summap(self, indi)
        result = np.apply_along_axis(self.run_nn, axis=1, arr=data)
        self.weights = old_weights
        self.summap = old_summap
        return result
    
    # TODO: pre- and post-processing of data


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
    

