import numpy as np
from typing import Tuple, Optional, List
from datasets.base_dataset import BaseDataset
from neural_network.base_nn import BaseNeuralNetwork
from genetic_algorithm.utils import diversity
from neural_network.utils import calc_accuracy

class NN(BaseNeuralNetwork):
    """
    Simple neural network simulator with LUT-based activation and custom quantised weights.
    
    This class implements a neural network with Content-Addressable Memory (CAM) based
    computation and Look-Up Table (LUT) based activation functions. The network uses
    quantized weights and supports customizable bit lengths for different components.

    Attributes
    ----------
    NN : np.ndarray
        Array containing the number of neurons in each layer.
    neur_len : int
        Bit length for neurons (default: 2).
    inp_len : int
        Bit length for input values (default: 7).
    bias_len : int
        Bit length for bias values (default: 2).
    wght_len : int
        Bit length for weights (default: 2).
    weights : List[np.ndarray]
        List of weight matrices for each layer.
    summap : List[np.ndarray]
        List of sum maps for ReLU digitization.
    description : str
        Description of the neural network instance.
        
    Notes
    -----
    The network uses 2-bit quantization for weights and implements custom
    activation functions through look-up tables for efficient hardware implementation.
    """
    _CAM_LUT = np.array([ # with input 0, 1, 2 ,3
    0, 0, 0, 0,   # bias = 0, n = 0..3
    0, 1, 2, 3,   # bias = 1, n = 0..3
    1, 2, 3, 3,   # bias = 2, n = 0..3
    3, 2, 1, 0,   # bias = 3, n = 0..3
    ], dtype=np.uint8)  
    
    def __init__(self, 
                 layers: Tuple[int, ...] = (128, 16, 128, 2), 
                 input: Optional[BaseDataset] = None, # just optional for now
                 # in bits
                 neur_len: int = 2, 
                 inp_len: int = 7, 
                 wght_len: int = 2,
                 individual: Optional[np.ndarray] = None, 
                 description: Optional[str] = None) -> None:
        """
        Initialize the neural network with the given parameters.

        Parameters
        ----------
        layers : Tuple[int, ...]
            Number of neurons in each layer.
        neur_len : int, optional
            Bit length for neurons (default: 2).
        inp_len : int, optional
            Bit length for input values (default: 7).
        bias_len : int, optional
            Bit length for bias values (default: 2).
        wght_len : int, optional
            Bit length for weights (default: 2).
        individual : Optional[np.ndarray], optional
            Initial weights in binary format (default: None).
        description : Optional[str], optional
            Description of the network (default: None).
        """
        if not all(n > 0 for n in layers):
            raise ValueError("All layer sizes must be positive integers")
        if not all(isinstance(x, int) and x > 0 for x in [neur_len, inp_len, wght_len]):
            raise ValueError("All bit lengths must be positive integers")
            
        self.NN = np.array(layers, dtype=np.int32)
        self.neur_len = neur_len
        self.wght_len = wght_len
        self.inp_len = inp_len
        self.inp_max = np.uint16((1 << inp_len) - 1)
        self.input = input
        
        # Calculate segment boundaries for weight conversion
        self.segm = np.cumsum(np.concatenate([[0], self.NN[:-1] * self.NN[1:] * wght_len]))
        
        if individual is None:
            individual = self.get_rand_indi()
        elif len(individual) != self.segm[-1]:
            raise ValueError(f"Expected individual of length {self.segm[-1]}, got {len(individual)}")
        self.individual = individual
            
        self.description = description or "NN"
        
        # Convert individual to weights and sum maps
        self.weights = self.conv_from_indi_to_wght(individual)
        self.summap = self.conv_from_indi_to_summap(individual)

    def set_weights(self, indi) -> None:
        """Return the weight matrices for each layer."""
        self.weights = self.conv_from_indi_to_wght(indi)

    def set_summap(self, indi) -> None:
        """Return the sum maps for each layer."""
        self.summap = self.conv_from_indi_to_summap(indi)

    def __str__(self) -> str:
        return f"{self.description} Net: {self.NN}, Weights: {self.wght_len}-bit, Neurons: {self.neur_len}-bit"
    
    
    
    # ============================
    # Forward / Layer
    # ============================
    # aenderbar
    def cam_neur(self, neur, wght: np.ndarray) -> np.ndarray: 
        """
        Perform CAM LUT lookup for neuron activation.
        
        This function implements the Content-Addressable Memory (CAM) lookup table
        operation for neuron activation values. It combines neuron and weight values
        to create an index into the CAM LUT.

        Parameters
        ----------
        neur : np.ndarray
            Input neuron values (should be 2-bit values: 0-3).
        wght : np.ndarray
            Weight values (should be 2-bit values: 0-3).

        Returns
        -------
        np.ndarray
            Activation values after CAM LUT lookup.

        Notes
        -----
        The function computes index = (wght << 2) | neur which is equivalent to 
        index = wght * 4 + neur, giving values from 0-15 for 2-bit inputs.
        """
        # if neur.shape != wght.shape:
        #     raise ValueError(f"Shape mismatch: neur {neur.shape} != wght {wght.shape}")
            
        # # Ensure inputs are uint8 and within valid range
        # neur = np.clip(neur.astype(np.uint8, copy=False), 0, 3)
        # wght = np.clip(wght.astype(np.uint8, copy=False), 0, 3)

        # Compute 4-bit LUT index (range 0-15)
        idx: np.ndarray = (wght << 2) | neur
        return self._CAM_LUT[idx]
    
    
    def cam_inp(self, inp: np.unsignedinteger, wght: np.ndarray) -> np.ndarray:
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


    def run_nn(self, inp: np.ndarray) :
        """
        Perform a forward pass through the neural network.
        
        Parameters
        ----------
        inp : np.ndarray
            Input vector to the neural network. Should be within valid input range.
            
        Returns
        -------
        np.ndarray
            Output vector after passing through all layers.
            
        Raises
        ------
        ValueError
            If input shape doesn't match the first layer's expected input size.
        """
        # for testing
        a = np.uint8(inp)
        for i in range(0,len(self.NN)-1):
            a = self.forward(a, i)
        return (a >= 2).astype(np.uint8)

        a = np.uint8(inp)
        a = np.apply_along_axis(func1d=self.forward, axis=0, arr=a)
        return (a >= 2).astype(np.uint8)


    
    # ========================
    # Fitness / Evaluation (abstract methods)
    # ========================
    def forward(self, x, i: int):
        """Perform one forward pass for one input x and individual indi."""
        #a = np.uint8(x)
        a = x
        
        if i == 0:
            a = self.cam_inp(a, self.weights[i])
        else:
            a = self.cam_neur(a, self.weights[i])
        neuron_sum = a.sum(axis=1)
        bin_edges = self.summap[i]
        a = np.fromiter((np.digitize(val, b) for val, b in zip(neuron_sum, bin_edges)), dtype=np.uint8)
        return a  

    def fitness(self, indi):
        X_good, X_ugly = self.input.gen_good_ugly_data() # type: ignore

        res_good = np.apply_along_axis(func1d=self.run_nn, axis=1, arr=X_good)
        res_ugly = np.apply_along_axis(func1d=self.run_nn, axis=1, arr=X_ugly)
        
        acc = calc_accuracy(res_g=res_good, res_b=res_ugly, labels=self.input.load_data()[1])[0] # type: ignore
        div = diversity(res_good, res_ugly)

        return acc , div, self.eval_size(indi)

    def eval_size(self, individual):
        zero_genes = np.sum( [len(np.where( np.ravel( i )==0)[0]) for i in self.weights] )
        return zero_genes  / ( len(individual)/2)


    def evaluate(self, x, y=None):
        indi = x
        self.set_weights(indi)
        self.set_summap(indi)
        return self.fitness(indi)

    # ========================
    # individual's conversions
    # ========================

    def get_rand_indi(self, size=None) -> np.ndarray:
        """Generate a random individual (binary array)."""
        if size is None:
            size = self.segm[-1]
        return np.random.binomial(1, 0.65, size=size)

    def conv_from_indi_to_wght(self, indi: np.ndarray) -> List[np.ndarray]:
        """Convert binary individual to 2-bit weight matrices per layer."""
        arr =  np.array(indi, dtype=np.int8)
        wghtlist: List[np.ndarray] = [] 
        for i,s in enumerate( zip( self.segm [:len(self.NN)+1],  self.segm [1:len(self.NN)] ) ):
            iwght = arr[slice(*s)].reshape([self.NN[i+1], self.NN[i],2])
            iwght_2bit = (iwght[:, :, 0]) | (iwght[:, :, 1] << 1)
            wghtlist.append(iwght_2bit)
        return wghtlist

    def conv_from_indi_to_summap(self, indi: np.ndarray) -> List[np.ndarray]:
        """Compute sum maps for ReLU digitisation."""
        summap: List[np.ndarray] = []
        for i in range(0, len(self.NN) - 1):
            nonzero = (self.NN[i] - np.uint8(self.conv_from_indi_to_wght(indi)[i] == 0).sum(axis = 1))
            summap.append(np.array(nonzero[:, np.newaxis] * [0.5, 1.5, 2.5], dtype=np.uint16))
        return summap