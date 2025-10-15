# ============================
# Individuals
# ============================
from typing import List
import numpy as np
from .nn_cam import NN


def rand_indi(size) -> np.ndarray:
    """Generate a random individual (binary array)."""
    return np.random.binomial(1, 0.65, size=size)


# ============================
# Conversions # TODO: dependent from bit width
# ============================
def conv_from_indi_to_wght(nn: NN, indi: np.ndarray) -> List[np.ndarray]:
    """Convert binary individual to 2-bit weight matrices per layer."""
    arr =  np.array(indi, dtype=np.int8)
    wghtlist: List[np.ndarray] = [] 
    for i,s in enumerate( zip( nn.segm [:len(nn.NN)+1],  nn.segm [1:len(nn.NN)] ) ):
        iwght = arr[slice(*s)].reshape([nn.NN[i+1], nn.NN[i],2])
        iwght_2bit = (iwght[:, :, 0]) | (iwght[:, :, 1] << 1)
        wghtlist.append(iwght_2bit)
    return wghtlist

def conv_from_indi_to_summap(nn: NN, indi: np.ndarray) -> List[np.ndarray]:
    """Compute sum maps for ReLU digitisation."""
    summap: List[np.ndarray] = []
    for i in range(0, len(nn.NN) - 1):
        nonzero = (nn.NN[i] - np.uint8(conv_from_indi_to_wght(nn, indi)[i] == 0).sum(axis = 1))
        summap.append(np.array(nonzero[:, np.newaxis] * [0.5, 1.5, 2.5], dtype=np.uint16))
    return summap