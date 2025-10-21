# ============================
# Individuals
# ============================
from typing import List
import numpy as np
# from .nn_cam import NN # This would create a circular import


def rand_indi(size) -> np.ndarray:
    """Generate a random individual (binary array)."""
    return np.random.binomial(1, 0.65, size=size)


# ============================
# Conversions # TODO: dependent from bit width
# ============================
def conv_from_indi_to_wght(nn, indi: np.ndarray) -> List[np.ndarray]:
    """Convert binary individual to 2-bit weight matrices per layer."""
    arr =  np.array(indi, dtype=np.int8)
    wghtlist: List[np.ndarray] = [] 
    for i,s in enumerate( zip( nn.segm [:len(nn.NN)+1],  nn.segm [1:len(nn.NN)] ) ):
        iwght = arr[slice(*s)].reshape([nn.NN[i+1], nn.NN[i],2])
        iwght_2bit = (iwght[:, :, 0]) | (iwght[:, :, 1] << 1)
        wghtlist.append(iwght_2bit)
    return wghtlist

def conv_from_indi_to_summap(nn, indi: np.ndarray) -> List[np.ndarray]:
    """Compute sum maps for ReLU digitisation."""
    summap: List[np.ndarray] = []
    for i in range(0, len(nn.NN) - 1):
        nonzero = (nn.NN[i] - np.uint8(conv_from_indi_to_wght(nn, indi)[i] == 0).sum(axis = 1))
        summap.append(np.array(nonzero[:, np.newaxis] * [0.5, 1.5, 2.5], dtype=np.uint16))
    return summap


# ============================
# Running NN with individual
# ============================
# TODO
# def run_NN_from_indi(data: np.ndarray, indi: np.ndarray) -> np.ndarray:
#     """Run NN with weights from individual on given data."""
#     nn = NN()
#     nn.weights = conv_from_indi_to_wght(nn, indi)
#     nn.summap = conv_from_indi_to_summap(nn, indi)
#     result = np.apply_along_axis(nn.run_nn, axis=1, arr=data)
#     return result

# def fitness(indi, tests=10): # WiP
#     res_g, res_b = run_NN_data_classes(indi, tests)
#     acc = calc_accuracy(res_g, res_b)[0]
#     div = diversity_score(res_g, res_b)
#     return acc , div

# TODO
# def run_NN_data_classes(indi, tests=10):
#     Train_D_good, Train_D_bad = gen_Data(tests, tests)

#     res_g = run_NN_from_indi(Train_D_good, indi)

#     res_b = run_NN_from_indi(Train_D_bad,  indi)
#     return res_g, res_b



# TODO
# def calc_accuracy(res_g, res_b,res_rand=np.tile([1,1]     , (0 , 1))):
#     # how_good = tuple_to_label(res_g) == SiPM_num_lbl
#     # how_good = on_target(res_g, SiPM_NNout)
#     # how_bad  = on_target(res_b, Nois_NNout)
#     all_probes  = tuple_to_label( np.concatenate( [res_g, res_b, res_rand] ) )
#     all_targets = tuple_to_label( np.concatenate( [
#             np.tile(SiPM_NNout, (len(res_g) , 1)) , 
#             np.tile(Nois_NNout, (len(res_b) , 1)) ,
#             np.tile([1,1]     , (len(res_rand) , 1)) ,
#         ]) )

#     cm = confusion_matrix(all_probes,all_targets)
#     return  (cm[0,0]+cm[1,1])/(len(res_g)+len(res_b)), cm



def diversity_score(res_g, res_b):
    assert len(res_g)==len(res_b), f"{len(res_g)} {len(res_b)}"
    return np.sum(np.uint8(res_g != res_b)) / 2 / len(res_g)
    # distmin = min(len(res_g),len(res_b))
    # return np.sum(np.uint8(res_g[:distmin] != res_b[:distmin]))
    
    # return normalized_Shannon_entropy(tuple_to_label(np.concatenate( [res_g, res_b] )))


# def normalized_Shannon_entropy(arr, num_classes=3):
#     values, counts = np.unique(arr, return_counts=True)
#     probs = counts / counts.sum()
#     entropy = -np.sum(probs * np.log(probs)) / np.log(num_classes)
#     return entropy






# ============================
# Data generation
# ============================


# Need to be changed

lenTrain_D_good = 50
lenTrain_D_bad = 50


def distill_uniform(arr,min_amp=10,sample_size=100):
    arr = arr[np.max(arr, axis=1) >= min_amp]
    maxima = np.max(arr, axis=1)
    num_bins = 50
    bins = np.linspace(np.min(maxima), np.max(maxima), num_bins + 1)
    idx = np.digitize(maxima, bins) - 1
    counts = np.bincount(idx, minlength=num_bins)
    weights = 1.0 / counts[idx]
    weights /= np.sum(weights)  # normalize
    k = sample_size
    sample_indices = np.random.choice(range(len(arr)), size=k, p=weights)
    # sample = [arr[i] for i in sample_indices]


    # hist, bins = np.histogram(maxima, bins=len(arr))
    # probabilities = hist / np.sum(hist)
    # sampled_indices = np.random.choice(len(arr), size=sample_size, p=probabilities)
    # print(sample_indices[:10])
    return arr[sample_indices]


# def gen_Data(good=lenTrain_D_good, bad=lenTrain_D_bad, min_amp = 10, ADC_smpls=): # used to be dependent on NN[0], but = ADC_smpls
#     Train_D_good = np.empty((good*20, ADC_smpls), dtype=np.uint8)
#     for i in range(good*20):
#         Train_D_good[i,:] = sipm_inp()
    
#     Train_D_bad = np.empty((bad*20, ADC_smpls), dtype=np.uint8)
#     for i in range(bad*20):
#         Train_D_bad[i,:] = Nois_inp()

#     Train_D_good = distill_uniform(Train_D_good, min_amp = min_amp, sample_size = good)
#     Train_D_bad  = distill_uniform(Train_D_bad,  min_amp = min_amp, sample_size = bad)
#     # print(len(Train_D_good), len(Train_D_bad) )
#     return Train_D_good, Train_D_bad