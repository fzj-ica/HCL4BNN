from random import random
import numpy as np
from genetic_algorithm.utils import tuple_to_label, confusion_matrix


def calc_matching(preds, targets, labels):
    return  np.sum(np.int8(np.asarray(preds) == np.asarray(targets))) / len(labels) / len(preds)

def calc_accuracy(preds, targets, labels):
    "fraction of diagnoal entries in confusion matrix"
    cm = confusion_matrix(tuple_to_label(preds), tuple_to_label(targets), len(labels))
    return  (np.trace(cm))/(len(preds)), cm

def calc_diversity_score(preds, targets, labels):
    # assert len(preds)==len(trgts), f"{len(preds)} {len(trgts)}"
    assert len(preds) % 2 == 0, f"{len(preds)} not divisible by 2"
    return np.sum(np.uint8(preds[:len(preds)//2] != preds[len(preds)//2:])) / len(labels) / (len(preds)//2)
 


def calc_accuracy_two_classes(res_g, res_b, labels, res_rand=np.tile([1,1], (0 , 1))):
    all_probes  = tuple_to_label( np.concatenate( [res_g, res_b, res_rand] ) )
    all_targets = tuple_to_label( np.concatenate( [
            np.tile(labels[0], (len(res_g) , 1)) , 
            np.tile(labels[1], (len(res_b) , 1)) ,
            np.tile([1,1]     , (len(res_rand) , 1)) ,
        ]) )

    cm = confusion_matrix(all_probes,all_targets)
    return  (cm[0,0]+cm[1,1])/(len(res_g)+len(res_b)), cm

def diversity_score_two_classes(res_g, res_b):
    assert len(res_g)==len(res_b), f"{len(res_g)} {len(res_b)}"
    return np.sum(np.uint8(res_g != res_b)) / 2 / len(res_g)



def skw(p: float = 0.2) -> int:
        """
        Generate a skewed binary value.

        Parameters
        ----------
        p : float, optional
            Probability of returning 0. Default is 0.2.

        Returns
        -------
        int
            1 with probability (1-p), 0 with probability p.
        """
        return 1 if random() > p else 0
