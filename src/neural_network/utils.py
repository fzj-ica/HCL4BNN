from random import random
import numpy as np
from genetic_algorithm.utils import tuple_to_label, confusion_matrix


def calc_accuracy(res_g, res_b, labels, res_rand=np.tile([1,1], (0 , 1))):
    all_probes  = tuple_to_label( np.concatenate( [res_g, res_b, res_rand] ) )
    all_targets = tuple_to_label( np.concatenate( [
            np.tile(labels[0], (len(res_g) , 1)) , 
            np.tile(labels[1], (len(res_b) , 1)) ,
            np.tile([1,1]     , (len(res_rand) , 1)) ,
        ]) )

    cm = confusion_matrix(all_probes,all_targets)
    return  (cm[0,0]+cm[1,1])/(len(res_g)+len(res_b)), cm

def diversity_score(res_g, res_b):
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