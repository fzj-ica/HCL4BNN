from random import random
import numpy as np

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

# def hit_percent(probe, target):
#     p = np.asarray(probe) # / np.sum(logits)   #softmax(logits)
#     t = np.asarray(target) # / np.sum(target)
#     eq = np.all(np.equal(p,t), axis=1)
#     # print(np.sum(eq))
#     return np.float32(np.sum(eq)/2)/len(p)#.shape[0]


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