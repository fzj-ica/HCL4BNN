from random import random
import numpy as np


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