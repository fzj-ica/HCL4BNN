import numpy as np


def diversity_score(res_g, res_b):
       assert len(res_g)==len(res_b), f"{len(res_g)} {len(res_b)}"
       return np.sum(np.uint8(res_g != res_b)) / 2 / len(res_g)


