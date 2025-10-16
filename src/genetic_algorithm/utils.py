import time
from dateutil.relativedelta import relativedelta
from typing import List

import numpy as np

def diversity(pop: List) -> float:
    """Fraction of unique genotypes."""
    unique = len({tuple(ind) for ind in pop})
    return unique / len(pop)

def time_elapsed(time_start: float) -> str:
    """Return elapsed time string."""
    td = time.time() - time_start
    rd = relativedelta(seconds=int(td))
    parts = []
    if rd.years: parts.append(f"{rd.years} y")
    if rd.months: parts.append(f"{rd.months} mon")
    if rd.days: parts.append(f"{rd.days} d")
    if rd.hours: parts.append(f"{rd.hours} h")
    if rd.minutes: parts.append(f"{rd.minutes} m")
    if rd.seconds: parts.append(f"{rd.seconds} s")
    return ", ".join(parts)

def confusion_matrix(y_true, y_pred, n_classes=3):
    """
    Compute a plain (non-normalized) confusion matrix.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth class indices (0 … n_classes-1).
    y_pred : array-like, shape (n_samples,)
        Predicted class indices (same coding as y_true).
    n_classes : int, optional (default=2)
        Number of distinct classes.

    Returns
    -------
    cm : ndarray, shape (n_classes, n_classes)
        cm[i, j] = # of samples whose true label = i and predicted label = j
        (rows = true, columns = predicted)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Efficient vectorised counting:
    #   for each possible pair (i, j) we count how many times it appears.
    for i in range(n_classes):
        for j in range(n_classes):
            cm[j, i] = np.sum((y_true == i) & (y_pred == j))

    return cm



def tuple_to_label(tuplist):
    """
    Convert a list or array of 2-element binary tuples into integer class labels.

    Parameters
    ----------
    tuplist : list of tuple or np.ndarray
        Each element is a tuple or array of two integers (0 or 1), e.g., (1,0), (0,1).

    Returns
    -------
    np.ndarray
        An array of integer labels:
        - 0 if tuple is (1,0)
        - 1 if tuple is (0,1)
        - 2 for any other tuple (e.g., (0,0) or (1,1))

    Examples
    --------
    >>> tuple_to_label([(1,0), (0,1), (0,0), (1,1)])
    array([0, 1, 2, 2])
    """
    p = np.asarray(tuplist)
    return np.where(
        (p[:, 0] == 1) & (p[:, 1] == 0), 0, 
        np.where((p[:, 0] == 0) & (p[:, 1] == 1), 1, 2)
    )

