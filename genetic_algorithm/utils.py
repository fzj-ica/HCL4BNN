import time
from dateutil.relativedelta import relativedelta
from typing import List

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
