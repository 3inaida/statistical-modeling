import numpy as np
import scipy
import math
import pandas as pd
from typing import Any
from numpy.typing import NDArray

# Вариант 11
N = 10
P = 0.351
SIZE = 200
ALPHA = 0.05
SEED = 111

np.random.seed(SEED)


def poisson_dist(n: int, p: float) -> list[float]:
    return [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def stand_method(distribution: list[float], size: int) -> NDArray[np.int64]:
    return np.searchsorted(np.cumsum(distribution), np.random.random(size))


def make_dicts_ans(
    ans: list[int], distr: list[float], value: int, size: int
) -> dict[str, list[Any] | NDArray[Any]]:
    counts = np.bincount(ans, minlength=value + 1)
    return {
        "x_i": list(range(value + 1)),
        "n_i": counts,
        "w_i": [i / size for i in counts],
        "p_i": distr,
        "s_i": np.cumsum(distr),
    }


def table(data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(data)


# ---------
poisson = poisson_dist(N, P)

# 1. Стандартный метод
ans = stand_method(poisson, SIZE)
print("\n=== Стандартный метод ===")
print(table(make_dicts_ans(ans, poisson, N, SIZE)))

# 2. scipy
sc = scipy.stats.binom.rvs(N, P, size=SIZE, random_state=SEED)
print("\n=== scipy ===")
print(table(make_dicts_ans(sc, poisson, N, SIZE)))
