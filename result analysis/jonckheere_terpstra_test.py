# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:03:53 2025

@author: roywinter
"""

import numpy as np
from scipy.stats import rankdata, norm
from itertools import combinations

def jonckheere_terpstra_test(groups):
    """
    groups: list of arrays, where each array contains observations from one group
            groups should be in hypothesized order (e.g., increasing obstacle ratio)
    """
    k = len(groups)
    n_total = sum(len(g) for g in groups)

    JT_stat = 0
    for i, j in combinations(range(k), 2):
        for x in groups[i]:
            for y in groups[j]:
                if x < y:
                    JT_stat += 1
                elif x == y:
                    JT_stat += 0.5

    # Compute mean and variance under H0
    ni = [len(g) for g in groups]
    mean_JT = sum(ni[i] * ni[j] / 2 for i in range(k) for j in range(i + 1, k))

    var_JT = 0
    for i in range(k):
        for j in range(i + 1, k):
            ni_, nj_ = ni[i], ni[j]
            var_JT += ni_ * nj_ * (ni_ + nj_ + 1) / 12

    z = (JT_stat - mean_JT) / np.sqrt(var_JT)
    p_value = 1 - norm.cdf(z)  # one-sided test for increasing trend

    return JT_stat, z, p_value