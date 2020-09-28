# %%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
items = list(range(0, 10))
user1_true_utilities = dict(zip(items, (1, 2.5, 4, 9, 7, 6, 8, 10, 5, 2.5)))
user2_true_utilities = dict(zip(items, (1, 2, 4, 6, 8, 10, 9, 7, 5, 3)))


def sample_comparison_query(user_utilities, inconsistency_rate=0.0):
    outcome1, outcome2 = random.sample(user_utilities.keys(), 2)
    utility1, utility2 = user_utilities[outcome1], user_utilities[outcome2]
    pairwise_comparison_result = 1 if utility1 > utility2 else 0
    if inconsistency_rate > np.random.uniform():
        pairwise_comparison_result = 1 - pairwise_comparison_result
    return (outcome1, outcome2), pairwise_comparison_result, (utility1, utility2)


(o1, o2), result, (u1, u2) = sample_comparison_query(user1_true_utilities)
print(f"User prefered {o1 if result == 1 else o2} over {o2 if result == 1 else o1}")
print(f"Reason: Utility of {u1 if result == 1 else u2} is higher than {u2 if result == 1 else u1}")
# %%
# %%
