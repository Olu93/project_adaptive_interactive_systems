# %%
import random
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from experiment import sample_comparison_query, process_cov_matrices, process_predictions, pair_wise_process_predictions, pair_wise_rbf_kernel, create_RBF_kernel, compute_experiment

# %%

# %%
items = list(range(0, 10))
user1_true_utilities = dict(zip(items, (1, 2.5, 4, 9, 7, 6, 8, 10, 5, 2.5)))
user2_true_utilities = dict(zip(items, (1, 2, 4, 6, 8, 10, 9, 7, 5, 3)))

(o1, o2), result, (u1, u2) = sample_comparison_query(user1_true_utilities)
print(f"User prefered {o1 if result == 1 else o2} over {o2 if result == 1 else o1}")
print(f"Reason: Utility of {u1 if result == 1 else u2} is higher than {u2 if result == 1 else u1}")
# %%

# %%
set_of_training_samples = [sample_comparison_query(user1_true_utilities) for i in range(10)]
observed = pd.DataFrame([{"xi": item1, "xj": item2, "y": result} for (item1, item2), result, _ in set_of_training_samples])
observed.head()

# %%
set_of_test_samples = [sample_comparison_query(user1_true_utilities) for i in range(50)]
unobserved = pd.DataFrame([(x1, x2, result, u1, u2) for (x1, x2), result, (u1, u2) in set_of_test_samples], columns=[
    "xi",
    "xj",
    "y_true",
    "true_util1",
    "true_util2",
])
unobserved.head()

# %%
covXX, covXX_, covX_X_ = process_cov_matrices(observed.iloc[:, :2].values, unobserved[["xi", "xj"]].values, create_RBF_kernel(1), .001)
f"K:{covXX.shape} | K*:{covXX_.shape} | K**:{covX_X_.shape}"

# %%

predictions, covariances = pair_wise_process_predictions(covXX, covXX_, covX_X_, observed["y"])
predictions

# %%
pred_comparison_results = pd.DataFrame(np.array(predictions[0] > predictions[1], dtype=np.int), columns=["pred"])
pred_utilities = pd.DataFrame(np.array(predictions).T, columns=["pred_util1", "pred_util2"])
correct_preds = pd.DataFrame(np.array(pred_comparison_results == unobserved["y_true"].values.reshape(-1, 1), dtype=np.bool), columns=["correct"])
# %%
final_result = pd.concat([unobserved, pred_comparison_results, pred_utilities, correct_preds], axis=1)
# %%

if __name__ == "__main__":
    items = list(range(0, 10))
    user1_true_utilities = dict(zip(items, (1, 2.5, 4, 9, 7, 6, 8, 10, 5, 2.5)))
    experiment_results = compute_experiment(user1_true_utilities)
    grouped = experiment_results.groupby(["training_examples", "sigma"]).mean().reset_index()
    
    # %%
    for num_train in grouped["training_examples"].unique():
        subset = grouped[grouped.training_examples == num_train].iloc[:, 1:]
        plt.plot(subset.iloc[:, 0], subset.iloc[:, 1], label=f"training_examples:{num_train}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./figs/test.png")