import random
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp


def create_RBF_kernel(h_sigma):
    def vectorized_rbf_kernel(a, b):
        a_squared = np.square(np.linalg.norm(a))
        b_squared = np.square(np.linalg.norm(b))
        inner_product = (2 * (np.dot(a, b)))
        squared_diff = a_squared + b_squared - inner_product
        denominator = 2 * np.square(h_sigma)
        return np.exp(-1 * squared_diff / denominator)

    return vectorized_rbf_kernel


def pair_wise_rbf_kernel(a, b, kernel_function, h_sigma=1):
    a1, a2 = a
    b1, b2 = b
    return [kernel_function(a1, b1), kernel_function(a2, b2)]


def create_covariance_matrix(a, b, kernel_function):
    shape = [2, len(a), len(b)]
    all_covariances = [pair_wise_rbf_kernel(x, y, kernel_function) for x, y in itertools.product(a, b)]
    container = np.array(all_covariances).reshape(shape)
    return container


def process_cov_matrices(observed_X, unobserved_X, kernel_function, noise_term=0):
    K = create_covariance_matrix(observed_X, observed_X, kernel_function)

    K_star = create_covariance_matrix(unobserved_X, observed_X, kernel_function)
    K_star_star = create_covariance_matrix(unobserved_X, unobserved_X, kernel_function) + noise_term
    return K, K_star, K_star_star


def process_predictions(K, K_star, K_star_star, Y):
    numerical_stabilizer = 1e-5 * np.eye(len(K))
    K_stable = K + numerical_stabilizer
    y_star = np.dot(K_star, np.linalg.inv(K_stable)).dot(Y)
    sigma_star = K_star_star - np.dot(K_star, np.linalg.inv(K_stable)).dot(K_star.T)
    return y_star, sigma_star


def pair_wise_process_predictions(K, K_star, K_star_star, Y):
    K1, K2 = K
    K_star1, K_star2 = K_star
    K_star_star1, K_star_star2 = K_star_star
    prediction1, covariances1 = process_predictions(K1, K_star1, K_star_star1, Y)
    prediction2, covariances2 = process_predictions(K2, K_star2, K_star_star2, Y)
    return [
        (prediction1, prediction2),
        (covariances1, covariances2),
    ]


def sample_comparison_query(user_utilities, inconsistency_rate=0.0):
    outcome1, outcome2 = random.sample(user_utilities.keys(), 2)
    utility1, utility2 = user_utilities[outcome1], user_utilities[outcome2]
    pairwise_comparison_result = 1 if utility1 > utility2 else 0
    if inconsistency_rate > np.random.uniform():
        pairwise_comparison_result = 1 - pairwise_comparison_result
    return (outcome1, outcome2), pairwise_comparison_result, (utility1, utility2)


def combining_everything(user_utilities, num_train_samples=10, num_prediction_samples=50, sigma=1, noise_level=0):
    set_of_training_samples = [sample_comparison_query(user_utilities) for i in range(num_train_samples)]
    observed = pd.DataFrame([{"xi": item1, "xj": item2, "y": result} for (item1, item2), result, _ in set_of_training_samples])
    set_of_test_samples = [sample_comparison_query(user_utilities) for i in range(50)]
    unobserved = pd.DataFrame([(x1, x2, result, u1, u2) for (x1, x2), result, (u1, u2) in set_of_test_samples], columns=[
        "xi",
        "xj",
        "y_true",
        "true_util1",
        "true_util2",
    ])
    covXX, covXX_, covX_X_ = process_cov_matrices(observed.iloc[:, :2].values, unobserved[["xi", "xj"]].values, create_RBF_kernel(sigma), noise_level)
    predictions, covariances = pair_wise_process_predictions(covXX, covXX_, covX_X_, observed["y"])
    pred_comparison_results = pd.DataFrame(np.array(predictions[0] > predictions[1], dtype=np.int), columns=["pred"])
    pred_utilities = pd.DataFrame(np.array(predictions).T, columns=["pred_util1", "pred_util2"])
    correct_preds = pd.DataFrame(np.array(pred_comparison_results == unobserved["y_true"].values.reshape(-1, 1), dtype=np.bool), columns=["correct"])
    final_result = pd.concat([unobserved, pred_comparison_results, pred_utilities, correct_preds], axis=1)
    return final_result


def compute_accuracy(params):
    sigma, noise_level, num_samples, iteration, user_utilities = params
    result = combining_everything(user_utilities, num_samples, 50, sigma, noise_level)
    return {"training_examples": num_samples, "noise": noise_level, "sigma": sigma, "acc": result["correct"].mean()}


def compute_experiment(user_utilities):
    sigma_values = np.linspace(0.000001, 2, 20)
    num_iterations = list(range(10))
    noise_level = np.linspace(0, 2, 10)
    training_samples = [5, 10, 20]
    param_configurations = list(itertools.product(sigma_values, noise_level, training_samples, num_iterations, [user_utilities]))
    pool = mp.Pool(10)
    results = pool.map(compute_accuracy, tqdm(param_configurations, total=len(param_configurations)), chunksize=2)
    return pd.DataFrame(list(results))


def save_fig(grouped, name):
    for num_train in grouped["training_examples"].unique():
        subset = grouped[grouped.training_examples == num_train].iloc[:, 1:]
        plt.plot(subset.iloc[:, 0], subset.iloc[:, 1], label=f"training_examples:{num_train}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./figs/{name}.png")


if __name__ == "__main__":
    items = list(range(0, 10))
    user1_true_utilities = dict(zip(items, (1, 2.5, 4, 9, 7, 6, 8, 10, 5, 2.5)))
    experiment_results = compute_experiment(user1_true_utilities)
    experiment_results.to_csv("./experiment_data.csv")
    # grouped = experiment_results.groupby(["training_examples", "sigma"]).mean().reset_index()
    # save_fig(grouped, "noise_variations_with_10_examples")
