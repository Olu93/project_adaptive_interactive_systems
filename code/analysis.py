# %%
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./experiment_data.csv", index_col=0)
grouped = data.groupby(["training_examples", "noise", "sigma"]).mean().reset_index()
grouped.head()
# %%
plot_zahl = len(grouped.training_examples.unique())
fig, axes = plt.subplots(plot_zahl, figsize=(10, 7 * plot_zahl))

for noise_level in grouped.noise.unique():
    for num_train, ax in zip(grouped.training_examples.unique(), axes):
        subset = grouped[grouped.training_examples == num_train][grouped.noise == noise_level]
        ax.plot(subset.sigma, subset.acc, label=f"sample:{num_train}, noise:{noise_level:.2f}")
        ax.legend()
    
plt.tight_layout()
