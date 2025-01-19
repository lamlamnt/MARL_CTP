import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import pandas as pd


def plot_learning_curve(testing_average_competitive_ratio, log_directory, args):
    learning_curve_values = testing_average_competitive_ratio[
        testing_average_competitive_ratio != 0
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(
        args.num_steps_before_update
        * args.frequency_testing
        * jnp.arange(len(learning_curve_values)),
        learning_curve_values,
        linestyle="-",
    )
    plt.title("Learning Curve")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Competitive Ratio")
    plt.savefig(os.path.join(log_directory, "Learning_Curve.png"))

    # Plot rolling average (mean and std) of competitive ratio
    learning_curve_series = pd.Series(learning_curve_values)
    rolling_mean = learning_curve_series.rolling(window=3, min_periods=1).mean()
    rolling_std = learning_curve_series.rolling(window=3, min_periods=1).std()
    plt.figure(figsize=(10, 6))
    plt.plot(
        args.num_steps_before_update
        * args.frequency_testing
        * jnp.arange(len(learning_curve_values)),
        rolling_mean,
        linestyle="-",
        color="red",
    )
    plt.fill_between(
        args.num_steps_before_update
        * args.frequency_testing
        * jnp.arange(len(learning_curve_values)),
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="blue",
        alpha=0.2,
    )
    plt.title("Learning Curve with Rolling Average")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Competitive Ratio")
    plt.savefig(os.path.join(log_directory, "Smoothened_Learning_Curve.png"))
