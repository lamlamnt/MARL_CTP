import matplotlib.pyplot as plt
import jax.numpy as jnp
import os


def plot_learning_curve(testing_average_competitive_ratio, log_directory, args):
    learning_curve_values = testing_average_competitive_ratio[
        testing_average_competitive_ratio != 0
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(
        args.num_steps_before_update * jnp.arange(len(learning_curve_values)),
        learning_curve_values,
        marker="o",
        linestyle="-",
    )
    plt.title("Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Average Competitive Ratio")
    plt.legend()
    # plt.grid(True)
    plt.savefig(os.path.join(log_directory, "Learning_curve.png"))
