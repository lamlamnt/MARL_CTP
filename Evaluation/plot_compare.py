import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import argparse
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
    parser.add_argument(
        "--folder_names",
        type=str,
        help="List of folder names to compare",
        required=False,
        default="DQN_uniform_5,DQN_per_5,double_dqn_5",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        help="Name of excel file",
        required=False,
        default="testing_episode_output.xlsx",
    )
    args = parser.parse_args()

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    log_dir = os.path.join(parent_dir, "Logs")

    folder_names = args.folder_names.split(",")
    # Load data from files
    data_dict = {
        f"{method}": pd.read_excel(
            os.path.join(log_dir, method, args.file_name), sheet_name="Sheet1"
        )
        for method in folder_names
    }

    # Plot reward, regret, and comparative ratio
    plt.figure(figsize=(10, 6))
    for file_name, data in data_dict.items():
        # Maybe different colours
        plt.plot(data["reward"], label=file_name, linestyle="-")
    plt.title(f"Plot of Reward (Testing)")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "Comparison_Plots", "reward_testing.png"))

    plt.figure(figsize=(10, 6))
    for file_name, data in data_dict.items():
        # Maybe different colours
        plt.plot(data["regret"], label=file_name, linestyle="-")
    plt.title(f"Plot of Regret (Testing)")
    plt.xlabel("Episode Number")
    plt.ylabel("Regret")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "Comparison_Plots", "regret_testing.png"))

    plt.figure(figsize=(10, 6))
    for file_name, data in data_dict.items():
        # Maybe different colours
        plt.plot(data["comparative_ratio"], label=file_name, linestyle="-")
    plt.title(f"Plot of Comparative Ratio (Testing)")
    plt.xlabel("Episode Number")
    plt.ylabel("Comparative Ratio")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(log_dir, "Comparison_Plots", "comparative_ratio_testing.png")
    )
