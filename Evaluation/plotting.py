import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

N_EPISODES_TO_AVERAGE_OVER = 100
PLOT_AFTER_THIS_MANY_EPISODES = 50000


# Plot and return the average reward over the last n episodes and max episodic reward
# File name not including extension
def plot_reward_over_episode(
    all_done,
    all_rewards,
    directory,
    num_nodes: int,
    file_name_episodic_reward="episodic_reward",
    file_name_episode_truncated="episodic_reward_truncated",
    file_name_log_reward="log_reward",
    file_name_excel_sheet="reward_output",
) -> tuple[float, float]:
    df = pd.DataFrame(
        data={
            "episode": all_done.cumsum(),
            "reward": all_rewards,
        },
    )
    df["episode"] = df["episode"].shift().fillna(0)
    episodes_df = df.groupby("episode").agg("sum")
    episodes_df = episodes_df.iloc[:-1]

    # Rolling average
    episodes_df["avg_reward_n_episodes"] = (
        episodes_df["reward"].rolling(window=N_EPISODES_TO_AVERAGE_OVER).mean()
    )

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["reward"], linestyle="-")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Average Rewards")
    plt.title("Episodic Reward Over Time")
    plt.savefig(
        os.path.join(
            directory, file_name_episodic_reward + "_" + str(num_nodes) + ".png"
        )
    )

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, np.log(np.abs(episodes_df["reward"])), linestyle="-")
    plt.xlabel("Episode Number")
    plt.ylabel("Log of Negative Episodic Rewards")
    plt.title("Log of Negative Episodic Rewards Over Time")
    plt.savefig(
        os.path.join(directory, file_name_log_reward + "_" + str(num_nodes) + ".png")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        episodes_df.index[PLOT_AFTER_THIS_MANY_EPISODES:],
        episodes_df["reward"][PLOT_AFTER_THIS_MANY_EPISODES:],
        linestyle="-",
    )
    plt.xlabel("Episode Number")
    plt.ylabel("Total Average Rewards")
    plt.title("Episodic Reward Over Time")
    plt.savefig(
        os.path.join(
            directory, file_name_episode_truncated + "_" + str(num_nodes) + ".png"
        )
    )

    # Save the df to excel
    episodes_df.to_excel(
        os.path.join(directory, file_name_excel_sheet + "_" + str(num_nodes) + ".xlsx"),
        sheet_name="Sheet1",
        index=False,
    )

    # Don't choose the last one because the episode is not done yet
    # Rolling average
    return (
        episodes_df["avg_reward_n_episodes"].iloc[-1],
        episodes_df["reward"].max(),
    )


def plot_loss_over_time_steps(all_loss, directory, num_nodes: int, file_name="loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(all_loss, linestyle="-")
    plt.xlabel("Time Steps")
    plt.ylabel("Loss")
    plt.title("Loss Per Time Step")
    plt.savefig(os.path.join(directory, file_name + "_" + str(num_nodes) + ".png"))


def get_last_average_episodic_loss(all_done, all_loss):
    df = pd.DataFrame(
        data={
            "episode": all_done.cumsum(),
            "loss": all_loss,
        },
    )
    df["episode"] = df["episode"].shift().fillna(0)
    episodes_df = df.groupby("episode").agg("sum")
    episodes_df["avg_loss_n_episodes"] = (
        episodes_df["loss"].rolling(window=N_EPISODES_TO_AVERAGE_OVER).mean()
    )
    return episodes_df["avg_loss_n_episodes"].iloc[-1]


# Low regret and low comparative ratio is good
def plot_regret_comparative_ratio(
    all_done,
    all_rewards,
    all_optimal_path_lengths,
    directory,
    num_nodes,
    file_name_regret="regret",
    file_name_comparative_ratio="comparative_ratio",
    file_name_excel_sheet="regret_output",
):
    df = pd.DataFrame(
        data={
            "episode": all_done.cumsum(),
            "reward": all_rewards,
            "optimal_path_length": all_optimal_path_lengths,
        },
    )
    df["episode"] = df["episode"].shift().fillna(0)
    episodes_df = df.groupby("episode").agg("sum")
    episodes_df = episodes_df.iloc[:-1]
    episodes_df["regret"] = (
        episodes_df["reward"].abs() - episodes_df["optimal_path_length"]
    )
    episodes_df["comparative_ratio"] = (
        episodes_df["reward"].abs() / episodes_df["optimal_path_length"]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["regret"], linestyle="-")
    plt.xlabel("Episode number")
    plt.ylabel("Regret")
    plt.title("Regret Over Time (Per Episode)")
    plt.savefig(
        os.path.join(directory, file_name_regret + "_" + str(num_nodes) + ".png")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["comparative_ratio"], linestyle="-")
    plt.xlabel("Episode number")
    plt.ylabel("Comparative Ratio")
    plt.title("Comparative Ratio Over Time (Per Episode)")
    plt.savefig(
        os.path.join(
            directory, file_name_comparative_ratio + "_" + str(num_nodes) + ".png"
        )
    )

    episodes_df.to_excel(
        os.path.join(directory, file_name_excel_sheet + "_" + str(num_nodes) + ".xlsx"),
        sheet_name="Sheet1",
        index=False,
    )

    return episodes_df["regret"].iloc[-1], episodes_df["comparative_ratio"].iloc[-1]
