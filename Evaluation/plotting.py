import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

N_EPISODES_TO_AVERAGE_OVER = 10


def plot_loss(all_done, all_loss, directory, num_nodes: int, file_name="loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(all_loss, linestyle="-")
    plt.xlabel("Time Steps")
    plt.ylabel("Loss")
    plt.title("Loss Per Time Step")
    plt.savefig(os.path.join(directory, file_name + "_" + str(num_nodes) + ".png"))

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


# Save data for plotting different agents on the same graph
# Low regret (0) and comparative ratio (1) is good
def save_data_and_plotting(
    all_done,
    all_rewards,
    all_optimal_path_lengths,
    directory,
    num_nodes,
    training=True,
    file_name_excel_sheet_episode="episode_output",
    file_name_excel_sheet_timestep="timestep_output",
    file_name_regret_episode="episode_regret",
    file_name_comparative_ratio_episode="episode_comparative_ratio",
    file_name_episodic_reward="episodic_reward",
    file_name_regret_timesteps="timesteps_regret",
    file_name_comparative_ratio_timesteps="timesteps_comparative_ratio",
) -> dict[str, float]:
    if training == True:
        beginning_str = "training_"
    else:
        beginning_str = "testing_"

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
    episodes_df.to_excel(
        os.path.join(
            directory,
            beginning_str
            + file_name_excel_sheet_episode
            + "_"
            + str(num_nodes)
            + ".xlsx",
        ),
        sheet_name="Sheet1",
        index=False,
    )

    # For time steps
    """
    series = pd.Series(all_optimal_path_lengths).copy()
    series.replace(0, np.nan, inplace=True)
    series.ffill(inplace=True)
    modified_optimal_path_length = series.tolist()
    df["optimal_path_length"] = modified_optimal_path_length
    df["regret"] = df["reward"].abs() - df["optimal_path_length"]
    df["comparative_ratio"] = df["reward"].abs() / df["optimal_path_length"]
    """
    df.to_excel(
        os.path.join(
            directory,
            beginning_str
            + file_name_excel_sheet_timestep
            + "_"
            + str(num_nodes)
            + ".xlsx",
        ),
        sheet_name="Sheet1",
        index=False,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["regret"], linestyle="-")
    plt.xlabel("Episode number")
    plt.ylabel("Regret")
    plt.title("Regret Over Time (By Episode)")
    plt.savefig(
        os.path.join(
            directory,
            beginning_str + file_name_regret_episode + "_" + str(num_nodes) + ".png",
        )
    )
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["comparative_ratio"], linestyle="-")
    plt.xlabel("Episode number")
    plt.ylabel("Comparative Ratio")
    plt.title("Comparative Ratio Over Time (By Episode)")
    plt.savefig(
        os.path.join(
            directory,
            beginning_str
            + file_name_comparative_ratio_episode
            + "_"
            + str(num_nodes)
            + ".png",
        )
    )

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["reward"], linestyle="-")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Average Rewards")
    plt.title("Episodic Reward Over Time")
    plt.savefig(
        os.path.join(
            directory,
            beginning_str + file_name_episodic_reward + "_" + str(num_nodes) + ".png",
        )
    )

    episodes_df["avg_reward_n_episodes"] = (
        episodes_df["reward"].rolling(window=N_EPISODES_TO_AVERAGE_OVER).mean()
    )

    result_dict = {
        "final_regret": float(episodes_df["regret"].iloc[-1]),
        "final_comparative_ratio": float(episodes_df["comparative_ratio"].iloc[-1]),
        "avg_reward_last_episode": float(episodes_df["avg_reward_n_episodes"].iloc[-1]),
        "max_reward": float(episodes_df["reward"].max()),
    }

    """
    # Plot regret and comparative ratio over time steps
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["regret"], linestyle="-")
    plt.xlabel("Time Steps")
    plt.ylabel("Regret")
    plt.title("Regret Over Time (By Time Steps)")
    plt.savefig(
        os.path.join(
            directory,
            beginning_str + file_name_regret_timesteps + str(num_nodes) + ".png",
        )
    )

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["comparative_ratio"], linestyle="-")
    plt.xlabel("Time Steps")
    plt.ylabel("Comparative Ratio")
    plt.title("Comparative Ratio Over Time (By Time Steps)")
    plt.savefig(
        os.path.join(
            directory,
            beginning_str
            + file_name_comparative_ratio_timesteps
            + str(num_nodes)
            + ".png",
        )
    )
    """

    return result_dict
