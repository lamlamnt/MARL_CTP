import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

N_EPISODES_TO_AVERAGE_OVER = 10


def plot_loss(all_done, all_loss, directory, file_name="loss.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(all_loss, linestyle="-")
    plt.xlabel("Time Steps")
    plt.ylabel("Loss")
    plt.title("Loss Per Time Step")
    plt.savefig(os.path.join(directory, file_name))

    df = pd.DataFrame(
        data={
            "episode": all_done.cumsum(),
            "loss": all_loss,
        },
    )
    df["episode"] = df["episode"].shift().fillna(0)
    episodes_df = df.groupby("episode").agg("sum").astype(np.float32)
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
    training=True,
    file_name_excel_sheet_episode="episode_output.xlsx",
    file_name_excel_sheet_timestep="timestep_output.xlsx",
    file_name_regret_episode="episode_regret.png",
    file_name_comparative_ratio_episode="episode_comparative_ratio.png",
    file_name_episodic_reward="episodic_reward.png",
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
    episodes_df = (
        df.groupby("episode")
        .agg("sum")
        .astype(np.float32)
        .round({"reward": 2, "optimal_path_length": 2})
    )
    episodes_df = episodes_df.iloc[:-1]
    episodes_df["regret"] = (
        episodes_df["reward"].abs() - episodes_df["optimal_path_length"]
    )
    episodes_df["comparative_ratio"] = (
        episodes_df["reward"].abs() / episodes_df["optimal_path_length"]
    )
    episodes_df.to_excel(
        os.path.join(directory, beginning_str + file_name_excel_sheet_episode),
        sheet_name="Sheet1",
        index=False,
    )

    if episodes_df.shape[0] < 1000000:
        df.to_excel(
            os.path.join(directory, beginning_str + file_name_excel_sheet_timestep),
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
            beginning_str + file_name_regret_episode,
        )
    )
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["comparative_ratio"], linestyle="-")
    plt.xlabel("Episode number")
    plt.ylabel("Comparative Ratio")
    plt.title("Comparative Ratio Over Time (By Episode)")
    plt.savefig(
        os.path.join(directory, beginning_str + file_name_comparative_ratio_episode)
    )

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["reward"], linestyle="-")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Average Rewards")
    plt.title("Episodic Reward Over Time")
    plt.savefig(
        os.path.join(
            directory,
            beginning_str + file_name_episodic_reward,
        )
    )

    # Plot histogram of rewards, regret, and comparative ratio for testing only
    if training == False:
        plt.figure(figsize=(10, 6))
        plt.hist(episodes_df["reward"], bins=10)
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Histogram of Rewards")
        plt.savefig(os.path.join(directory, beginning_str + "histogram_reward.png"))

        plt.figure(figsize=(10, 6))
        plt.hist(episodes_df["regret"], bins=10)
        plt.xlabel("Regret")
        plt.ylabel("Frequency")
        plt.title("Histogram of Regret")
        plt.savefig(os.path.join(directory, beginning_str + "histogram_regret.png"))

        plt.figure(figsize=(10, 6))
        plt.hist(episodes_df["comparative_ratio"], bins=10)
        plt.xlabel("Comparative Ratio")
        plt.ylabel("Frequency")
        plt.title("Histogram of Comparative Ratio")
        plt.savefig(
            os.path.join(directory, beginning_str + "histogram_comparative_ratio.png")
        )

    episodes_df["avg_reward_n_episodes"] = (
        episodes_df["reward"].rolling(window=N_EPISODES_TO_AVERAGE_OVER).mean()
    )

    if training == True:
        result_dict = {
            "final_regret": float(episodes_df["regret"].iloc[-1]),
            "final_comparative_ratio": float(episodes_df["comparative_ratio"].iloc[-1]),
            "avg_reward_last_episode": float(
                episodes_df["avg_reward_n_episodes"].iloc[-1]
            ),
            "max_reward": float(episodes_df["reward"].max()),
        }
    else:
        result_dict = {
            "average_regret": float(episodes_df["regret"].mean()),
            "average_comparative_ratio": float(episodes_df["comparative_ratio"].mean()),
            "average_reward": float(episodes_df["reward"].mean()),
        }
    return result_dict
