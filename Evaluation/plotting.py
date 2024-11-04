import matplotlib.pyplot as plt
import pandas as pd
import os

N_EPISODES_TO_AVERAGE_OVER = 200


# Plot and return the average reward over the last n episodes
def plot_reward_over_episode(all_done, all_rewards, directory, file_name="reward.png"):
    df = pd.DataFrame(
        data={
            "episode": all_done.cumsum(),
            "reward": all_rewards,
        },
    )
    df["episode"] = df["episode"].shift().fillna(0)
    episodes_df = df.groupby("episode").agg("sum")
    episodes_df["avg_reward_n_episodes"] = (
        episodes_df["reward"].rolling(window=N_EPISODES_TO_AVERAGE_OVER).mean()
    )

    # If want to group them instead of rolling average
    # average_reward_df = episodes_df.groupby("episode_block").agg(avg_reward=("reward", "mean"))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_df.index, episodes_df["avg_reward_n_episodes"], linestyle="-")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Average Rewards")
    plt.title(
        "Total Average Rewards Per Episode (Averaged Over "
        + str(N_EPISODES_TO_AVERAGE_OVER)
        + " Episodes)"
    )

    # Display the plot
    plt.savefig(os.path.join(directory, file_name))
    return episodes_df["avg_reward_n_episodes"].iloc[-1]


def plot_loss_over_time_steps(all_loss, directory, file_name="loss.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(all_loss, linestyle="-")
    plt.xlabel("Time Steps")
    plt.ylabel("Loss")
    plt.title("Loss Per Time Step")
    plt.savefig(os.path.join(directory, file_name))


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
