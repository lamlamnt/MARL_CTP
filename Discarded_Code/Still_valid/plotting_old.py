PLOT_AFTER_THIS_MANY_EPISODES = 51000


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
