for i, row in episodes_df.iterrows():
    wandb.log(
        {
            "Reward": row["reward"],
            "Regret": row["regret"],
            "Comparative_Ratio": row["comparative_ratio"],
        },
    )
