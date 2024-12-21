import yaml
import wandb
import random


# Define the training function
def train():
    with wandb.init() as run:
        config = run.config
        print(run.name)
        """
        run.name = (
            f"lr_{config.learning_rate}_bs_{config.batch_size}_do_{config.dropout}"
        )
        """
        # Simulate a training process
        validation_loss = (
            random.uniform(0.5, 1.5) / config.learning_rate + config.dropout
        )
        # wandb.log({"validation_loss": validation_loss})
        wandb.summary["validation_loss"] = validation_loss


# Main function to load configuration and initiate the sweep
def main():
    # Path to the YAML file
    yaml_file = "sweep_config.yaml"

    # Load the sweep configuration
    with open(yaml_file, "r") as file:
        sweep_config = yaml.safe_load(file)

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="test_bayes_optimization_4",
        entity="lam-lam-university-of-oxford",
    )

    # Start the sweep agent
    wandb.agent(sweep_id, function=train, count=3)


if __name__ == "__main__":
    main()
