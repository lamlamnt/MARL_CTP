from Agents.base_agent import BaseAgent
import jax
import jax.numpy as jnp
import jaxmarl
from jaxmarl.environments import spaces
from Environment import CTP_environment
from functools import partial
import optax
import flax.linen as nn


class DQN_Agent(BaseAgent):
    def __init__(
        self,
        n_actions: int,
        model: nn.Module,
        discount_factor=0.99,
        learning_rate=0.1,
        epsilon=0.1,
    ):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # epsilon-greedy policy - epsilon decreases over time
        self.n_actions = n_actions
        self.model = model

        # data structure for replay buffer (use library)
        # replay buffer contains tuples of (state, action, reward, next_state, terminate)

    def reset(self):
        # Reset network and hyperparameters like learning_rate and epsilon
        pass

    # pass in graph into act?
    @partial(jax.jit, static_argnums=(0,))
    def act(
        self,
        key: jax.random.PRNGKey,
        state: CTP_environment.EnvState,
        observation: CTP_environment.Observation,
        online_net_params: dict,
    ) -> int:
        def _forward_pass():
            # Convert the observation sequence to matrix form -> Combine with weight and blocking probability matrices to get input to network
            # Dummy input for now. Need the goals position too
            x = jax.random.uniform(key, (1, 5, 5, 3))
            # Use online network to do action selection (max Q value)
            q_values = self.model.apply(online_net_params, x)
            # Fix this to not return array.
            return jnp.array([jnp.argmax(q_values)])

        # Epsilon greedy policy
        explore = jax.random.uniform(key) < self.epsilon
        key, subkey = jax.random.split(key)
        action = jax.lax.cond(
            explore,
            lambda subkey: jax.random.randint(
                subkey, shape=(1,), minval=0, maxval=self.n_actions
            ).astype(jnp.int32),
            lambda _: _forward_pass(),
            subkey,
        )
        return action

    def update(
        self,
        state: CTP_environment.EnvState,
        observation: CTP_environment.Observation,
        action: int,
        reward: float,
        next_state: CTP_environment.EnvState,
        next_observation: CTP_environment.Observation,
        terminate: bool,
    ):
        # Update the network
        pass

    def _loss_fn(
        self,
        online_net_params,
        target_net_params,
        states,
        actions,
        rewards,
        next_states,
        terminate,
    ):
        pass
