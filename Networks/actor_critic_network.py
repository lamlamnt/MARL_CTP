import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal, he_normal
from typing import Sequence, NamedTuple, Any
import distrax
import sys

sys.path.append("..")
from Utils.invalid_action_masking import decide_validity_of_action_space


class ActorCritic(nn.Module):
    num_actions: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        action_mask = decide_validity_of_action_space(x)

        x = x.reshape(-1)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Do action masking
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic_Narrow(nn.Module):
    num_actions: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        if self.activation == "relu":
            activation = nn.relu
            kernel_init=he_normal(bias_init=constant(0.0))
            kernel_init_last_actor=he_normal(bias_init=constant(0.0))
            kernel_init_last_critic=he_normal(bias_init=constant(0.0))
        else:
            activation = nn.tanh
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            kernel_init_last_actor=orthogonal(0.01), bias_init=constant(0.0)
            kernel_init_last_critic=orthogonal(1.0), bias_init=constant(0.0)

        action_mask = decide_validity_of_action_space(x)

        x = x.reshape(-1)
        actor_mean = nn.Dense(
            128, kernel_init=kernel_init
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=kernel_init
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            32, kernel_init=kernel_init
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.num_actions, kernel_init=kernel_init_last_actor
        )(actor_mean)

        # Do action masking
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            128, kernel_init=kernel_init
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=kernel_init
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            32, kernel_init=kernel_init
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=kernel_init_last_critic)(
            critic
        )
        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic_CNN(nn.Module):
    num_actions: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        action_mask = decide_validity_of_action_space(x)

        actor_mean = jnp.transpose(x, (1, 2, 0))
        actor_mean = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = actor_mean.reshape(-1)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Do action masking
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = jnp.transpose(x, (1, 2, 0))
        critic = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = critic.reshape(-1)

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return pi, jnp.squeeze(critic, axis=-1)

    