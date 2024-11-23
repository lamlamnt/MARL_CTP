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


class ActorCritic_CNN_10(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        activation = nn.tanh

        action_mask = decide_validity_of_action_space(x)

        actor_mean = jnp.transpose(x, (1, 2, 0))
        actor_mean = nn.Conv(
            features=32,
            kernel_size=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_conv_1_32",
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.max_pool(
            actor_mean, window_shape=(3, 3), strides=(2, 2), padding="VALID"
        )
        actor_mean = actor_mean.reshape(-1)

        actor_mean = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense_1",
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense_2",
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense_3",
        )(actor_mean)
        actor_mean = activation(actor_mean)

        # Convert to 2D to use max pool
        actor_mean = actor_mean.reshape(-1, 1)
        actor_mean = nn.max_pool(
            actor_mean,
            window_shape=(4,),
            strides=(1,),
            padding="SAME",
        )
        actor_mean = actor_mean.reshape(-1)

        actor_mean = nn.Dense(
            32,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense_4",
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_dense_5_actions",
        )(actor_mean)

        # Do action masking
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = jnp.transpose(x, (1, 2, 0))
        critic = nn.Conv(
            features=32,
            kernel_size=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_conv_1_32",
        )(critic)
        critic = activation(critic)
        critic = nn.max_pool(
            critic,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="VALID",
        )
        critic = critic.reshape(-1)

        critic = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense_1",
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense_2",
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense_3",
        )(critic)
        critic = activation(critic)

        critic = critic.reshape(-1, 1)
        critic = nn.max_pool(critic, window_shape=(4,), strides=(1,), padding="SAME")
        critic = critic.reshape(-1)

        critic = nn.Dense(
            32,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense_4",
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_dense_5",
        )(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class Middle_FC_Block(nn.Module):
    num_neurons_1: int
    num_neurons_2: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.num_neurons_1,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            self.num_neurons_2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="VALID")
        return x


class Beginning_CNN_Block(nn.Module):
    num_filters: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.num_filters,
            kernel_size=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_1",
        )(x)
        x = nn.tanh(x)
        x = nn.Conv(
            features=self.num_filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_2",
        )(x)
        x = nn.tanh(x)
        x = nn.Conv(
            features=self.num_filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_3",
        )(x)
        x = nn.tanh(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x.reshape(-1)
        return x


class End_Block(nn.Module):
    num_neurons: int

    @nn.compact
    def __call__(self, x):
        # Convert to 2D to use max pool
        x = x.reshape(-1, 1)
        x = nn.max_pool(
            x,
            window_shape=(4,),
            strides=(1,),
            padding="SAME",
        )
        x = x.reshape(-1)

        x = nn.Dense(
            self.num_neurons,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="next_to_last_dense",
        )(x)
        x = nn.tanh(x)
        return x


class ActorCritic_CNN_30(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        action_mask = decide_validity_of_action_space(x)
        actor_mean = jnp.transpose(x, (1, 2, 0))
        actor_mean = Beginning_CNN_Block(15)(actor_mean)  # 3 conv layers
        actor_mean = Middle_FC_Block(512, 256)(actor_mean)  # 2 dense layers
        actor_mean = Middle_FC_Block(128, 64)(actor_mean)  # 2 dense layers
        actor_mean = End_Block(32)(actor_mean)  # 1 dense layer
        actor_mean = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_dense_actions",
        )(actor_mean)

        # Do action masking
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = jnp.transpose(x, (1, 2, 0))
        critic = Beginning_CNN_Block(15)(critic)
        critic = Middle_FC_Block(512, 256)(critic)
        critic = Middle_FC_Block(128, 64)(critic)
        critic = End_Block(32)(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_dense_actions",
        )(critic)
        return pi, jnp.squeeze(critic, axis=-1)
