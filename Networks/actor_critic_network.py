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


class Block_CNN_10(nn.Module):
    # This block is for both the actor and critic
    @nn.compact
    def __call__(self, x):
        out = jnp.transpose(x, (1, 2, 0))
        out = nn.Conv(
            features=40,
            kernel_size=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_1",
        )(out)
        out = nn.tanh(out)

        out = nn.max_pool(out, window_shape=(3, 3), strides=(2, 2), padding="VALID")
        out = out.reshape(-1)

        out = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_1",
        )(out)
        out = nn.tanh(out)
        out = nn.Dense(
            128,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_2",
        )(out)
        out = nn.tanh(out)
        out = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_3",
        )(out)
        out = nn.tanh(out)

        # Convert to 2D to use max pool
        out = out.reshape(-1, 1)
        out = nn.max_pool(
            out,
            window_shape=(4,),
            strides=(1,),
            padding="SAME",
        )
        out = out.reshape(-1)

        out = nn.Dense(
            32,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="dense_4",
        )(out)
        out = nn.tanh(out)
        return out


class ActorCritic_CNN_10(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        action_mask = decide_validity_of_action_space(x)
        actor_mean = Block_CNN_10()(x)
        actor_mean = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_dense_actions",
        )(actor_mean)

        # Do action masking
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = Block_CNN_10()(x)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_dense_actions",
        )(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class Middle_FC_Block_30(nn.Module):
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


class Beginning_CNN_Block_30(nn.Module):
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


class End_Block_30(nn.Module):
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


class Combined_Block_CNN_30(nn.Module):
    # for both the actor and critic
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        out = jnp.transpose(x, (1, 2, 0))
        out = Beginning_CNN_Block_30(64)(out)  # 3 conv layers
        out = Middle_FC_Block_30(512, 256)(out)  # 2 dense layers
        out = Middle_FC_Block_30(128, 64)(out)  # 2 dense layers
        out = End_Block_30(64)(out)  # 1 dense layer
        return out


class ActorCritic_CNN_30(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        action_mask = decide_validity_of_action_space(x)
        actor_mean = Combined_Block_CNN_30()(x)
        actor_mean = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_dense_actions",
        )(actor_mean)

        # Do action masking
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = Combined_Block_CNN_30()(x)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_dense_actions",
        )(critic)
        return pi, jnp.squeeze(critic, axis=-1)
