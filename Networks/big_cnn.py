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
from Networks.actor_critic_network import End_Block_30, Middle_FC_Block_30


class Beginning_CNN_Block_30_Bigger(nn.Module):
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
        z = nn.tanh(x)
        z = nn.Conv(
            features=self.num_filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_2",
        )(z)
        z = nn.tanh(z)
        z = nn.Conv(
            features=self.num_filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_3",
        )(z)
        z = nn.tanh(z)
        z = nn.Conv(
            features=self.num_filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_4",
        )(z)
        z = nn.tanh(z)
        z = nn.Conv(
            features=self.num_filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_5",
        )(z)
        z = nn.tanh(z)
        z = nn.Conv(
            features=self.num_filters,
            kernel_size=(2, 2),
            strides=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_6",
        )(z)
        z = nn.tanh(z)
        z = nn.Conv(
            features=self.num_filters,
            kernel_size=(2, 2),
            strides=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_7",
        )(z)
        z = nn.tanh(z)
        z = nn.Conv(
            features=self.num_filters,
            kernel_size=(2, 2),
            strides=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_8",
        )(z)
        z = nn.tanh(z)
        z = nn.Conv(
            features=self.num_filters,
            kernel_size=(2, 2),
            strides=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="conv_9",
        )(z)
        z = nn.tanh(z)
        x_out = nn.max_pool(z, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x_out.reshape(-1)
        return x_out


class Combined_Block_CNN_30_Bigger(nn.Module):
    # for both the actor and critic
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        out = jnp.transpose(x, (1, 2, 0))
        out = Beginning_CNN_Block_30_Bigger(128)(out)  # 3 conv layers
        out = Middle_FC_Block_30(512, 256)(out)  # 2 dense layers
        out = Middle_FC_Block_30(128, 64)(out)  # 2 dense layers
        out = End_Block_30(64)(out)  # 1 dense layer
        return out


class Big_CNN_30(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        action_mask = decide_validity_of_action_space(x)
        actor_mean = Combined_Block_CNN_30_Bigger()(x)
        actor_mean = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_dense_actions",
        )(actor_mean)

        # Do action masking
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = Combined_Block_CNN_30_Bigger()(x)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_dense_actions",
        )(critic)
        return pi, jnp.squeeze(critic, axis=-1)
