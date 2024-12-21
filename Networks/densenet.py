import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import sys

sys.path.append("..")
from Utils.invalid_action_masking import decide_validity_of_action_space

# Highly modified version of the original DenseNet implementation
# densenet_kernel_init = nn.initializers.orthogonal(jnp.sqrt(2))
# densenet_kernel_init = nn.initializers.kaiming_normal()


class DenseLayer(nn.Module):
    bn_size: int  # Bottleneck size (factor of growth rate) for the output of the 1x1 convolution
    growth_rate: int  # Number of output channels of the 3x3 convolution
    act_fn: callable  # Activation function
    densenet_kernel_init: callable

    @nn.compact
    def __call__(self, x):
        z = self.act_fn(x)
        z = nn.Conv(
            self.bn_size * self.growth_rate,
            kernel_size=(1, 1),
            kernel_init=self.densenet_kernel_init,
            bias_init=constant(0.0),
        )(z)
        z = self.act_fn(z)
        z = nn.Conv(
            self.growth_rate,
            kernel_size=(3, 3),
            kernel_init=self.densenet_kernel_init,
            bias_init=constant(0.0),
        )(z)
        x_out = jnp.concatenate([x, z], axis=-1)
        return x_out


class DenseBlock(nn.Module):
    num_layers: int  # Number of dense layers to apply in the block
    bn_size: int  # Bottleneck size to use in the dense layers
    growth_rate: int  # Growth rate to use in the dense layers
    act_fn: callable  # Activation function to use in the dense layers
    densenet_kernel_init: callable

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.num_layers):
            x = DenseLayer(
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                act_fn=self.act_fn,
                densenet_kernel_init=self.densenet_kernel_init,
            )(x)
        return x


class TransitionLayer(nn.Module):
    c_out: int  # Output feature size
    act_fn: callable  # Activation function
    densenet_kernel_init: callable

    @nn.compact
    def __call__(self, x):
        x = self.act_fn(x)
        x = nn.Conv(
            self.c_out,
            kernel_size=(1, 1),
            kernel_init=self.densenet_kernel_init,
            bias_init=constant(0.0),
        )(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        return x


class DenseNet(nn.Module):
    act_fn: callable
    num_layers: tuple
    bn_size: int
    growth_rate: int
    densenet_kernel_init: callable

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = jnp.transpose(x, (1, 2, 0))

        c_hidden = (
            self.growth_rate * self.bn_size
        )  # The start number of hidden channels

        for block_idx, num_layers in enumerate(self.num_layers):
            x = DenseBlock(
                num_layers=num_layers,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                act_fn=self.act_fn,
                densenet_kernel_init=self.densenet_kernel_init,
            )(x)
            c_hidden += num_layers * self.growth_rate
            if (
                block_idx < len(self.num_layers) - 1
            ):  # Don't apply transition layer on last block
                x = TransitionLayer(
                    c_out=c_hidden // 2,
                    act_fn=self.act_fn,
                    densenet_kernel_init=self.densenet_kernel_init,
                )(x)
                c_hidden //= 2
        x = self.act_fn(x)
        x = x.mean(axis=(0, 1))  # global average pooling
        return x


class DenseNet_ActorCritic(nn.Module):
    num_classes: int
    act_fn: callable = nn.leaky_relu
    num_layers: tuple = (4, 4, 4)
    bn_size: int = 4
    growth_rate: int = 32
    densenet_kernel_init: callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[distrax.Categorical, float]:
        action_mask = decide_validity_of_action_space(x)
        actor_mean = DenseNet(
            act_fn=self.act_fn,
            num_layers=self.num_layers,
            bn_size=self.bn_size,
            growth_rate=self.growth_rate,
            densenet_kernel_init=self.densenet_kernel_init,
        )(x)
        actor_mean = nn.Dense(
            self.num_classes, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = DenseNet(
            act_fn=self.act_fn,
            num_layers=self.num_layers,
            bn_size=self.bn_size,
            growth_rate=self.growth_rate,
            densenet_kernel_init=self.densenet_kernel_init,
        )(x)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
