import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

# Highly modified version of the original DenseNet implementation
densenet_kernel_init = nn.initializers.kaiming_normal()


class DenseLayer(nn.Module):
    bn_size: int  # Bottleneck size (factor of growth rate) for the output of the 1x1 convolution
    growth_rate: int  # Number of output channels of the 3x3 convolution
    act_fn: callable  # Activation function

    @nn.compact
    def __call__(self, x):
        z = self.act_fn(z)
        z = nn.Conv(
            self.bn_size * self.growth_rate,
            kernel_size=(1, 1),
            kernel_init=densenet_kernel_init,
            bias_init=constant(0.0),
        )(z)
        z = self.act_fn(z)
        z = nn.Conv(
            self.growth_rate,
            kernel_size=(3, 3),
            kernel_init=densenet_kernel_init,
            bias_init=constant(0.0),
        )(z)
        x_out = jnp.concatenate([x, z], axis=-1)
        return x_out


class DenseBlock(nn.Module):
    num_layers: int  # Number of dense layers to apply in the block
    bn_size: int  # Bottleneck size to use in the dense layers
    growth_rate: int  # Growth rate to use in the dense layers
    act_fn: callable  # Activation function to use in the dense layers

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.num_layers):
            x = DenseLayer(
                bn_size=self.bn_size, growth_rate=self.growth_rate, act_fn=self.act_fn
            )(x)
        return x


class TransitionLayer(nn.Module):
    c_out: int  # Output feature size
    act_fn: callable  # Activation function

    @nn.compact
    def __call__(self, x):
        x = self.act_fn(x)
        x = nn.Conv(
            self.c_out,
            kernel_size=(1, 1),
            kernel_init=densenet_kernel_init,
            bias_init=constant(0.0),
        )(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x


class DenseNet(nn.Module):
    num_classes: int
    act_fn: callable = nn.relu
    num_layers: tuple = (6, 12, 24, 16)
    bn_size: int = 4
    growth_rate: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        c_hidden = (
            self.growth_rate * self.bn_size
        )  # The start number of hidden channels

        x = nn.Conv(c_hidden, kernel_size=(3, 3), kernel_init=densenet_kernel_init)(x)

        for block_idx, num_layers in enumerate(self.num_layers):
            x = DenseBlock(
                num_layers=num_layers,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                act_fn=self.act_fn,
            )(x)
            c_hidden += num_layers * self.growth_rate
            if (
                block_idx < len(self.num_layers) - 1
            ):  # Don't apply transition layer on last block
                x = TransitionLayer(c_out=c_hidden // 2, act_fn=self.act_fn)(x)
                c_hidden //= 2
        x = self.act_fn(x)
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x
