import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import sys

sys.path.append("..")
from Utils.invalid_action_masking import decide_validity_of_action_space

# Use SGD with momentum instead of Adam

# resnet_kernel_init = nn.initializers.variance_scaling(
#    2.0, mode="fan_out", distribution="normal"
# )
resnet_kernel_init = orthogonal(jnp.sqrt(2.0))


class ResNetBlock(nn.Module):
    act_fn: callable  # Activation function
    c_out: int  # Output feature size
    subsample: bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x):
        # Network representing F
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            bias_init=constant(0.0),
        )(x)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            bias_init=constant(0.0),
        )(z)

        if self.subsample:
            x = nn.Conv(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=resnet_kernel_init,
                bias_init=constant(0.0),
            )(x)

        x_out = self.act_fn(z + x)
        return x_out


class ResNet(nn.Module):
    act_fn: callable
    num_blocks: tuple
    c_hidden: tuple

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (1, 2, 0))
        x = nn.Conv(
            self.c_hidden[0],
            kernel_size=(1, 1),
            kernel_init=resnet_kernel_init,
            bias_init=constant(0.0),
        )(x)
        x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = bc == 0 and block_idx > 0
                # ResNet block
                x = ResNetBlock(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample,
                )(x)

        x = x.mean(axis=(0, 1))  # Global average pooling
        return x


class ResNet_ActorCritic(nn.Module):
    num_classes: int
    # act_fn: callable = nn.relu
    act_fn: callable = nn.tanh
    num_blocks: tuple = (3, 3, 3)
    c_hidden: tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x):
        action_mask = decide_validity_of_action_space(x)
        actor_mean = ResNet(
            act_fn=self.act_fn, num_blocks=self.num_blocks, c_hidden=self.c_hidden
        )(x)
        actor_mean = nn.Dense(
            self.num_classes, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = jnp.where(action_mask == -jnp.inf, -jnp.inf, actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = ResNet(
            act_fn=self.act_fn, num_blocks=self.num_blocks, c_hidden=self.c_hidden
        )(x)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
