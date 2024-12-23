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


class Small_Concatenate_CNN_10(nn.Module):
    pass
