import sys

sys.path.append("..")
from Agents.ddqn_per import DDQN_PER
from Networks import MLP
from edited_jym.utils.replay_buffers import Experience
import jax
import jax.numpy as jnp
import optax
import pytest


# Check that the update function is overridden and other functions are inherited
def test_ddqn_per():
    model = MLP.Flax_FCNetwork([128, 64], 5)
    discount = 0.99
    n_actions = 2
    agent = DDQN_PER(model, discount, n_actions)

    key = jax.random.PRNGKey(10)
    subkeys = jax.random.split(key, num=3)
    target_key, online_key, action_key = subkeys
    state_shape = (
        3,
        6,
        5,
    )
    target_net_params = model.init(
        target_key, jax.random.normal(target_key, state_shape)
    )
    online_net_params = model.init(
        online_key, jax.random.normal(online_key, state_shape)
    )
    optimizer = optax.adam(learning_rate=0.001)
    current_belief_state = jnp.zeros((3, 6, 5))
    action, action_key = agent.act(
        action_key, online_net_params, current_belief_state, 1
    )
    optimizer_state = optimizer.init(online_net_params)
    experience = Experience(
        state=current_belief_state,
        action=action,
        reward=1,
        next_state=current_belief_state,
        done=0,
    )
    online_net_params, optimizer_state, loss = agent.update(
        online_net_params,
        target_net_params,
        optimizer,
        optimizer_state,
        jnp.ones(1),
        experience,
    )
