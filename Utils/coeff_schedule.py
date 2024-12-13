import jax
import jax.numpy as jnp


# Return the frac for ent_coeff schedule
def ent_coeff_plateau_decay(loop_count, num_loops, division=4):
    part = num_loops // division
    return jax.lax.cond(
        loop_count < part,
        lambda _: 1.0,  # First part
        lambda _: jax.lax.cond(
            loop_count < (division - 1) * part,
            lambda _: 1.0
            - (loop_count - part) / ((division - 2) * part),  # Middle part
            lambda _: 0.0,  # Last part
            operand=None,
        ),
        None,
    )
