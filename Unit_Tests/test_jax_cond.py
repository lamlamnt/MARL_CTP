import jax
import jax.numpy as jnp


@jax.jit
def func(x):
    a = jax.lax.cond(x > 0, lambda x: x, lambda x: -x, x)
    return a


if __name__ == "__main__":
    x = 3
    print(func(x))
