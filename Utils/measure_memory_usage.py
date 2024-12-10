import sys

sys.path.append("..")
from Environment import CTP_environment_generalize, CTP_generator
import jax
import jax.numpy as jnp
import gc

if __name__ == "__main__":
    graph_list = []
    key = jax.random.PRNGKey(0)
    gc.collect()
    for i in range(1000):
        subkey1, subkey2 = jax.random.split(key)
        new_graph = graph_realisation = CTP_generator.CTPGraph_Realisation(
            subkey2,
            5,
            10,
            0.4,
            k_edges=None,
            num_goals=1,
            factor_expensive_edge=1,
            expensive_edge=False,
        )
        graph_list.append(new_graph)
    print(len(graph_list))
    print(f"Memory for large_list: {sys.getsizeof(graph_list) / 1024 ** 2:.10f} MB")
