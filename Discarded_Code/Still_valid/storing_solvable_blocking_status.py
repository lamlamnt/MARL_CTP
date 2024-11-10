# Related to sampling and storing solvable blocking status
num_stored_realisations: int
patience_factor: int
stored_realisations: jnp.ndarray
self.stored_realisations = self.get_solvable_blocking_status(subkey)

new_blocking_status = self.stored_realisations[indice]


def get_solvable_blocking_status(self, key) -> jnp.ndarray:
    realisation_counter = 0
    # This is to avoid us being stuck in an infinite loop if the prop_stock is too high
    patience_counter = 0
    stored_realisations = jnp.zeros(
        (self.num_stored_realisations, self.num_nodes, self.num_nodes),
        dtype=jnp.int32,
    )
    key, subkey = jax.random.split(key)
    while realisation_counter < self.num_stored_realisations and patience_counter < (
        self.patience_factor * self.num_stored_realisations
    ):
        new_blocking_status = self.graph_realisation.sample_blocking_status(subkey)
        if self.graph_realisation.is_solvable(new_blocking_status):
            stored_realisations = stored_realisations.at[realisation_counter, :, :].set(
                new_blocking_status
            )
            realisation_counter += 1
        patience_counter += 1
        key, subkey = jax.random.split(key)
    if patience_counter >= self.patience_factor * self.num_stored_realisations:
        raise ValueError(
            "Could not find enough solvable blocking status. Please decrease the prop_stoch."
        )
    return stored_realisations
