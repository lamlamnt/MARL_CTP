# Get layer name and weights from FLAX params
def extract_params(params, prefix=""):
    """Recursively extract layer names and weights from a parameter dictionary."""
    for name, value in params.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):  # If the value is a nested dictionary, recurse
            yield from extract_params(value, prefix=full_name)
        else:  # If the value is a weight array, yield it
            yield full_name, value
