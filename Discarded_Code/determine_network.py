import ast

FACTOR_TO_MULTIPLY_NETWORK_SIZE = 5


# Excluding the last layer
def determine_network_size(args):
    if args.network_size is not None:
        return ast.literal_eval(args.network_size)
    else:
        first_layer_num_params = (
            FACTOR_TO_MULTIPLY_NETWORK_SIZE * args.n_node * (args.n_node + args.n_agent)
        )
        return [
            first_layer_num_params,
            int(first_layer_num_params / 2),
            int(first_layer_num_params / 4),
            int(first_layer_num_params / 8),
        ]
