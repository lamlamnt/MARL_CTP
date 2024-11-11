import sys

sys.path.append("..")
from Environment import CTP_generator, CTP_environment


# Returns a n_node x n_node matrix where the element at (i,j) is the probability of moving from node i to node j
def get_policy(key, model, environment: CTP_environment, agent):
    pass
