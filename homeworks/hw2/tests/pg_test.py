import pytest
import numpy as np
from train_pg_f18 import Agent
import pprint
import sys
pprint.pprint(sys.path)

def test_sample_action():
    #discrete sampling
    batch_size = 10
    action_dimension = 5
    policy_parameters = np.zeros((batch_size, action_dimension))
    # this doesn not work anymore, before I was using numpy
    # now I am using symbolic with tf
    sampled_actions = Agent.sample_action(None, policy_parameters, discrete=True)
    assert sampled_actions.shape == (batch_size,)
    assert sampled_actions[0] == 4
