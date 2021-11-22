"""Evaluates policy for CMAES."""
"""Train gait policies using CMAES"""
from absl import logging

import cma
from datetime import datetime
import numpy as np
import os
import ray

from src.agents.cmaes import logz


@ray.remote
class RolloutServer(object):
  def __init__(self, server_id, config):
    self.server_id = server_id
    self.config = config
    self.env = config.env_constructor(**config.env_args)
    policy_params = {
        'ob_filter': config.filter,
        'ob_dim': self.env.observation_space.low.shape[0],
        'ac_dim': self.env.action_space.low.shape[0]
    }
    self.policy = config.policy_constructor(self.config, policy_params,
                                            self.env.observation_space,
                                            self.env.action_space)

  def eval_policy(self, step, policy_weight, eval=False):
    self.policy.update_weights(policy_weight)
    if eval:
      np.random.seed(0)
    else:
      np.random.seed(step * 100 + self.server_id)
    state = self.env.reset()
    sum_reward = 0
    for _ in range(self.config.rollout_length):
      action = self.policy.act(state)
      state, rew, done, _ = self.env.step(action)
      sum_reward += rew
      if done:
        break
    return sum_reward

  def get_weights_plus_stats(self):
    return self.policy.get_weights_plus_stats()
