"""Config for ARS training of gait_change_env."""
from ml_collections import ConfigDict
import numpy as np

from locomotion.agents.ars import policies
from locomotion.intermediate_envs.configs import pronk_e2e
from locomotion.intermediate_envs import gait_change_e2e_env


def get_config():
  config = ConfigDict()

  # Env construction
  config.env_constructor = gait_change_e2e_env.GaitChangeEnv
  config.env_args = dict(config=pronk_e2e.get_config(),
                         use_real_robot=False,
                         show_gui=False)

  # Policy configs:
  config.policy_constructor = policies.LinearPolicy
  config.filter = 'MeanStdFilter'
  config.action_limit_method = 'tanh'  # 'tanh'
  config.num_hidden_layers = 1
  config.hidden_layer_size = 256
  # ARS hyperparams
  config.logdir = 'logs'
  config.num_iters = 1000
  config.num_deltas = 16
  config.deltas_used = 8
  config.step_size = 0.02
  config.delta_std = 0.03
  config.num_workers = 18
  config.rollout_length = 10000
  config.shift = 0

  # Experiment configs:
  config.seed = 237
  return config
