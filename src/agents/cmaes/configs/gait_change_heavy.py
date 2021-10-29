"""Config for ARS training of gait_change_env."""
from ml_collections import ConfigDict
import numpy as np

from locomotion.agents.ars import policies
from locomotion.intermediate_envs.configs import pronk_heavy
from locomotion.intermediate_envs import gait_change_env


def get_config():
  config = ConfigDict()

  # Env construction
  config.env_constructor = gait_change_env.GaitChangeEnv
  config.env_args = dict(config=pronk_heavy.get_config(),
                         use_real_robot=False,
                         show_gui=False)

  # Policy configs:
  config.policy_constructor = policies.GaitChangePolicy
  config.filter = 'NoFilter'  #'MeanStdFilter'
  config.action_limit_method = 'tanh'  # 'tanh'
  config.num_hidden_layers = 1
  config.hidden_layer_size = 256

  # CMAES hyperparams
  config.logdir = 'logs'
  config.num_iters = 10000
  config.rollout_length = 200
  config.eval_every = 10
  return config
