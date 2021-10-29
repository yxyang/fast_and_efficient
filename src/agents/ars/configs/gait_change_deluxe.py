"""Config for ARS training of gait_change_env."""
from ml_collections import ConfigDict
import numpy as np

from locomotion.agents.ars import policies
from locomotion.intermediate_envs.configs import pronk_deluxe
from locomotion.intermediate_envs import gait_change_env


def get_config():
  config = ConfigDict()

  # Env construction
  config.env_constructor = gait_change_env.GaitChangeEnv
  config.env_args = dict(config=pronk_deluxe.get_config(),
                         use_real_robot=False,
                         show_gui=False)

  # Policy configs:
  config.policy_constructor = policies.GaitChangePolicy
  config.filter = 'NoFilter'  #'MeanStdFilter'
  config.action_limit_method = 'tanh'  # 'tanh'
  config.num_hidden_layers = 1
  config.hidden_layer_size = 256
  # config.policy_init = ('expert_individual_policies/hybrid_test/test_kp_granular.npz')
  # ARS hyperparams
  config.logdir = 'logs'
  config.num_iters = 1000
  config.num_deltas = 16
  config.deltas_used = 8
  config.step_size = 0.02
  config.delta_std = 0.03
  config.num_workers = 18
  config.rollout_length = 400
  config.shift = 0

  # Experiment configs:
  config.seed = 237
  return config
