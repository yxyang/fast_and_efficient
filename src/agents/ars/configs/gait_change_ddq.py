"""Config for ARS training of gait_change_env."""
from ml_collections import ConfigDict
import numpy as np

from locomotion.agents.ars import policies
from locomotion.intermediate_envs.configs import pronk_ddq
from locomotion.intermediate_envs import gait_change_env


def get_config():
  config = ConfigDict()

  # Env construction
  config.env_constructor = gait_change_env.GaitChangeEnv
  config.env_args = dict(config=pronk_ddq.get_config(),
                         use_real_robot=False,
                         show_gui=False)

  # Policy configs:
  config.policy_constructor = policies.GaitChangePolicy
  config.filter = 'NoFilter'  #'MeanStdFilter'
  config.action_limit_method = 'clip'  # 'tanh'
  config.num_hidden_layers = 1
  config.hidden_layer_size = 64
  # config.policy_init = ('expert_individual_policies/individual_2_0_sincos/'
  #                       '2021_02_19_07_47_29/lin_policy_plus_80.npz')

  # ARS hyperparams
  config.logdir = 'logs'
  config.num_iters = 1000
  config.num_deltas = 16
  config.deltas_used = 8
  config.step_size = 0.02
  config.delta_std = 0.03
  config.num_workers = 18
  config.rollout_length = 5000
  config.shift = 0

  # Experiment configs:
  config.seed = 237
  return config
