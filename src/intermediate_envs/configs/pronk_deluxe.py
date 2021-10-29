"""Env configuration for pronking."""
from ml_collections import ConfigDict
import numpy as np

from src.convex_mpc_controller import gait_generator as gait_generator_lib
LegState = gait_generator_lib.LegState


def get_config():
  config = ConfigDict()
  config.actions = ['phase']
  config.init_phase = np.array([0., 0., 0., 0])
  config.qp_acc_weight = np.array([1., 1., 1., 10., 10, 1.])

  config.speed_profile = (np.array([0, 15, 20, 21, 22]),
                          np.array([[0., 0., 0., 0.], [2.5, 0., 0., 0.],
                                    [2.5, 0., 0., 0.], [0., 0., 0., 0.],
                                    [0., 0., 0., 0.]]))

  config.use_mpc_stance_controller = True
  config.mass_ratio = 1.8
  config.foot_friction = 0.5
  config.mpc_foot_friction = 0.2
  config.vel_estimation_ratio = [0.8, 1.2]
  config.high_level_dt = 0.05

  # Observation and Action
  config.use_full_observation = False
  config.action_high = np.array([4, 1, 1, 1, 1, 1, 1, 0.99])
  config.action_low = np.array([0.001, -1, -1, -1, -1, -1, -1, 0.01])

  # Reward weights
  config.action_penalty_weight = 0.
  config.alive_bonus = 3
  config.use_cot = True
  config.speed_penalty_weight = 1.
  config.power_penalty_weight = 0.0025
  config.speed_penalty_type = 'relative'

  return config
