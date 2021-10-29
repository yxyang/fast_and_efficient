from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
from stable_baselines3 import DDPG

from locomotion.agents.ppo import env_wrappers
from locomotion.intermediate_envs import gait_change_env
from locomotion.intermediate_envs.configs import pronk_deluxe_fullobs

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

flags.DEFINE_string('logdir', 'path/to/dir', 'path to logdir')
FLAGS = flags.FLAGS


def make_env(seed=0):
  config = pronk_deluxe_fullobs.get_config()
  env = gait_change_env.GaitChangeEnv(config, show_gui=True)
  env = env_wrappers.LimitDuration(env, 400)
  env = env_wrappers.RangeNormalize(env)
  set_random_seed(seed)
  return env


def main(_):
  model = DDPG.load(FLAGS.logdir)
  env = make_env()

  state = env.reset()
  for _ in range(1000):
    action, _ = model.predict(state)
    state, reward, done, _ = env.step(action)
    if done:
      break

if __name__ == "__main__":
  app.run(main)

