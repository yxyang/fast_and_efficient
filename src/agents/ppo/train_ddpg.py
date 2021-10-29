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

# config_flags.DEFINE_config_file(
#     'config', 'locomotion/intermediate_envs/configs/pronk_deluxe.py',
#     'env_config')
# FLAGS = flags.FLAGS


def make_env(seed=0):
  config = pronk_deluxe_fullobs.get_config()
  env = gait_change_env.GaitChangeEnv(config)
  env = env_wrappers.LimitDuration(env, 400)
  env = env_wrappers.RangeNormalize(env)
  set_random_seed(seed)
  return env


def main(_):
  config = pronk_deluxe_fullobs.get_config()
  num_cpu = 8
  env = make_env()#SubprocVecEnv([make_env for i in range(num_cpu)])
  checkpoint_callback = CheckpointCallback(
      save_freq=5e3, save_path='logs/ddpg/model_checkpoints/')

  model = DDPG('MlpPolicy',
               env,
               learning_starts=100,
               tensorboard_log='logs/ddpg',
               verbose=1)
  model.learn(total_timesteps=10000000,
              log_interval=1,
              eval_env=make_env(),
              eval_freq=1000,
              callback=[checkpoint_callback])


if __name__ == "__main__":
  app.run(main)
