from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
import os
from stable_baselines3 import PPO

from locomotion.agents.ppo import env_wrappers
from locomotion.intermediate_envs import gait_change_env
from locomotion.intermediate_envs.configs import pronk_deluxe

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


class CustomCheckpointCallback(CheckpointCallback):
  def _on_step(self) -> bool:
    if self.n_calls % self.save_freq == 0:
      path = os.path.join(self.save_path,
                          f"{self.name_prefix}_{self.num_timesteps}_steps")
      self.model.save(path)
      env_norm_path = os.path.join(
          self.save_path, f"env_{self.name_prefix}_{self.num_timesteps}_steps")
      self.training_env.save(env_norm_path)
      if self.verbose > 1:
        print(f"Saving model checkpoint to {path}")
    return True


def make_env(seed=0):
  config = pronk_deluxe.get_config()
  env = gait_change_env.GaitChangeEnv(config)
  env = env_wrappers.LimitDuration(env, 400)
  env = env_wrappers.RangeNormalize(env)
  set_random_seed(seed)
  return env


def main(_):
  config = pronk_deluxe.get_config()
  num_cpu = 32
  env = SubprocVecEnv([make_env for i in range(num_cpu)])
  env = VecNormalize(env)
  model = PPO('MlpPolicy',
              env,
              n_steps=25,
              tensorboard_log='logs/ppo_true_reduced',
              verbose=1)

  model.learn(total_timesteps=0,
              log_interval=1,
              eval_env=make_env(),
              eval_freq=312,
              callback=[])
  checkpoint_callback = CustomCheckpointCallback(save_freq=1250,
                                                 save_path=os.path.join(
                                                     model.logger.get_dir(),
                                                     'model_checkpoints'))

  model.learn(total_timesteps=3000000,
              log_interval=1,
              eval_env=VecNormalize(DummyVecEnv([make_env]),
                                    norm_reward=False,
                                    training=False),
              eval_freq=312,
              callback=[checkpoint_callback])


if __name__ == "__main__":
  app.run(main)
