from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
from stable_baselines3 import PPO
import pickle
import os

from locomotion.agents.ppo import env_wrappers
from locomotion.intermediate_envs import gait_change_env
from locomotion.intermediate_envs.configs import pronk_deluxe

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

flags.DEFINE_string('logdir', 'path/to/dir', 'path to logdir')
FLAGS = flags.FLAGS


def make_env(seed=0):
  config = pronk_deluxe.get_config()
  config.high_level_dt = 0.002
  env = gait_change_env.GaitChangeEnv(config, show_gui=True)
  env = env_wrappers.LimitDuration(env, 400 * 25)
  env = env_wrappers.RangeNormalize(env)
  set_random_seed(seed)
  return env


def main(_):
  model = PPO.load(FLAGS.logdir)

  env = VecNormalize(DummyVecEnv([make_env]),
                     norm_reward=False,
                     training=False)
  env_dir = FLAGS.logdir[:-4].replace('rl_model', 'env_rl_model')
  env = VecNormalize.load(env_dir, env)
  env.training = False

  state = env.reset()
  sum_reward = 0
  states = []
  for t in range(1000):
    action, _ = model.predict(state)
    for _ in range(25):
      state, reward, done, _ = env.step(action)
      sum_reward += env.get_original_reward()

      original_env = env.venv.envs[0]
      states.append(
          dict(
              desired_speed=original_env.get_desired_speed(original_env.robot.GetTimeSinceReset()),
              timestamp=original_env.robot.GetTimeSinceReset(),
              base_rpy=original_env.robot.GetBaseRollPitchYaw(),
              motor_angles=original_env.robot.GetMotorAngles(),
              base_vel=original_env.robot.GetBaseVelocity(),
              base_vels_body_frame=original_env.state_estimator.com_velocity_body_frame,
              base_rpy_rate=original_env.robot.GetBaseRollPitchYawRate(),
              motor_vels=original_env.robot.GetTrueMotorVelocities(),
              motor_torques=original_env.robot.GetTrueMotorTorques(),
              contacts=original_env.robot.GetFootContacts(),
              knee_contacts=original_env.robot.GetKneeContacts(),
              # desired_grf=env.qp_sol,
              foot_forces=original_env.robot.GetFootContactForces(),
              reward=reward,
              state=state,
              robot_action=original_env.robot_action,
              env_action=action,
              base_acc=original_env.robot.GetBaseAcceleration(),
              gait_generator_phase=original_env.gait_generator.current_phase.copy(),
              gait_generator_state=original_env.gait_generator.leg_state))
      if done:
        break
    if done:
      break
  print("Total Reward: {}".format(sum_reward))
  pickle.dump(
      states,
      open(os.path.join(os.path.dirname(FLAGS.logdir), 'states_0.pkl'), 'wb'))
  logging.info("Data logged to: {}".format(
      os.path.join(os.path.dirname(FLAGS.logdir), 'states_0.pkl')))


if __name__ == "__main__":
  app.run(main)
