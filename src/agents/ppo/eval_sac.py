from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
from stable_baselines3 import SAC
import pickle
import os

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
  config.high_level_dt = 0.002
  env = gait_change_env.GaitChangeEnv(config, show_gui=True)
  env = env_wrappers.LimitDuration(env, 400 * 25)
  env = env_wrappers.RangeNormalize(env)
  set_random_seed(seed)
  return env


def main(_):
  model = SAC.load(FLAGS.logdir)
  env = make_env()

  state = env.reset()
  sum_reward = 0
  states = []
  for t in range(400):
    action, _ = model.predict(state)
    for _ in range(25):
      state, reward, done, _ = env.step(action)
      sum_reward += reward
      states.append(
          dict(
              desired_speed=env.get_desired_speed(env.robot.GetTimeSinceReset()),
              timestamp=env.robot.GetTimeSinceReset(),
              base_rpy=env.robot.GetBaseRollPitchYaw(),
              motor_angles=env.robot.GetMotorAngles(),
              base_vel=env.robot.GetBaseVelocity(),
              base_vels_body_frame=env.state_estimator.com_velocity_body_frame,
              base_rpy_rate=env.robot.GetBaseRollPitchYawRate(),
              motor_vels=env.robot.GetTrueMotorVelocities(),
              motor_torques=env.robot.GetTrueMotorTorques(),
              contacts=env.robot.GetFootContacts(),
              knee_contacts=env.robot.GetKneeContacts(),
              # desired_grf=env.qp_sol,
              foot_forces=env.robot.GetFootContactForces(),
              reward=reward,
              state=state,
              robot_action=env.robot_action,
              env_action=action,
              base_acc=env.robot.GetBaseAcceleration(),
              gait_generator_phase=env.gait_generator.current_phase.copy(),
              gait_generator_state=env.gait_generator.leg_state))
      if done:
        break
    if done:
      break
  print("Total Reward: {}".format(sum_reward))
  pickle.dump(states, open(os.path.join(os.path.dirname(FLAGS.logdir), 'states_0.pkl'), 'wb'))
  logging.info("Data logged to: {}".format(os.path.join(os.path.dirname(FLAGS.logdir), 'states_0.pkl')))

if __name__ == "__main__":
  app.run(main)
