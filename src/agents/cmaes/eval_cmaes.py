"""Code to evaluate a learned ARS policy.
"""
from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import numpy as np
import gym
import os
import pickle
import time
from tqdm import tqdm
import yaml

flags.DEFINE_string('logdir', '/path/to/log/dir', 'path to log dir.')
flags.DEFINE_bool('show_gui', False, 'whether to show pybullet GUI.')
flags.DEFINE_bool('save_video', False, 'whether to save video.')
flags.DEFINE_bool('save_data', True, 'whether to save data.')
flags.DEFINE_integer('num_rollouts', 1, 'number of rollouts.')
flags.DEFINE_integer('rollout_length', 0, 'rollout_length, 0 for default.')
flags.DEFINE_bool('use_real_robot', False, 'whether to use real robot.')
flags.DEFINE_bool('use_gamepad', False,
                  'whether to use gamepad for speed command.')
FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
  files = [
      entry for entry in os.listdir(logdir)
      if os.path.isfile(os.path.join(logdir, entry))
  ]
  files.sort(key=lambda entry: os.path.getmtime(os.path.join(logdir, entry)))
  files = files[::-1]

  idx = 0
  for entry in files:
    if entry.startswith('lin_policy_plus'):
      return os.path.join(logdir, entry)
  raise ValueError('No Valid Policy Found.')


def main(_):
  # Load config and policy
  if FLAGS.logdir.endswith('npz'):
    config_path = os.path.join(os.path.dirname(FLAGS.logdir), 'config.yaml')
    policy_path = FLAGS.logdir
    log_path = os.path.dirname(FLAGS.logdir)
  else:
    # Find the latest policy ckpt
    config_path = os.path.join(FLAGS.logdir, 'config.yaml')
    policy_path = get_latest_policy_path(FLAGS.logdir)
    log_path = FLAGS.logdir

  if FLAGS.save_video or FLAGS.save_data:
    log_path = os.path.join(log_path,
                            datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(log_path)

  with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    config.env_args.show_gui = FLAGS.show_gui
    config.env_args.use_real_robot = FLAGS.use_real_robot
    config.env_args.use_gamepad_speed_command = FLAGS.use_gamepad

  if FLAGS.rollout_length:
    config.rollout_length = FLAGS.rollout_length
  env = config.env_constructor(**config.env_args)

  # Load Policy
  lin_policy = dict(np.load(policy_path, allow_pickle=True))
  lin_policy = list(lin_policy.items())[0][1]
  M = lin_policy[0]
  # mean and std of state vectors estimated online by ARS.
  mean = lin_policy[1]
  std = lin_policy[2]

  policy_params = {
      'ob_filter': config.filter,
      'ob_dim': env.observation_space.low.shape[0],
      'ac_dim': env.action_space.low.shape[0]
  }
  policy = config.policy_constructor(config, policy_params,
                                     env.observation_space, env.action_space)
  policy.update_weights(M)
  if config.filter != 'NoFilter':
    policy.observation_filter.mean = mean
    policy.observation_filter.std = std

  returns = []
  observations = []
  actions = []
  renders = []
  episode_lengths = []
  max_placement = 0

  obs = env.reset()
  done = False
  totalr = 0.
  steps = 0
  p = env.pybullet_client
  if FLAGS.save_video:
    video_dir = os.path.join(log_path, 'video.mp4')
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_dir)
  states = []
  for t in range(config.rollout_length):
    start_time = time.time()
    action = policy.act(obs)

    observations.append(obs)
    actions.append(action)
    rew = 0
    for _ in range(int(env.config.high_level_dt / env.robot.control_timestep)):
      obs, step_rew, done, _ = env.step(action, single_step=True)

      rew += step_rew
      states.append(
          dict(
              desired_speed=env.get_desired_speed(env.robot.time_since_reset),
              timestamp=env.robot.time_since_reset,
              base_rpy=env.robot.base_orientation_rpy,
              motor_angles=env.robot.motor_angles,
              base_vel=env.robot.base_velocity,
              base_vels_body_frame=env.state_estimator.com_velocity_body_frame,
              base_angular_velocity_body_frame=env.robot.
              base_angular_velocity_body_frame,
              motor_vels=env.robot.motor_velocities,
              motor_torques=env.robot.motor_torques,
              contacts=env.robot.foot_contacts,
              desired_grf=env.qp_sol,
              reward=step_rew,
              state=obs,
              robot_action=env.robot_action,
              env_action=action,
              gait_generator_phase=env.gait_generator.current_phase.copy(),
              gait_generator_state=env.gait_generator.leg_state))
      if done:
        break
    totalr += rew
    print("Step: {}, Rew: {}".format(t, rew))
    steps += 1
    if done:
      break

    duration = time.time() - start_time
    if duration < env.robot.control_timestep and not FLAGS.use_real_robot:
      time.sleep(env.robot.control_timestep - duration)

  if FLAGS.save_video:
    p.stopStateLogging(log_id)

  if FLAGS.save_data:
    pickle.dump(states, open(os.path.join(log_path, 'states_0.pkl'), 'wb'))
    logging.info("Data logged to: {}".format(log_path))
  print(totalr)
  # print("End Phase: {}".format(env.gait_generator.current_phase))
  episode_lengths.append(steps)
  returns.append(totalr)

  print('episode lengths', episode_lengths)
  print('returns', returns)


if __name__ == '__main__':
  app.run(main)
