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

  MAX_SPEED = 3.
  config.env_args.config.speed_profile = (np.array([-0.01, 8, 10, 14]),
                                          np.array([[0., 0, 0, 0],
                                                    [MAX_SPEED, 0, 0, 0],
                                                    [MAX_SPEED, 0, 0, 0],
                                                    [MAX_SPEED, 0, 0, 0]]))
  config.rollout_length = 5000
  # config.env_args.config.speed_profile = (np.array([0, 10, 11, 12, 13]),
  #                                         np.array([[0., 0., 0., 0.],
  #                                                   [2.5, 0., 0., 0.],
  #                                                   [2.5, 0., 0., 0.],
  #                                                   [0., 0., 0., 0.],
  #                                                   [0., 0., 0., 0.]]))
  # config.rollout_length = 5000
  # config.env_args.config.speed_profile = (np.array(
  #     [0, 2, 2.01, 4, 4.01, 6, 6.01, 8, 8.01, 10, 11]),
  #                                         np.array([[0.5, 0., 0., 0.],
  #                                                   [0.5, 0., 0., 0.],
  #                                                   [1.0, 0., 0., 0.],
  #                                                   [1.0, 0., 0., 0.],
  #                                                   [1.5, 0., 0., 0.],
  #                                                   [1.5, 0., 0., 0.],
  #                                                   [2.0, 0., 0., 0.],
  #                                                   [2.0, 0., 0., 0.],
  #                                                   [2.5, 0., 0., 0.],
  #                                                   [2.5, 0., 0., 0.],
  #                                                   [0., 0., 0., 0.]]))
  # config.rollout_length = 5000

  config.env_args.show_gui = FLAGS.show_gui
  config.env_args.use_real_robot = FLAGS.use_real_robot
  if FLAGS.rollout_length:
    config.rollout_length = FLAGS.rollout_length
  env = config.env_constructor(**config.env_args)
  env = gym.wrappers.Monitor(env, './renders', force=True)

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
  policy = config.policy_constructor(config, policy_params, env.action_space)
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
  sum_energy = 0
  for t in range(config.rollout_length):
    start_time = time.time()
    action = policy.act(obs)
    # action = np.array([1.633, 0.12, -0.134, 0.198, -0.996, -0.697, 1., 0.509])
    # input("Any Key...")
    # action = np.array([2.5, 0, 0, 0, -1, 1, -1, 0.25])
    # action = np.array([3.5, 0, 0, 0, -1, -1, 1, 0.5])
    # action = np.array([2, 0, 0, 0, 1, -1, -1, 0.5])
    # off_fl, off_rr, off_rl = np.pi, np.pi * 3 / 2, np.pi / 2
    # action = np.array([
    #     2,
    #     np.sin(off_fl),
    #     np.sin(off_rr),
    #     np.sin(off_rl),
    #     np.cos(off_fl),
    #     np.cos(off_rr),
    #     np.cos(off_rl), 0.3
    # ])

    observations.append(obs)
    actions.append(action)
    # print(action)
    print(action[0], np.arctan2(action[1:4], action[4:7]), action[7])
    print(env.robot.GetBaseVelocity())
    obs, rew, done, _ = env.step(action)
    # states.append(
    #     dict(desired_speed=env.get_desired_speed(
    #         env.robot.GetTimeSinceReset()),
    #          timestamp=env.robot.GetTimeSinceReset(),
    #          base_rpy=env.robot.GetBaseRollPitchYaw(),
    #          motor_angles=env.robot.GetMotorAngles(),
    #          base_vel=env.robot.GetBaseVelocity(),
    #          base_vels_body_frame=env.state_estimator.com_velocity_body_frame,
    #          base_rpy_rate=env.robot.GetBaseRollPitchYawRate(),
    #          motor_vels=env.robot.GetMotorVelocities(),
    #          motor_torques=env.robot.GetMotorTorques(),
    #          contacts=env.robot.GetFootContacts(),
    #          desired_grf=env.qp_sol,
    #          foot_forces=env.robot.GetFootContactForces(),
    #          reward=rew,
    #          state=obs,
    #          robot_action=env.robot_action,
    #          env_action=action,
    #          base_acc=env.robot.GetBaseAcceleration(),
    #          gait_generator_phase=env.gait_generator.current_phase.copy(),
    #          gait_generator_state=env.gait_generator.leg_state))

    totalr += rew
    print("Step: {}, Rew: {}".format(t, rew))
    steps += 1
    if done:
      break

    duration = time.time() - start_time
    if duration < env.robot.control_time_step and not FLAGS.use_real_robot:
      time.sleep(env.robot.control_time_step - duration)

  if FLAGS.save_video:
    p.stopStateLogging(log_id)

  if FLAGS.save_data:
    pickle.dump(states, open(os.path.join(log_path, 'states_0.pkl'), 'wb'))
    logging.info("Data logged to: {}".format(log_path))

  env.close()
  print(totalr)
  episode_lengths.append(steps)
  returns.append(totalr)

  print('episode lengths', episode_lengths)
  print('returns', returns)


if __name__ == '__main__':
  app.run(main)
