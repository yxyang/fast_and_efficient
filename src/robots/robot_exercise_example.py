"""Example of running A1 robot with position control.

To run:
python -m src.robots.a1_robot_exercise_example.py
"""
from absl import app
from absl import flags

import ml_collections
import numpy as np
import pybullet
from pybullet_utils import bullet_client
import time
from typing import Tuple

from src.robots import a1
from src.robots import a1_robot
from src.robots import go1
from src.robots import go1_robot
from src.robots.motors import MotorCommand

flags.DEFINE_enum('robot_type', 'a1_sim',
                  ['a1_sim', 'a1_real', 'go1_sim', 'go1_real'],
                  'the robot type to use.')
FLAGS = flags.FLAGS


def get_action(robot, t):
  mid_action = np.array([0.0, 0.9, -1.8] * 4)
  amplitude = np.array([0.0, 0.2, -0.4] * 4)
  freq = 1.0
  return MotorCommand(desired_position=mid_action +
                      amplitude * np.sin(2 * np.pi * freq * t),
                      kp=robot.motor_group.kps,
                      desired_velocity=np.zeros(robot.num_motors),
                      kd=robot.motor_group.kds)


def get_sim_conf():
  config = ml_collections.ConfigDict()
  config.timestep: float = 0.002
  config.action_repeat: int = 1
  config.reset_time_s: float = 3.
  config.num_solver_iterations: int = 30
  config.init_position: Tuple[float, float, float] = (0., 0., 0.32)
  config.init_rack_position: Tuple[float, float, float] = [0., 0., 1]
  config.on_rack: bool = True
  return config


def main(_):
  if FLAGS.robot_type in ['a1_real', 'go1_real']:
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
  else:
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
  p.setAdditionalSearchPath('src/data')
  p.loadURDF("plane.urdf")
  p.setGravity(0.0, 0.0, -9.8)

  if FLAGS.robot_type == 'a1_real':
    robot = a1_robot.A1Robot(pybullet_client=p, sim_conf=get_sim_conf())
  elif FLAGS.robot_type == 'a1_sim':
    robot = a1.A1(pybullet_client=p, sim_conf=get_sim_conf())
  elif FLAGS.robot_type == 'go1_real':
    robot = go1_robot.Go1Robot(pybullet_client=p, sim_conf=get_sim_conf)
  else:
    robot = go1.Go1(pybullet_client=p, sim_conf=get_sim_conf())

  robot.reset()

  for _ in range(10000):
    action = get_action(robot, robot.time_since_reset)
    robot.step(action)
    time.sleep(0.002)
    print(robot.base_orientation_rpy)


if __name__ == "__main__":
  app.run(main)
