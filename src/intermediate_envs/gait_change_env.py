"""A gym environment that changes the takes in gait schedule commands."""
from absl import logging
import gym
import ml_collections
import numpy as np
import scipy
import pybullet
from pybullet_utils import bullet_client
from typing import Tuple

from src.robots.motors import MotorControlMode
from src.robots import a1
from src.robots import a1_robot
from src.robots import gamepad_reader
from src.robots.motors import MotorCommand
from src.convex_mpc_controller import com_velocity_estimator
from src.convex_mpc_controller import raibert_swing_leg_controller as swing_controller
from src.convex_mpc_controller import torque_stance_leg_controller_mpc as stance_controller_mpc
from src.convex_mpc_controller import offset_gait_generator


def get_gamepad_desired_speed_fn(gamepad):
  def get_desired_speed(time_since_reset):
    lin_speed, rot_speed, _ = gamepad.get_command(time_since_reset)
    return np.array([lin_speed[0], lin_speed[1], 0, rot_speed])

  return get_desired_speed


def _get_sim_conf():
  config = ml_collections.ConfigDict()
  config.timestep: float = 0.002
  config.action_repeat: int = 1
  config.reset_time_s: float = 3.
  config.num_solver_iterations: int = 30
  config.init_position: Tuple[float, float, float] = (0., 0., 0.32)
  config.init_rack_position: Tuple[float, float, float] = [0., 0., 1]
  config.on_rack: bool = False
  return config


class GaitChangeEnv(gym.Env):
  """A gym wrapper over the low level leg controller."""
  def __init__(self,
               config,
               use_real_robot=False,
               use_gamepad_speed_command=False,
               show_gui=False):
    self.config = config
    self.observation_space = self._construct_observation_space()
    self.action_space = self._construct_action_space()
    self.show_gui = show_gui
    self.use_real_robot = use_real_robot

    x, y = self.config.speed_profile
    self.get_desired_speed = scipy.interpolate.interp1d(
        x, y, kind="linear", fill_value="extrapolate", axis=0)

    # Construct robot
    if show_gui and not use_real_robot:
      p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
      p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setAdditionalSearchPath('src/data')

    self.pybullet_client = p
    if use_real_robot or use_gamepad_speed_command:
      self.gamepad = gamepad_reader.Gamepad(vel_scale_x=2.5, vel_scale_y=0.3)
    if use_gamepad_speed_command:
      self.get_desired_speed = get_gamepad_desired_speed_fn(self.gamepad)

    # Construct robot class:
    if use_real_robot:
      self.robot = a1_robot.A1Robot(pybullet_client=p,
                                    sim_conf=_get_sim_conf(),
                                    motor_control_mode=MotorControlMode.HYBRID)
    else:
      self.robot = a1.A1(pybullet_client=p,
                         sim_conf=_get_sim_conf(),
                         motor_control_mode=MotorControlMode.HYBRID)

    if show_gui and not use_real_robot:
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # Construct controller
    self._clock = lambda: self.robot.time_since_reset
    self._gait_generator = offset_gait_generator.OffsetGaitGenerator(
        self.robot, config.init_phase)
    window_size = 60
    self._state_estimator = com_velocity_estimator.COMVelocityEstimator(
        self.robot,
        velocity_window_size=window_size,
        ground_normal_window_size=10)

    friction_coef = self.config.get("mpc_foot_friction", 0.2)
    self._stance_controller = stance_controller_mpc.TorqueStanceLegController(
        self.robot,
        self._gait_generator,
        self._state_estimator,
        desired_speed=(0, 0),
        desired_twisting_speed=0,
        desired_body_height=self.robot.mpc_body_height,
        body_mass=self.robot.mpc_body_mass,
        body_inertia=self.robot.mpc_body_inertia,
        friction_coeffs=np.ones(4) * friction_coef,
    )

    # Since the swing controller is not aware of swing_duration, we cannot
    # use the raibert heuristic here.
    self._swing_controller = swing_controller.RaibertSwingLegController(
        self.robot,
        self._gait_generator,
        self._state_estimator,
        desired_speed=(0, 0),
        desired_twisting_speed=0,
        desired_height=self.robot.mpc_body_height,
        foot_height=0.1,
        foot_landing_clearance=0.,
        use_raibert_heuristic=True,
    )
    self.desired_yaw = 0.
    self.reset()

  def reset(self):
    p = self.pybullet_client
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.002)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath('src/data')

    self.ground_id = p.loadURDF(
        self.config.get('terrain_urdf_name', 'plane.urdf'))
    self.pybullet_client.changeDynamics(self.ground_id, -1, restitution=1)
    self.pybullet_client.changeDynamics(self.ground_id, -1, lateralFriction=0.5)

    self.robot.reset(hard_reset=True)
    if self.show_gui and not self.use_real_robot:
      self.pybullet_client.configureDebugVisualizer(
          self.pybullet_client.COV_ENABLE_RENDERING, 1)
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._gait_generator.reset()
    self._state_estimator.reset(self._time_since_reset)
    self._swing_controller.reset(self._time_since_reset)
    self._stance_controller.reset(self._time_since_reset)

    # Reset MPC Solver
    return self.get_observation()

  def update_desired_speed(self, lin_speed, ang_speed):
    self._swing_controller.desired_speed = lin_speed
    self._swing_controller.desired_twisting_speed = ang_speed
    self._stance_controller.desired_speed = lin_speed
    self._stance_controller.desired_twisting_speed = ang_speed

  @property
  def action_types(self):
    return self.config.get('action_types', ACTION_TYPES)

  def step(self, action, single_step=False):
    if single_step:
      num_steps = 1
    else:
      num_steps = int(self.config.high_level_dt / self.robot.control_timestep)

    sum_reward = 0
    for _ in range(num_steps):
      self._time_since_reset = self._clock() - self._reset_time
      # Update individual controller components
      desired_speed = self.get_desired_speed(self._time_since_reset)
      # Perform "Position" control over yaw direction.
      # desired_yaw_rate = -self.robot.base_orientation_rpy[2]
      # desired_speed[3] = desired_yaw_rate

      lin_speed, ang_speed = desired_speed[:3], desired_speed[3:]
      self.update_desired_speed(lin_speed, ang_speed)
      gait_params = np.concatenate((
          action[0:1],
          np.arctan2(action[1:4], action[4:7]),
          action[7:],
      ))
      self._gait_generator.gait_params = gait_params
      self._gait_generator.update()
      self._state_estimator.update(self._gait_generator.desired_leg_state)

      future_contacts = self._gait_generator.get_estimated_contact_states(
          stance_controller_mpc.PLANNING_HORIZON_STEPS,
          stance_controller_mpc.PLANNING_TIMESTEP,
      )

      self._swing_controller.update(self._time_since_reset)
      self._stance_controller.update(self._time_since_reset,
                                     future_contact_estimate=future_contacts)

      # Get robot action and step the robot
      self.robot_action, self.qp_sol = self.get_robot_action()
      self.robot.step(self.robot_action)

      if self.show_gui:
        self.pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=30 + self.robot.base_orientation_rpy[2] / np.pi * 180,
            cameraPitch=-30,
            cameraTargetPosition=self.robot.base_position,
        )
      sum_reward += self._reward_fn(action)
      done = not self.is_safe
      if done:
        logging.info("Unsafe, terminating episode...")
        break
    return self.get_observation(), sum_reward, done, dict()

  def _reward_fn(self, action):
    # del action # unused
    desired_speed = self.get_desired_speed(self._time_since_reset)[0]
    if self.use_real_robot:
      actual_speed = self.robot.base_velocity[0]
    else:
      actual_speed = self.robot.base_velocity[0]

    motor_heat = 0.3 * self.robot.motor_torques**2
    motor_mech = self.robot.motor_torques * self.robot.motor_velocities

    # Maximize over 0 since the battery can not be charged by the motor yet
    power_penalty = np.maximum(motor_heat + motor_mech, 0)
    power_penalty = np.sum(power_penalty)
    if self.config.get('use_cot', False):
      power_penalty /= np.maximum(desired_speed, 0.3)

    action_norm_penalty = np.sum(np.maximum(np.abs(action[1:7]) - 0.5, 0))

    alive_bonus = self.config.get('alive_bonus', 3.)

    speed_penalty_type = self.config.get('speed_penalty_type',
                                         'symmetric_square')
    if speed_penalty_type == 'symmetric_square':
      speed_penalty = (desired_speed - actual_speed)**2
    elif speed_penalty_type == 'asymmetric_square':
      speed_penalty = np.maximum(desired_speed - actual_speed, 0)**2
    elif speed_penalty_type == 'soft_symmetric_square':
      speed_diff = np.abs(desired_speed - actual_speed)
      speed_penalty = np.maximum(speed_diff - 0.2, 0)**2
    else:
      speed_diff = np.abs(desired_speed - actual_speed) / np.maximum(
          actual_speed, 0.3)
      speed_diff = np.clip(speed_diff, -1, 1)
      speed_penalty = speed_diff**2

    # rew = alive_bonus - power_penalty * 0.0025 - np.maximum(
    #     (desired_speed - actual_speed), 0
    # )**2 - action_norm_penalty * self.config.get('action_penalty_weight', 0)
    rew = alive_bonus - \
        power_penalty * self.config.get('power_penalty_weight', 0.0025) - \
        speed_penalty * self.config.get('speed_penalty_weight', 1) - \
        action_norm_penalty * self.config.get('action_penalty_weight', 0)
    return rew

  @property
  def is_safe(self):
    rot_mat = np.array(
        self.robot.pybullet_client.getMatrixFromQuaternion(
            self._state_estimator.com_orientation_quat_ground_frame)).reshape(
                (3, 3))
    up_vec = rot_mat[2, 2]
    base_height = self.robot.base_position[2]
    if self.use_real_robot:
      return (up_vec > 0.85 and base_height > 0.18
              and (not self.gamepad.estop_flagged))
    else:
      return up_vec > 0.85 and base_height > 0.18

  def get_robot_action(self):
    swing_action = self._swing_controller.get_action()
    stance_action, qp_sol = self._stance_controller.get_action()
    actions = []
    for joint_id in range(self.robot.num_motors):
      if joint_id in swing_action:
        actions.append(swing_action[joint_id])
      else:
        assert joint_id in stance_action
        actions.append(stance_action[joint_id])

    vectorized_action = MotorCommand(
        desired_position=[action.desired_position for action in actions],
        kp=[action.kp for action in actions],
        desired_velocity=[action.desired_velocity for action in actions],
        kd=[action.kd for action in actions],
        desired_extra_torque=[
            action.desired_extra_torque for action in actions
        ])

    return vectorized_action, dict(qp_sol=qp_sol)

  def _construct_observation_space(self):
    if self.config.use_full_observation:
      obs_low = -1 * np.ones(39)
      obs_high = 1 * np.ones(39)
    else:
      obs_low = -1 * np.ones(4)
      obs_high = 1 * np.ones(4)
    return gym.spaces.Box(obs_low, obs_high)

  def _construct_action_space(self):
    return gym.spaces.Box(self.config.action_low,
                          self.config.action_high)

  def get_observation(self):
    gait_generator_state = self._gait_generator.get_observation()  # 16
    base_height = self.robot.base_position[2:]
    robot_orientation = self.robot.base_orientation_rpy
    robot_velocity = self.robot.base_velocity
    robot_rpy_rate = self.robot.base_rpy_rate  # 3
    foot_position = self.robot.foot_positions_in_base_frame.flatten()  # 12
    desired_velocity = self.get_desired_speed(self._time_since_reset)

    if self.config.use_full_observation:
      return np.concatenate(
          (
              gait_generator_state,  # 16
              base_height,  # 1
              robot_orientation,  # 3
              robot_velocity,  # 3
              robot_rpy_rate,  # 3
              foot_position,  # 12
              desired_velocity[:1],  # 1
          ),
          axis=-1,
      )
    else:
      return np.concatenate(
          (
              robot_velocity,  # 3
              desired_velocity[:1],  # 1
          ),
          axis=-1,
      )

  @property
  def gait_generator(self):
    return self._gait_generator

  @property
  def state_estimator(self):
    return self._state_estimator
