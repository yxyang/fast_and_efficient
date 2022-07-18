"""A model based controller framework."""
from absl import logging

from datetime import datetime
import enum
import ml_collections
import numpy as np
import os
import pickle
import pybullet
from pybullet_utils import bullet_client
import threading
import time
from typing import Tuple

from src.convex_mpc_controller import com_velocity_estimator
from src.convex_mpc_controller import offset_gait_generator
from src.convex_mpc_controller import raibert_swing_leg_controller
from src.convex_mpc_controller import torque_stance_leg_controller_mpc
from src.convex_mpc_controller.gait_configs import crawl, trot, flytrot
from src.robots import a1
from src.robots import a1_robot
from src.robots.motors import MotorCommand
from src.robots.motors import MotorControlMode
from src.worlds import abstract_world, plane_world


class ControllerMode(enum.Enum):
  DOWN = 1
  STAND = 2
  WALK = 3
  TERMINATE = 4


class GaitType(enum.Enum):
  CRAWL = 1
  TROT = 2
  FLYTROT = 3


def get_sim_conf():
  config = ml_collections.ConfigDict()
  config.timestep: float = 0.002
  config.action_repeat: int = 1
  config.reset_time_s: float = 3.
  config.num_solver_iterations: int = 30
  config.init_position: Tuple[float, float, float] = (0., 0., 0.32)
  config.init_rack_position: Tuple[float, float, float] = [0., 0., 1]
  config.on_rack: bool = False
  return config


class LocomotionController(object):
  """Generates the quadruped locomotion.

  The actual effect of this controller depends on the composition of each
  individual subcomponent.

  """
  def __init__(
      self,
      use_real_robot: bool = False,
      show_gui: bool = False,
      logdir: str = 'logs/',
      world_class: abstract_world.AbstractWorld = plane_world.PlaneWorld):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the leg swing/stance pattern.
      state_estimator: Estimates the state of the robot (e.g. center of mass
        position or velocity that may not be observable from sensors).
      swing_leg_controller: Generates motor actions for swing legs.
      stance_leg_controller: Generates motor actions for stance legs.
      clock: A real or fake clock source.
    """
    self._use_real_robot = use_real_robot
    self._show_gui = show_gui
    self._world_class = world_class
    self._setup_robot_and_controllers()
    self.reset_robot()
    self.reset_controllers()
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._logs = []
    self._logdir = logdir

    self._mode = ControllerMode.DOWN
    self.set_controller_mode(ControllerMode.STAND)
    self._gait = None
    self._desired_gait = GaitType.CRAWL
    self._handle_gait_switch()
    self.run_thread = threading.Thread(target=self.run)
    self.run_thread.start()

  def _setup_robot_and_controllers(self):
    # Construct robot
    if self._show_gui and not self._use_real_robot:
      p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
      p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setAdditionalSearchPath('src/data')

    self.pybullet_client = p
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.002)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    world = self._world_class(self.pybullet_client)
    world.build_world()

    # Construct robot class:
    if self._use_real_robot:
      self._robot = a1_robot.A1Robot(
          pybullet_client=p,
          sim_conf=get_sim_conf(),
          motor_control_mode=MotorControlMode.HYBRID)
    else:
      self._robot = a1.A1(pybullet_client=p,
                          sim_conf=get_sim_conf(),
                          motor_control_mode=MotorControlMode.HYBRID)

    if self._show_gui and not self._use_real_robot:
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    self._clock = lambda: self._robot.time_since_reset

    self._gait_generator = offset_gait_generator.OffsetGaitGenerator(
        self._robot, [0., np.pi, np.pi, 0.])

    desired_speed, desired_twisting_speed = (0., 0.), 0.

    self._state_estimator = com_velocity_estimator.COMVelocityEstimator(
        self._robot, velocity_window_size=60, ground_normal_window_size=10)

    self._swing_controller = \
      raibert_swing_leg_controller.RaibertSwingLegController(
          self._robot,
          self._gait_generator,
          self._state_estimator,
          desired_speed=desired_speed,
          desired_twisting_speed=desired_twisting_speed,
          desired_height=self._robot.mpc_body_height,
          foot_landing_clearance=0.01,
          foot_height=0.1,
          use_raibert_heuristic=True)

    mpc_friction_coef = 0.6
    self._stance_controller = \
      torque_stance_leg_controller_mpc.TorqueStanceLegController(
          self._robot,
          self._gait_generator,
          self._state_estimator,
          desired_speed=(desired_speed[0], desired_speed[1]),
          desired_twisting_speed=desired_twisting_speed,
          desired_body_height=self._robot.mpc_body_height,
          body_mass=self._robot.mpc_body_mass,
          body_inertia=self._robot.mpc_body_inertia,
          friction_coeffs=np.ones(4) * mpc_friction_coef)

  @property
  def swing_leg_controller(self):
    return self._swing_controller

  @property
  def stance_leg_controller(self):
    return self._stance_controller

  @property
  def gait_generator(self):
    return self._gait_generator

  @property
  def state_estimator(self):
    return self._state_estimator

  @property
  def time_since_reset(self):
    return self._time_since_reset

  def reset_robot(self):
    self._robot.reset(hard_reset=False)
    if self._show_gui and not self._use_real_robot:
      self.pybullet_client.configureDebugVisualizer(
          self.pybullet_client.COV_ENABLE_RENDERING, 1)

  def reset_controllers(self):
    # Resetting other components
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._gait_generator.reset()
    self._state_estimator.reset(self._time_since_reset)
    self._swing_controller.reset(self._time_since_reset)
    self._stance_controller.reset(self._time_since_reset)

  def update(self):
    self._time_since_reset = self._clock() - self._reset_time
    self._gait_generator.update()
    self._state_estimator.update(self._gait_generator.desired_leg_state)
    self._swing_controller.update(self._time_since_reset)
    future_contact_estimate = self._gait_generator.get_estimated_contact_states(
        torque_stance_leg_controller_mpc.PLANNING_HORIZON_STEPS,
        torque_stance_leg_controller_mpc.PLANNING_TIMESTEP)
    self._stance_controller.update(self._time_since_reset,
                                   future_contact_estimate)

  def get_action(self):
    """Returns the control ouputs (e.g. positions/torques) for all motors."""
    swing_action = self._swing_controller.get_action()
    stance_action, qp_sol = self._stance_controller.get_action()

    actions = []
    for joint_id in range(self._robot.num_motors):
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

  def _get_stand_action(self):
    return MotorCommand(
        desired_position=self._robot.motor_group.init_positions,
        kp=self._robot.motor_group.kps,
        desired_velocity=0,
        kd=self._robot.motor_group.kds,
        desired_extra_torque=0)

  def _handle_mode_switch(self):
    if self._mode == self._desired_mode:
      return
    self._mode = self._desired_mode
    if self._desired_mode == ControllerMode.DOWN:
      logging.info("Entering joint damping mode.")
      self._flush_logging()
    elif self._desired_mode == ControllerMode.STAND:
      logging.info("Standing up.")
      self.reset_robot()
    else:
      logging.info("Walking.")
      self.reset_controllers()
      self._start_logging()

  def _start_logging(self):
    self._logs = []

  def _update_logging(self, action, qp_sol):
    frame = dict(
        desired_speed=(self._swing_controller.desired_speed,
                       self._swing_controller.desired_twisting_speed),
        timestamp=self._time_since_reset,
        base_rpy=self._robot.base_orientation_rpy,
        motor_angles=self._robot.motor_angles,
        base_vel=self._robot.motor_velocities,
        base_vels_body_frame=self._state_estimator.com_velocity_body_frame,
        base_angular_velocity_body_frame=self._robot.
        base_angular_velocity_body_frame,
        motor_vels=self._robot.motor_velocities,
        motor_torques=self._robot.motor_torques,
        contacts=self._robot.foot_contacts,
        desired_grf=qp_sol,
        robot_action=action,
        gait_generator_phase=self._gait_generator.current_phase.copy(),
        gait_generator_state=self._gait_generator.leg_state,
        ground_orientation=self._state_estimator.
        ground_orientation_world_frame,
    )
    self._logs.append(frame)

  def _flush_logging(self):
    if not os.path.exists(self._logdir):
      os.makedirs(self._logdir)
    filename = 'log_{}.pkl'.format(
        datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    pickle.dump(self._logs, open(os.path.join(self._logdir, filename), 'wb'))
    logging.info("Data logged to: {}".format(
        os.path.join(self._logdir, filename)))

  def _handle_gait_switch(self):
    if self._gait == self._desired_gait:
      return
    if self._desired_gait == GaitType.CRAWL:
      logging.info("Switched to Crawling gait.")
      self._gait_config = crawl.get_config()
    elif self._desired_gait == GaitType.TROT:
      logging.info("Switched  to Trotting gait.")
      self._gait_config = trot.get_config()
    else:
      logging.info("Switched to Fly-Trotting gait.")
      self._gait_config = flytrot.get_config()

    self._gait = self._desired_gait
    self._gait_generator.gait_params = self._gait_config.gait_parameters
    self._swing_controller.foot_height = self._gait_config.foot_clearance_max
    self._swing_controller.foot_landing_clearance = \
      self._gait_config.foot_clearance_land

  def run(self):
    logging.info("Low level thread started...")
    while True:
      self._handle_mode_switch()
      self._handle_gait_switch()
      self.update()
      if self._mode == ControllerMode.DOWN:
        time.sleep(0.1)
      elif self._mode == ControllerMode.STAND:
        action = self._get_stand_action()
        self._robot.step(action)
        time.sleep(0.001)
      elif self._mode == ControllerMode.WALK:
        action, qp_sol = self.get_action()
        self._robot.step(action)
        self._update_logging(action, qp_sol)
      else:
        logging.info("Running loop terminated, exiting...")
        break

      # Camera setup:
      if self._show_gui:
        self.pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=30 + self._robot.base_orientation_rpy[2] / np.pi * 180,
            cameraPitch=-30,
            cameraTargetPosition=self._robot.base_position,
        )

  def set_controller_mode(self, mode):
    self._desired_mode = mode

  def set_gait(self, gait):
    self._desired_gait = gait

  @property
  def is_safe(self):
    if self.mode != ControllerMode.WALK:
      return True
    rot_mat = np.array(
        self._robot.pybullet_client.getMatrixFromQuaternion(
            self._state_estimator.com_orientation_quat_ground_frame)).reshape(
                (3, 3))
    up_vec = rot_mat[2, 2]
    base_height = self._robot.base_position[2]
    return up_vec > 0.85 and base_height > 0.18

  @property
  def mode(self):
    return self._mode

  def set_desired_speed(self, desired_lin_speed_ratio,
                        desired_rot_speed_ratio):
    desired_lin_speed = (
        self._gait_config.max_forward_speed * desired_lin_speed_ratio[0],
        self._gait_config.max_side_speed * desired_lin_speed_ratio[1],
        0,
    )
    desired_rot_speed = \
      self._gait_config.max_rot_speed * desired_rot_speed_ratio
    self._swing_controller.desired_speed = desired_lin_speed
    self._swing_controller.desired_twisting_speed = desired_rot_speed
    self._stance_controller.desired_speed = desired_lin_speed
    self._stance_controller.desired_twisting_speed = desired_rot_speed

  def set_gait_parameters(self, gait_parameters):
    raise NotImplementedError()

  def set_qp_weight(self, qp_weight):
    raise NotImplementedError()

  def set_mpc_mass(self, mpc_mass):
    raise NotImplementedError()

  def set_mpc_inertia(self, mpc_inertia):
    raise NotImplementedError()

  def set_mpc_foot_friction(self, mpc_foot_friction):
    raise NotImplementedError()

  def set_foot_landing_clearance(self, foot_landing_clearance):
    raise NotImplementedError()
