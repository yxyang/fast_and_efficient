# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from absl import logging
from typing import Any, Sequence, Tuple

import numpy as np
import pybullet as p  # pytype: disable=import-error
import sys

from src.robots.motors import MotorCommand
from src.convex_mpc_controller import gait_generator as gait_generator_lib

try:
  import mpc_osqp as convex_mpc  # pytype: disable=import-error
except:  #pylint: disable=W0702
  print("You need to install fast_and_efficient")
  print("Run python3 setup.py install --user in this repo")
  sys.exit()

_FORCE_DIMENSION = 3
# The QP weights in the convex MPC formulation. See the MIT paper for details:
#   https://ieeexplore.ieee.org/document/8594448/
# Intuitively, this is the weights of each state dimension when tracking a
# desired CoM trajectory. The full CoM state is represented by
# (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder).

# Best results
_MPC_WEIGHTS = (1., 1., 0, 0, 0, 10, 0., 0., .1, .1, .1, .0, 0)

# These weights also give good results.
# _MPC_WEIGHTS = (1., 1., 0, 0, 0, 20, 0., 0., .1, 1., 1., .0, 0.)

PLANNING_HORIZON_STEPS = 10
PLANNING_TIMESTEP = 0.025


class TorqueStanceLegController:
  """A torque based stance leg controller framework.

  Takes in high level parameters like walking speed and turning speed, and
  generates necessary the torques for stance legs.
  """
  def __init__(
      self,
      robot: Any,
      gait_generator: Any,
      state_estimator: Any,
      desired_speed: Tuple[float, float] = (0, 0),
      desired_twisting_speed: float = 0,
      desired_body_height: float = 0.45,
      body_mass: float = 220 / 9.8,
      body_inertia: Tuple[float, float, float, float, float, float, float,
                          float, float] = (0.07335, 0, 0, 0, 0.25068, 0, 0, 0,
                                           0.25447),
      num_legs: int = 4,
      friction_coeffs: Sequence[float] = (0.6, 0.6, 0.6, 0.6),
  ):
    """Initializes the class.

    Tracks the desired position/velocity of the robot by computing proper joint
    torques using MPC module.

    Args:
      robot: A robot instance.
      gait_generator: Used to query the locomotion phase and leg states.
      state_estimator: Estimate the robot states (e.g. CoM velocity).
      desired_speed: desired CoM speed in x-y plane.
      desired_twisting_speed: desired CoM rotating speed in z direction.
      desired_body_height: The standing height of the robot.
      body_mass: The total mass of the robot.
      body_inertia: The inertia matrix in the body principle frame. We assume
        the body principle coordinate frame has x-forward and z-up.
      num_legs: The number of legs used for force planning.
      friction_coeffs: The friction coeffs on the contact surfaces.
    """
    # TODO: add acc_weight support for mpc torque stance controller.
    self._robot = robot
    self._gait_generator = gait_generator
    self._state_estimator = state_estimator
    self.desired_speed = desired_speed
    self.desired_twisting_speed = desired_twisting_speed

    self._desired_body_height = desired_body_height
    self._body_mass = body_mass
    self._num_legs = num_legs
    self._friction_coeffs = np.array(friction_coeffs)
    if np.any(np.isclose(self._friction_coeffs, 1.)):
      raise ValueError("self._cpp_mpc.compute_contact_forces seg faults when "
                       "a friction coefficient is equal to 1.")
    body_inertia_list = list(body_inertia)
    self._body_inertia_list = body_inertia_list
    weights_list = list(_MPC_WEIGHTS)
    self._weights_list = weights_list
    self._cpp_mpc = convex_mpc.ConvexMpc(body_mass, body_inertia_list,
                                         self._num_legs,
                                         PLANNING_HORIZON_STEPS,
                                         PLANNING_TIMESTEP, weights_list, 1e-5,
                                         convex_mpc.QPOASES)
    self._future_contact_estimate = np.ones((PLANNING_HORIZON_STEPS, 4))

  def reset(self, current_time):
    del current_time
    # Re-construct CPP solver to remove stochasticity due to warm-start
    self._cpp_mpc = convex_mpc.ConvexMpc(self._body_mass,
                                         self._body_inertia_list,
                                         self._num_legs,
                                         PLANNING_HORIZON_STEPS,
                                         PLANNING_TIMESTEP, self._weights_list,
                                         1e-5, convex_mpc.QPOASES)

  def update(self, current_time, future_contact_estimate=None):
    del current_time
    self._future_contact_estimate = future_contact_estimate

  def get_action(self):
    """Computes the torque for stance legs."""
    desired_com_position = np.array((0., 0., self._desired_body_height),
                                    dtype=np.float64)
    desired_com_velocity = np.array(
        (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)
    # Walk parallel to the ground
    desired_com_roll_pitch_yaw = np.zeros(3)
    desired_com_angular_velocity = np.array(
        (0., 0., self.desired_twisting_speed), dtype=np.float64)
    foot_contact_state = np.array(
        [(leg_state in (gait_generator_lib.LegState.STANCE,
                        gait_generator_lib.LegState.EARLY_CONTACT,
                        gait_generator_lib.LegState.LOSE_CONTACT))
         for leg_state in self._gait_generator.leg_state],
        dtype=np.int32)
    if not foot_contact_state.any():
      # logging.info("No foot in contact...")
      return {}, None

    if self._future_contact_estimate is not None:
      contact_estimates = self._future_contact_estimate.copy()
      contact_estimates[0] = foot_contact_state
      # print(contact_estimates)
      # input("Any Key...")
    else:
      contact_estimates = np.array([foot_contact_state] *
                                   PLANNING_HORIZON_STEPS)

    # com_position = np.array(self._robot.base_position)
    com_position = np.array(self._state_estimator.com_position_ground_frame)

    # We use the body yaw aligned world frame for MPC computation.
    # com_roll_pitch_yaw = np.array(self._robot.base_orientation_rpy,
    #                               dtype=np.float64)
    com_roll_pitch_yaw = np.array(
        p.getEulerFromQuaternion(
            self._state_estimator.com_orientation_quat_ground_frame))
    # print("Com Position: {}".format(com_position))
    com_roll_pitch_yaw[2] = 0
    # gravity_projection_vec = np.array([0., 0., 1.])
    gravity_projection_vec = np.array(
        self._state_estimator.gravity_projection_vector)
    predicted_contact_forces = [0] * self._num_legs * _FORCE_DIMENSION
    # print("Com RPY: {}".format(com_roll_pitch_yaw))
    # print("Com pos: {}".format(com_position))
    # print("Com Vel: {}".format(
    #         self._state_estimator.com_velocity_ground_frame))
    # print("Ground orientation_world_frame: {}".format(
    #     p.getEulerFromQuaternion(
    #         self._state_estimator.ground_orientation_world_frame)))
    # print("Gravity projection: {}".format(gravity_projection_vec))
    # print("Com RPY Rate: {}".format(self._robot.base_rpy_rate))
    p.submitProfileTiming("predicted_contact_forces")
    predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
        com_position,  #com_position
        np.asarray(self._state_estimator.com_velocity_ground_frame,
                   dtype=np.float64),  #com_velocity
        np.array(com_roll_pitch_yaw, dtype=np.float64),  #com_roll_pitch_yaw
        gravity_projection_vec,  # Normal Vector of ground
        # Angular velocity in the yaw aligned world frame is actually different
        # from rpy rate. We use it here as a simple approximation.
        # np.asarray(self._state_estimator.com_rpy_rate_ground_frame,
        #            dtype=np.float64),  #com_angular_velocity
        self._robot.base_angular_velocity_body_frame,
        np.asarray(contact_estimates,
                   dtype=np.float64).flatten(),  # Foot contact states
        np.array(self._robot.foot_positions_in_base_frame.flatten(),
                 dtype=np.float64),  #foot_positions_base_frame
        self._friction_coeffs,  #foot_friction_coeffs
        desired_com_position,  #desired_com_position
        desired_com_velocity,  #desired_com_velocity
        desired_com_roll_pitch_yaw,  #desired_com_roll_pitch_yaw
        desired_com_angular_velocity  #desired_com_angular_velocity
    )
    p.submitProfileTiming()

    # sol = np.array(predicted_contact_forces).reshape((-1, 12))
    # x_dim = np.array([0, 3, 6, 9])
    # y_dim = x_dim + 1
    # z_dim = y_dim + 1

    # logging.info("X_forces: {}".format(-sol[:5, x_dim]))
    # logging.info("Y_forces: {}".format(-sol[:5, y_dim]))
    # logging.info("Z_forces: {}".format(-sol[:5, z_dim]))
    # import pdb
    # pdb.set_trace()
    # input("Any Key...")

    contact_forces = {}
    for i in range(self._num_legs):
      contact_forces[i] = np.array(
          predicted_contact_forces[i * _FORCE_DIMENSION:(i + 1) *
                                   _FORCE_DIMENSION])
    # print(contact_forces)
    # input("Any Key...")

    action = {}
    for leg_id, force in contact_forces.items():
      # While "Lose Contact" is useful in simulation, in real environment it's
      # susceptible to sensor noise. Disabling for now.
      motor_torques = self._robot.map_contact_force_to_joint_torques(
          leg_id, force)
      for joint_id, torque in motor_torques.items():
        action[joint_id] = MotorCommand(desired_position=0,
                                        kp=0,
                                        desired_velocity=0,
                                        kd=0,
                                        desired_extra_torque=torque)
    # print("After IK: {}".format(time.time() - start_time))
    return action, contact_forces
