"""The swing leg controller class."""
# from absl import logging

import copy
import math
import numpy as np
from typing import Any, Mapping, Sequence, Tuple

from src.robots.motors import MotorCommand
from src.convex_mpc_controller import gait_generator as gait_generator_lib

# The position correction coefficients in Raibert's formula.
_KP = np.array([0.01, 0.01, 0.01]) * 0.
# At the end of swing, we leave a small clearance to prevent unexpected foot
# collision.
_FOOT_CLEARANCE_M = 0.01


def _gen_parabola(phase: float, start: float, mid: float, end: float) -> float:
  """Gets a point on a parabola y = a x^2 + b x + c.

  The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
  the plane.

  Args:
    phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
    start: The y value at x == 0.
    mid: The y value at x == 0.5.
    end: The y value at x == 1.

  Returns:
    The y value at x == phase.
  """
  mid_phase = 0.5
  delta_1 = mid - start
  delta_2 = end - start
  delta_3 = mid_phase**2 - mid_phase
  coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
  coef_b = (delta_2 * mid_phase**2 - delta_1) / delta_3
  coef_c = start

  return coef_a * phase**2 + coef_b * phase + coef_c


def _gen_swing_foot_trajectory(input_phase: float, start_pos: Sequence[float],
                               end_pos: Sequence[float],
                               foot_height: float) -> Tuple[float]:
  """Generates the swing trajectory using a parabola.

  Args:
    input_phase: the swing/stance phase value between [0, 1].
    start_pos: The foot's position at the beginning of swing cycle.
    end_pos: The foot's desired position at the end of swing cycle.

  Returns:
    The desired foot position at the current phase.
  """
  # We augment the swing speed using the below formula. For the first half of
  # the swing cycle, the swing leg moves faster and finishes 80% of the full
  # swing trajectory. The rest 20% of trajectory takes another half swing
  # cycle. Intuitely, we want to move the swing foot quickly to the target
  # landing location and stay above the ground, in this way the control is more
  # robust to perturbations to the body that may cause the swing foot to drop
  # onto the ground earlier than expected. This is a common practice similar
  # to the MIT cheetah and Marc Raibert's original controllers.
  phase = input_phase
  if input_phase <= 0.5:
    phase = 0.8 * math.sin(input_phase * math.pi)
  else:
    phase = 0.8 + (input_phase - 0.5) * 0.4

  x = (1 - phase) * start_pos[0] + phase * end_pos[0]
  y = (1 - phase) * start_pos[1] + phase * end_pos[1]
  mid = max(end_pos[2], start_pos[2]) + foot_height
  z = _gen_parabola(phase, start_pos[2], mid, end_pos[2])

  # PyType detects the wrong return type here.
  return (x, y, z)  # pytype: disable=bad-return-type


# def cubic_bezier(x0: Sequence[float], x1: Sequence[float],
#                  t: float) -> Sequence[float]:
#   progress = t**3 + 3 * t**2 * (1 - t)
#   return x0 + progress * (x1 - x0)

# def _gen_swing_foot_trajectory(input_phase: float, start_pos: Sequence[float],
#                                end_pos: Sequence[float]) -> Tuple[float]:
#   max_clearance = 0.10
#   mid_z = max(end_pos[2], start_pos[2]) + max_clearance
#   mid_pos = (start_pos + end_pos) / 2
#   mid_pos[2] = mid_z
#   if input_phase < 0.5:
#     t = input_phase * 2
#     foot_pos = cubic_bezier(start_pos, mid_pos, t)
#   else:
#     t = input_phase * 2 - 1
#     foot_pos = cubic_bezier(mid_pos, end_pos, t)
#   return foot_pos


class RaibertSwingLegController:
  """Controls the swing leg position using Raibert's formula.

  For details, please refer to chapter 2 in "Legged robbots that balance" by
  Marc Raibert. The key idea is to stablize the swing foot's location based on
  the CoM moving speed.

  """
  def __init__(self, robot: Any, gait_generator: Any, state_estimator: Any,
               desired_speed: Tuple[float,
                                    float], desired_twisting_speed: float,
               desired_height: float, foot_landing_clearance: float,
               foot_height: float, use_raibert_heuristic: bool):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the stance/swing pattern.
      state_estimator: Estiamtes the CoM speeds.
      desired_speed: Behavior parameters. X-Y speed.
      desired_twisting_speed: Behavior control parameters.
      desired_height: Desired standing height.
      foot_landing_clearance: The foot clearance on the ground at the end of
        the swing cycle.
    """
    self._robot = robot
    self._state_estimator = state_estimator
    self._gait_generator = gait_generator
    self._last_leg_state = gait_generator.desired_leg_state
    self.desired_speed = np.array((desired_speed[0], desired_speed[1], 0))
    self.desired_twisting_speed = desired_twisting_speed
    self._desired_height = desired_height
    self._desired_landing_height = np.array(
        (0, 0, desired_height - foot_landing_clearance))

    self._phase_switch_foot_local_position = None
    self.foot_placement_position = np.zeros(12)
    self.use_raibert_heuristic = use_raibert_heuristic
    self._foot_height = foot_height
    self.reset(0)

  def reset(self, current_time: float) -> None:
    """Called during the start of a swing cycle.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time
    self._last_leg_state = self._gait_generator.desired_leg_state
    self._phase_switch_foot_local_position = \
      self._robot.foot_positions_in_base_frame.copy()

  def update(self, current_time: float) -> None:
    """Called at each control step.
    Args:
      current_time: The wall time in seconds.
    """
    del current_time
    new_leg_state = self._gait_generator.desired_leg_state
    # Detects phase switch for each leg so we can remember the feet position at
    # the beginning of the swing phase.
    for leg_id, state in enumerate(new_leg_state):
      if (state == gait_generator_lib.LegState.SWING
          and state != self._last_leg_state[leg_id]):
        self._phase_switch_foot_local_position[leg_id] = (
            self._robot.foot_positions_in_base_frame[leg_id])

    self._last_leg_state = copy.deepcopy(new_leg_state)

  @property
  def foot_height(self):
    return self._foot_height

  @foot_height.setter
  def foot_height(self, foot_height: float) -> None:
    self._foot_height = foot_height

  @property
  def foot_landing_clearance(self):
    return self._desired_height - self._desired_landing_height[2]

  @foot_landing_clearance.setter
  def foot_landing_clearance(self, landing_clearance: float) -> None:
    self._desired_landing_height = np.array(
        (0., 0., self._desired_height - landing_clearance))

  def get_action(self) -> Mapping[Any, Any]:
    com_velocity = self._state_estimator.com_velocity_body_frame
    com_velocity = np.array((com_velocity[0], com_velocity[1], 0))

    _, _, yaw_dot = self._robot.base_angular_velocity_body_frame
    hip_positions = self._robot.swing_reference_positions

    all_joint_angles = {}
    for leg_id, leg_state in enumerate(self._gait_generator.leg_state):
      if leg_state in (gait_generator_lib.LegState.STANCE,
                       gait_generator_lib.LegState.EARLY_CONTACT,
                       gait_generator_lib.LegState.LOSE_CONTACT):
        continue

      # For now we did not consider the body pitch/roll and all calculation is
      # in the body frame. TODO(b/143378213): Calculate the foot_target_position
      # in world frame and then project back to calculate the joint angles.
      hip_offset = hip_positions[leg_id]
      twisting_vector = np.array((-hip_offset[1], hip_offset[0], 0))
      hip_horizontal_velocity = com_velocity + yaw_dot * twisting_vector
      target_hip_horizontal_velocity = (
          self.desired_speed + self.desired_twisting_speed * twisting_vector)
      if self.use_raibert_heuristic or (
          not self.foot_placement_position.any()):
        # Use raibert heuristic to determine target foot position
        foot_target_position = (
            hip_horizontal_velocity *
            self._gait_generator.stance_duration[leg_id] / 2 - _KP *
            (target_hip_horizontal_velocity - hip_horizontal_velocity))
        foot_target_position = np.clip(foot_target_position,
                                       [-0.15, -0.1, -0.05], [0.15, 0.1, 0.05])
        foot_target_position = foot_target_position - \
          self._desired_landing_height + \
          np.array((hip_offset[0], hip_offset[1], 0))
      else:
        foot_target_position = self.foot_placement_position[
            leg_id] - self._desired_landing_height + np.array(
                (hip_offset[0], hip_offset[1], 0))

      # Compute target position compensation due to slope
      gravity_projection_vector = \
        self._state_estimator.gravity_projection_vector
      multiplier = -self._desired_landing_height[
          2] / gravity_projection_vector[2]
      foot_target_position[:2] += gravity_projection_vector[:2] * multiplier
      # logging.info("Compsenation: {}".format(gravity_projection_vector[:2] *
      #                                        multiplier))

      foot_position = _gen_swing_foot_trajectory(
          self._gait_generator.normalized_phase[leg_id],
          self._phase_switch_foot_local_position[leg_id], foot_target_position,
          self._foot_height)

      joint_ids, joint_angles = (
          self._robot.get_motor_angles_from_foot_position(
              leg_id, foot_position))
      # Update the stored joint angles as needed.
      for joint_id, joint_angle in zip(joint_ids, joint_angles):
        all_joint_angles[joint_id] = (joint_angle, leg_id)
    action = {}
    kps = self._robot.motor_group.kps
    kds = self._robot.motor_group.kds
    for joint_id, joint_angle_leg_id in all_joint_angles.items():
      leg_id = joint_angle_leg_id[1]
      action[joint_id] = MotorCommand(desired_position=joint_angle_leg_id[0],
                                      kp=kps[joint_id],
                                      desired_velocity=0,
                                      kd=kds[joint_id],
                                      desired_extra_torque=0)

    return action
