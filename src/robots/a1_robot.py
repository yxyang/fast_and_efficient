"""Real A1 robot class."""
import ml_collections
import numpy as np
import robot_interface
import time
from typing import Any
from typing import Tuple

from src.robots import a1
from src.robots import a1_robot_state_estimator
from src.robots.motors import MotorControlMode
from src.robots.motors import MotorCommand

# Constants for analytical FK/IK
COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET


class A1Robot(a1.A1):
  """Class for interfacing with A1 hardware."""
  def __init__(
      self,
      pybullet_client: Any = None,
      sim_conf: ml_collections.ConfigDict = None,
      urdf_path: str = "a1.urdf",
      base_joint_names: Tuple[str, ...] = (),
      foot_joint_names: Tuple[str, ...] = (
          "FR_toe_fixed",
          "FL_toe_fixed",
          "RR_toe_fixed",
          "RL_toe_fixed",
      ),
      motor_control_mode: MotorControlMode = MotorControlMode.POSITION,
      mpc_body_height: float = 0.26,
      mpc_body_mass: float = 110 / 9.8,
      mpc_body_inertia: Tuple[float] = np.array(
          (0.027, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 5.,
  ) -> None:
    self._raw_state = robot_interface.LowState()
    self._contact_force_threshold = np.zeros(4)
    # Send an initial zero command in order to receive state information.
    self._robot_interface = robot_interface.RobotInterface(0xff)
    self._state_estimator = a1_robot_state_estimator.A1RobotStateEstimator(
        self)
    self._last_reset_time = time.time()

    super(A1Robot, self).__init__(pybullet_client, sim_conf, urdf_path,
                                  base_joint_names, foot_joint_names,
                                  motor_control_mode, mpc_body_height,
                                  mpc_body_mass, mpc_body_inertia)

  def _receive_observation(self) -> None:
    """Receives observation from robot and saves the state.

    Note that the returned state from robot's receive_observation() function
    is mutable. So we need to copy the value out.
    """
    self._raw_state = self._robot_interface.receive_observation()

  def step(self,
           action: MotorCommand,
           motor_control_mode: MotorControlMode = None) -> None:
    self._step_counter += 1
    for _ in range(self._sim_conf.action_repeat):
      self._apply_action(action, motor_control_mode)
      self._receive_observation()
      self._state_estimator.update(self._raw_state)
      self._update_contact_history()

  def _apply_action(self,
                    action: MotorCommand,
                    motor_control_mode: MotorControlMode = None) -> None:
    """Clips and then apply the motor commands using the motor model.
    Args:
      action: np.array. Can be motor angles, torques, or hybrid commands.
      motor_control_mode: A MotorControlMode enum.
    """
    if motor_control_mode is None:
      motor_control_mode = self._motor_group.motor_control_mode
    command = np.zeros(60, dtype=np.float32)
    if motor_control_mode == MotorControlMode.POSITION:
      for motor_id in range(self.num_motors):
        command[motor_id * 5] = action.desired_position[motor_id]
        command[motor_id * 5 + 1] = action.kp[motor_id]
        command[motor_id * 5 + 3] = action.kd[motor_id]
    elif motor_control_mode == MotorControlMode.TORQUE:
      for motor_id in range(self.num_motors):
        command[motor_id * 5 + 4] = action.desired_torque[motor_id]
    elif motor_control_mode == MotorControlMode.HYBRID:
      command[0::5] = action.desired_position
      command[1::5] = action.kp
      command[2::5] = action.desired_velocity
      command[3::5] = action.kd
      command[4::5] = action.desired_extra_torque
    else:
      raise ValueError('Unknown motor control mode for A1 robot: {}.'.format(
          motor_control_mode))

    self._robot_interface.send_command(command)

  def reset(self, hard_reset: bool = False, reset_time=1.5):
    """Reset the robot to default motor angles."""
    super(A1Robot, self).reset(hard_reset, num_reset_steps=0)
    for _ in range(10):
      self._robot_interface.send_command(np.zeros(60, dtype=np.float32))
      time.sleep(0.001)
      self._receive_observation()

    print("About to reset the robot.")
    initial_motor_position = self.motor_angles
    end_motor_position = self.motor_group.init_positions
    # Stand up in 1.5 seconds, and fix the standing pose afterwards.
    standup_time = min(reset_time, 1.)
    stand_foot_forces = []
    for t in np.arange(0, reset_time,
                       self._sim_conf.timestep * self._sim_conf.action_repeat):
      blend_ratio = min(t / standup_time, 1)
      desired_motor_position = blend_ratio * end_motor_position + (
          1 - blend_ratio) * initial_motor_position
      action = MotorCommand(desired_position=desired_motor_position,
                            kp=self.motor_group.kps,
                            desired_velocity=np.zeros(self.num_motors),
                            kd=self.motor_group.kds)
      self.step(action, MotorControlMode.POSITION)
      time.sleep(self._sim_conf.timestep * self._sim_conf.action_repeat)
      if t > standup_time:
        stand_foot_forces.append(self.foot_forces)

    # Calibrate foot force sensors
    stand_foot_forces = np.mean(stand_foot_forces, axis=0)
    for leg_id in range(4):
      self.update_foot_contact_force_threshold(leg_id,
                                               stand_foot_forces[leg_id] * 0.8)

    self._last_reset_time = time.time()
    self._state_estimator.reset()

  @property
  def sim_conf(self):
    return self._sim_conf

  @property
  def foot_forces(self):
    return np.array(self._raw_state.footForce)

  def update_foot_contact_force_threshold(self, leg_id, threshold):
    self._contact_force_threshold[leg_id] = threshold

  @property
  def foot_contacts(self):
    return np.array(self._raw_state.footForce) > self._contact_force_threshold

  @property
  def base_position(self):
    contacts = np.array(self.foot_contacts)
    if not np.sum(contacts):
      return np.array([0., 0., self.mpc_body_height])
    foot_positions_base_frame = self.foot_positions_in_base_frame
    foot_heights = -foot_positions_base_frame[:, 2]
    base_height = np.sum(foot_heights * contacts) / np.sum(contacts)
    return np.array([0., 0., base_height])

  @property
  def base_velocity(self):
    return self._state_estimator.estimated_velocity.copy()

  @property
  def base_orientation_rpy(self):
    return np.array(self._raw_state.imu.rpy)

  @property
  def base_orientation_quat(self):
    q = self._raw_state.imu.quaternion
    return np.array([q[1], q[2], q[3], q[0]])

  @property
  def motor_angles(self):
    return np.array([motor.q for motor in self._raw_state.motorState[:12]])

  @property
  def motor_velocities(self):
    return np.array([motor.dq for motor in self._raw_state.motorState[:12]])

  @property
  def motor_torques(self):
    return np.array(
        [motor.tauEst for motor in self._raw_state.motorState[:12]])

  @property
  def base_angular_velocity_body_frame(self):
    return np.array(self._raw_state.imu.gyroscope)

  @property
  def foot_positions_in_base_frame(self):
    """Use analytical FK/IK/Jacobian"""
    return self._foot_positions_in_hip_frame + HIP_OFFSETS

  @property
  def time_since_reset(self):
    return time.time() - self._last_reset_time

  @property
  def _foot_positions_in_hip_frame(self):
    motor_angles = self.motor_angles.reshape((4, 3))
    foot_positions = np.zeros((4, 3))
    for i in range(4):
      foot_positions[i] = self._foot_position_in_hip_frame(
          motor_angles[i], l_hip_sign=(-1)**(i + 1))
    return foot_positions

  def _foot_position_in_hip_frame(self, angles, l_hip_sign=1):
    """Computes foot forward kinematics analytically."""
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = np.sqrt(l_up**2 + l_low**2 +
                           2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])

  def compute_foot_jacobian(self, leg_id):
    """Computes foot jacobian matrix analytically."""
    motor_angles = self.motor_angles[leg_id * 3:(leg_id + 1) * 3]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1)**(leg_id + 1)

    t1, t2, t3 = motor_angles[0], motor_angles[1], motor_angles[2]
    l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * np.cos(t_eff)
    J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(
        t_eff) / l_eff - l_eff * np.cos(t_eff) / 2
    J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J

  def get_motor_angles_from_foot_position(self, leg_id, foot_local_position):
    motors_per_leg = self.num_motors // self.num_legs
    joint_position_idxs = list(
        range(leg_id * motors_per_leg,
              leg_id * motors_per_leg + motors_per_leg))

    joint_angles = self._foot_position_in_hip_frame_to_joint_angle(
        foot_local_position - HIP_OFFSETS[leg_id],
        l_hip_sign=(-1)**(leg_id + 1))
    return joint_position_idxs, joint_angles.tolist()

  def _foot_position_in_hip_frame_to_joint_angle(self,
                                                 foot_position,
                                                 l_hip_sign=1):
    """Computes foot inverse kinematics analytically."""
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(
        np.clip((x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
                (2 * l_low * l_up), -1, 1))
    l = np.sqrt(
        np.maximum(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee),
                   1e-7))
    theta_hip = np.arcsin(np.clip(-x / l, -1, 1)) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])
