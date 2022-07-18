"""Base class for all robots."""
import ml_collections
import numpy as np
from typing import Any
from typing import Optional
from typing import Tuple

from src.robots import kinematics
from src.robots.motors import MotorCommand
from src.robots.motors import MotorControlMode
from src.robots.motors import MotorGroup


class Robot:
  """Robot Base
    A `Robot` requires access to a pre-instantiated `pybullet_client` and
    information about how the simulator was configured.

    A `Robot` is composed of joints which correspond to motors. For the most
    flexibility, we choose to pass motor objects to the robot when it is
    constructed. This allows for easy config-driven instantiation of
    different robot morphologies.

    Motors are passed as a collection of another collection of motors. A
    collection of motors at the lowest level is implemented in a `MotorGroup`.
    This means a robot can support the following configurations:
    1 Motor Robot: [ [ Arm Motor ] ]
    1 MotorGroup Robot: [ [ Arm Motor1, Arm Motor2 ] ]
    2 MotorGroup Robot: [ [ Leg Motor1, Leg Motor2 ],
      [ Arm Motor1, Arm Motor2 ] ]
    """
  def __init__(
      self,
      motors: MotorGroup,
      urdf_path: str,
      pybullet_client: Any,
      sim_conf: ml_collections.ConfigDict,
      base_joint_names: Tuple[str, ...],
      foot_joint_names: Tuple[str, ...],
  ) -> None:
    """Constructs a base robot and resets it to the initial states.
        TODO
        """
    self._sim_conf = sim_conf
    self._pybullet_client = pybullet_client
    self._motor_group = motors
    self._base_joint_names = base_joint_names
    self._foot_joint_names = foot_joint_names
    self._num_motors = self._motor_group.num_motors if self._motor_group else 0
    self._motor_torques = None
    self._urdf_path = urdf_path
    self._load_robot_urdf(self._urdf_path)
    self._step_counter = 0
    self._foot_contact_history = self.foot_positions_in_base_frame.copy()
    self._foot_contact_history[:, 2] = -self.mpc_body_height
    self._last_timestamp = 0
    self.reset()

  def _load_robot_urdf(self, urdf_path: str) -> None:
    # TODO: how do we want to check for missing attributes when not using
    # configs? One way to do it:
    if not self._pybullet_client:
      raise AttributeError("No pybullet client specified!")
    p = self._pybullet_client
    if self._sim_conf.on_rack:
      self.quadruped = p.loadURDF(urdf_path,
                                  self._sim_conf.init_rack_position,
                                  useFixedBase=True)
    else:
      self.quadruped = p.loadURDF(urdf_path, self._sim_conf.init_position)

    self._build_urdf_ids()

  def _build_urdf_ids(self) -> None:
    """Records ids of base link, foot links and motor joints.

        For detailed documentation of links and joints, please refer to the
        pybullet documentation.
        """
    self._chassis_link_ids = [-1]
    self._motor_joint_ids = []
    self._foot_link_ids = []

    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    for joint_id in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, joint_id)
      joint_name = joint_info[1].decode("UTF-8")
      if joint_name in self._base_joint_names:
        self._chassis_link_ids.append(joint_id)
      elif joint_name in self._motor_group.motor_joint_names:
        self._motor_joint_ids.append(joint_id)
      elif joint_name in self._foot_joint_names:
        self._foot_link_ids.append(joint_id)

  def reset(self,
            hard_reset: bool = False,
            num_reset_steps: Optional[int] = None) -> None:
    """Resets the robot."""
    if hard_reset:
      # This assumes that resetSimulation() is already called.
      self._load_robot_urdf(self._urdf_path)
    else:
      init_position = (self._sim_conf.init_rack_position
                       if self._sim_conf.on_rack else
                       self._sim_conf.init_position)
      self._pybullet_client.resetBasePositionAndOrientation(
          self.quadruped, init_position, [0.0, 0.0, 0.0, 1.0])

    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    for joint_id in range(num_joints):
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0,
      )

    # Set motors to the initial position
    # TODO: these should be set already when they are instantiated?
    for i in range(len(self._motor_joint_ids)):
      self._pybullet_client.resetJointState(
          self.quadruped,
          self._motor_joint_ids[i],
          self._motor_group.init_positions[i],
          targetVelocity=0,
      )

    # Steps the robot with position command
    if num_reset_steps is None:
      num_reset_steps = int(self._sim_conf.reset_time_s /
                            self._sim_conf.timestep)
    motor_command = MotorCommand(
        desired_position=self._motor_group.init_positions)
    for _ in range(num_reset_steps):
      self.step(motor_command, MotorControlMode.POSITION)
    self._last_timestamp = self.time_since_reset
    self._step_counter = 0

  def _apply_action(self, action, motor_control_mode=None) -> None:
    torques, observed_torques = self._motor_group.convert_to_torque(
        action, self.motor_angles, self.motor_velocities, motor_control_mode)
    # import pdb
    # pdb.set_trace()
    self._pybullet_client.setJointMotorControlArray(
        bodyIndex=self.quadruped,
        jointIndices=self._motor_joint_ids,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        forces=torques,
    )
    self._motor_torques = observed_torques

  def _update_contact_history(self):
    dt = self.time_since_reset - self._last_timestamp
    self._last_timestamp = self.time_since_reset
    base_orientation = self.base_orientation_quat
    rot_mat = self.pybullet_client.getMatrixFromQuaternion(base_orientation)
    rot_mat = np.array(rot_mat).reshape((3, 3))
    base_vel_body_frame = rot_mat.T.dot(self.base_velocity)

    foot_contacts = self.foot_contacts.copy()
    foot_positions = self.foot_positions_in_base_frame.copy()
    for leg_id in range(4):
      if foot_contacts[leg_id]:
        self._foot_contact_history[leg_id] = foot_positions[leg_id]
      else:
        self._foot_contact_history[leg_id] -= base_vel_body_frame * dt

  def step(self, action, motor_control_mode=None) -> None:
    self._step_counter += 1
    for _ in range(self._sim_conf.action_repeat):
      self._apply_action(action, motor_control_mode)
      self._pybullet_client.stepSimulation()
      self._update_contact_history()

  @property
  def foot_contacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

    contacts = [False, False, False, False]
    for contact in all_contacts:
      # Ignore self contacts
      if contact[2] == self.quadruped:
        continue
      try:
        toe_link_index = self._foot_link_ids.index(contact[3])
        contacts[toe_link_index] = True
      except ValueError:
        continue
    return contacts

  @property
  def base_position(self):
    return np.array(
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped)[0])

  @property
  def base_velocity(self):
    return self._pybullet_client.getBaseVelocity(self.quadruped)[0]

  @property
  def base_orientation_rpy(self):
    return self._pybullet_client.getEulerFromQuaternion(
        self.base_orientation_quat)

  @property
  def base_orientation_quat(self):
    return np.array(
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped)[1])

  @property
  def motor_angles(self):
    joint_states = self._pybullet_client.getJointStates(
        self.quadruped, self._motor_joint_ids)
    return np.array([s[0] for s in joint_states])

  @property
  def motor_velocities(self):
    joint_states = self._pybullet_client.getJointStates(
        self.quadruped, self._motor_joint_ids)
    return np.array([s[1] for s in joint_states])

  @property
  def motor_torques(self):
    return self._motor_torques

  @property
  def foot_contact_history(self):
    return self._foot_contact_history

  @property
  def base_angular_velocity_body_frame(self):
    angular_velocity = self._pybullet_client.getBaseVelocity(self.quadruped)[1]
    orientation = self.base_orientation_quat
    _, orientation_inversed = self._pybullet_client.invertTransform(
        [0, 0, 0], orientation)
    relative_velocity, _ = self._pybullet_client.multiplyTransforms(
        [0, 0, 0],
        orientation_inversed,
        angular_velocity,
        self._pybullet_client.getQuaternionFromEuler([0, 0, 0]),
    )
    return np.asarray(relative_velocity)

  @property
  def foot_positions_in_base_frame(self):
    foot_positions = []
    for foot_id in self._foot_link_ids:
      foot_positions.append(
          kinematics.link_position_in_base_frame(
              robot=self,
              link_id=foot_id,
          ))
    return np.array(foot_positions)

  def compute_foot_jacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    full_jacobian = kinematics.compute_jacobian(
        robot=self,
        link_id=self._foot_link_ids[leg_id],
    )
    motors_per_leg = self.num_motors // self.num_legs
    com_dof = 6
    return full_jacobian[:, com_dof + leg_id * motors_per_leg:com_dof +
                         (leg_id + 1) * motors_per_leg]

  def get_motor_angles_from_foot_position(self, leg_id, foot_local_position):
    toe_id = self._foot_link_ids[leg_id]

    motors_per_leg = self.num_motors // self.num_legs
    joint_position_idxs = list(
        range(leg_id * motors_per_leg,
              leg_id * motors_per_leg + motors_per_leg))

    joint_angles = kinematics.joint_angles_from_link_position(
        robot=self,
        link_position=foot_local_position,
        link_id=toe_id,
        joint_ids=joint_position_idxs,
    )
    # Return the joing index (the same as when calling GetMotorAngles) as well
    # as the angles.
    return joint_position_idxs, joint_angles

  def map_contact_force_to_joint_torques(self, leg_id, contact_force):
    """Maps the foot contact force to the leg joint torques."""
    jv = self.compute_foot_jacobian(leg_id)
    motor_torques_list = np.matmul(contact_force, jv)
    motor_torques_dict = {}
    motors_per_leg = self.num_motors // self.num_legs
    for torque_id, joint_id in enumerate(
        range(leg_id * motors_per_leg, (leg_id + 1) * motors_per_leg)):
      motor_torques_dict[joint_id] = motor_torques_list[torque_id]
    return motor_torques_dict

  @property
  def control_timestep(self):
    return self._sim_conf.timestep * self._sim_conf.action_repeat

  @property
  def time_since_reset(self):
    return self._step_counter * self.control_timestep

  @property
  def motor_group(self):
    return self._motor_group

  @property
  def num_legs(self):
    return 4

  @property
  def num_motors(self):
    raise NotImplementedError()

  @property
  def swing_reference_positions(self):
    raise NotImplementedError()

  @property
  def pybullet_client(self):
    return self._pybullet_client

  @property
  def mpc_body_height(self):
    raise NotImplementedError()

  @property
  def mpc_body_mass(self):
    raise NotImplementedError()

  @property
  def mpc_body_inertia(self):
    raise NotImplementedError()
