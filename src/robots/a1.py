"""Base class for all robots."""
import ml_collections
import numpy as np
from typing import Any
from typing import Sequence
from typing import Tuple

from src.robots.motors import MotorControlMode
from src.robots.motors import MotorGroup
from src.robots.motors import MotorModel
from src.robots.robot import Robot


class A1(Robot):
  """A1 Robot."""
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
      mpc_body_height: float = 0.3,
      mpc_body_mass: float = 110 / 9.8,
      mpc_body_inertia: Tuple[float] = np.array(
          (0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 10.,
  ) -> None:
    """Constructs an A1 robot and resets it to the initial states.
        Initializes a tuple with a single MotorGroup containing 12 MotoroModels.
        Each MotorModel is by default configured for the parameters of the A1.
        """
    motors = MotorGroup((
        MotorModel(
            name="FR_hip_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.0,
            min_position=-0.802851455917,
            max_position=0.802851455917,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="FR_upper_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="FR_lower_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="FL_hip_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.0,
            min_position=-0.802851455917,
            max_position=0.802851455917,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="FL_upper_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="FL_lower_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RR_hip_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.0,
            min_position=-0.802851455917,
            max_position=0.802851455917,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="RR_upper_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RR_lower_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RL_hip_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.0,
            min_position=-0.802851455917,
            max_position=0.802851455917,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="RL_upper_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RL_lower_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-33.5,
            max_torque=33.5,
            kp=100,
            kd=2,
        ),
    ))
    self._mpc_body_height = mpc_body_height
    self._mpc_body_mass = mpc_body_mass
    self._mpc_body_inertia = mpc_body_inertia

    super().__init__(
        pybullet_client=pybullet_client,
        sim_conf=sim_conf,
        urdf_path=urdf_path,
        motors=motors,
        base_joint_names=base_joint_names,
        foot_joint_names=foot_joint_names,
    )

  @property
  def mpc_body_height(self):
    return self._mpc_body_height

  @mpc_body_height.setter
  def mpc_body_height(self, mpc_body_height: float):
    self._mpc_body_height = mpc_body_height

  @property
  def mpc_body_mass(self):
    return self._mpc_body_mass

  @mpc_body_mass.setter
  def mpc_body_mass(self, mpc_body_mass: float):
    self._mpc_body_mass = mpc_body_mass

  @property
  def mpc_body_inertia(self):
    return self._mpc_body_inertia

  @mpc_body_inertia.setter
  def mpc_body_inertia(self, mpc_body_inertia: Sequence[float]):
    self._mpc_body_inertia = mpc_body_inertia

  @property
  def swing_reference_positions(self):
    return (
        (0.17, -0.135, 0),
        (0.17, 0.13, 0),
        (-0.195, -0.135, 0),
        (-0.195, 0.13, 0),
    )

  @property
  def num_motors(self):
    return 12
