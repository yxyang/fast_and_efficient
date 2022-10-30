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


class Go1(Robot):
  """Go1 Robot."""
  def __init__(
      self,
      pybullet_client: Any = None,
      sim_conf: ml_collections.ConfigDict = None,
      urdf_path: str = "go1.urdf",
      base_joint_names: Tuple[str, ...] = (),
      foot_joint_names: Tuple[str, ...] = (
          "FR_foot_fixed",
          "FL_foot_fixed",
          "RR_foot_fixed",
          "RL_foot_fixed",
      ),
      motor_control_mode: MotorControlMode = MotorControlMode.POSITION,
      mpc_body_height: float = 0.26,
      mpc_body_mass: float = 110 / 9.8,
      mpc_body_inertia: Tuple[float] = np.array(
          (0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 10.,
  ) -> None:
    """Constructs an Go1 robot and resets it to the initial states.
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
            min_torque=-23.7,
            max_torque=23.7,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="FR_thigh_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-23.7,
            max_torque=23.7,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="FR_calf_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-35.55,
            max_torque=35.55,
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
            min_torque=-23.7,
            max_torque=23.7,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="FL_thigh_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-23.7,
            max_torque=23.7,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="FL_calf_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-35.55,
            max_torque=35.55,
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
            min_torque=-23.7,
            max_torque=23.7,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="RR_thigh_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-23.7,
            max_torque=23.7,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RR_calf_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-35.55,
            max_torque=35.55,
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
            min_torque=-23.7,
            max_torque=23.7,
            kp=100,
            kd=1,
        ),
        MotorModel(
            name="RL_thigh_joint",
            motor_control_mode=motor_control_mode,
            init_position=0.9,
            min_position=-1.0471975512,
            max_position=4.18879020479,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-23.7,
            max_torque=23.7,
            kp=100,
            kd=2,
        ),
        MotorModel(
            name="RL_calf_joint",
            motor_control_mode=motor_control_mode,
            init_position=-1.8,
            min_position=-2.6965336943,
            max_position=-0.916297857297,
            min_velocity=-16,
            max_velocity=16,
            min_torque=-35.55,
            max_torque=35.55,
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
        (0.1835, -0.131, 0),
        (0.1835, 0.122, 0),
        (-0.1926, -0.131, 0),
        (-0.1926, 0.122, 0),
    )

  @property
  def num_motors(self):
    return 12
