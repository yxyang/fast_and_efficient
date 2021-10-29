"""Example of whole body controller on A1 robot."""
from absl import app
from absl import flags

import time

from src.robots import gamepad_reader
from src.convex_mpc_controller import locomotion_controller


flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 1., "maximum time to run the robot.")
FLAGS = flags.FLAGS


def _update_controller(controller, gamepad):
  # Update speed
  lin_speed, rot_speed = gamepad.speed_command
  controller.set_desired_speed(lin_speed, rot_speed)
  if (gamepad.estop_flagged) and (controller.mode !=
                                  locomotion_controller.ControllerMode.DOWN):
    controller.set_controller_mode(locomotion_controller.ControllerMode.DOWN)

  # Update controller moce
  controller.set_controller_mode(gamepad.mode_command)

  # Update gait
  controller.set_gait(gamepad.gait_command)


def main(argv):
  del argv  # unused
  gamepad = gamepad_reader.Gamepad(vel_scale_x=1,
                                   vel_scale_y=1,
                                   vel_scale_rot=1,
                                   max_acc=0.3)
  controller = locomotion_controller.LocomotionController(
      FLAGS.use_real_robot, FLAGS.show_gui)

  try:
    start_time = controller.time_since_reset
    current_time = start_time
    while current_time - start_time < FLAGS.max_time_secs:
      current_time = controller.time_since_reset
      time.sleep(0.05)
      _update_controller(controller, gamepad)
      if not controller.is_safe:
        gamepad.flag_estop()

  finally:
    gamepad.stop()
    controller.set_controller_mode(
        locomotion_controller.ControllerMode.TERMINATE)


if __name__ == "__main__":
  app.run(main)
