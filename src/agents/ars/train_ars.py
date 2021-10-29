"""Launches training of ARS."""
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import ray
import socket
from datetime import datetime

from src.agents.ars import ars_learner

config_flags.DEFINE_config_file(
    'config', 'locomotion/agents/ars/configs/gait_change.py',
    'experiment configuration.')
flags.DEFINE_string('experiment_name', 'test', 'name of experiment.')
FLAGS = flags.FLAGS


def main(_):
  config = FLAGS.config
  # Initiate Ray session
  local_ip = socket.gethostbyname(socket.gethostname())
  ray.init(address='auto', _redis_password='1234')

  logdir = os.path.join(config.logdir, FLAGS.experiment_name,
                        datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f'))
  if not (os.path.exists(logdir)):
    os.makedirs(logdir)
  config.logdir = logdir
  ARS = ars_learner.ARSLearner(config)

  ARS.train(config.num_iters)


if __name__ == '__main__':
  app.run(main)
