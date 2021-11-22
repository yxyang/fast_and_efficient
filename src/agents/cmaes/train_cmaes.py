"""Train gait policies using CMAES"""
from absl import app
from absl import logging
from absl import flags

import cma
from datetime import datetime
from ml_collections.config_flags import config_flags
import numpy as np
import os
import ray
import tensorflow as tf
import time

from src.agents.cmaes import logz
from src.agents.cmaes import rollout_server

config_flags.DEFINE_config_file(
    'config', 'locomotion/agents/cmaes/configs/gait_change_deluxe.py',
    'experiment configuration.')
flags.DEFINE_string('experiment_name', 'deluxe_cmaes', 'expriment_name')
flags.DEFINE_integer('random_seed', 1000, 'random seed')
FLAGS = flags.FLAGS


def _get_policy_dimension(config):
  env = config.env_constructor(**config.env_args)
  policy_params = {
      'ob_filter': config.filter,
      'ob_dim': env.observation_space.low.shape[0],
      'ac_dim': env.action_space.low.shape[0]
  }
  policy = config.policy_constructor(config, policy_params,
                                     env.observation_space, env.action_space)
  return policy.weights.shape[0]


def main(_):
  config = FLAGS.config
  logdir = os.path.join(config.logdir, FLAGS.experiment_name,
                        datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
  if not (os.path.exists(logdir)):
    os.makedirs(logdir)
  config.logdir = logdir

  logz.configure_output_dir(config.logdir)
  logz.save_params(config)
  writer = tf.summary.create_file_writer(config.logdir)
  writer.set_as_default()

  dim_weights = _get_policy_dimension(config)
  x0, sigma0 = np.zeros(dim_weights), config.get('initial_sigma', .03)
  popsize = 32
  es = cma.CMAEvolutionStrategy(x0, sigma0,
                                dict(seed=FLAGS.random_seed, popsize=popsize))

  ray.init(address='auto', _redis_password='1234')
  workers = [
      rollout_server.RolloutServer.remote(server_id, config)
      for server_id in range(es.popsize)
  ]
  start_time = time.time()

  for step in range(config.num_iters):
    solutions = es.ask()
    rollout_ids = [
        worker.eval_policy.remote(step, solution)
        for worker, solution in zip(workers, solutions)
    ]
    rewards = np.array(ray.get(rollout_ids))

    es.tell(solutions, -rewards)
    logz.log_tabular("Time", time.time() - start_time)
    logz.log_tabular("Iteration", step)
    logz.log_tabular("AverageReward", np.mean(rewards))
    logz.log_tabular("StdRewards", np.std(rewards))
    logz.log_tabular("MaxRewardRollout", np.max(rewards))
    logz.log_tabular("MinRewardRollout", np.min(rewards))
    logz.dump_tabular()
    with tf.name_scope('train'):
      tf.summary.scalar('Average Rewards', data=np.mean(rewards), step=step)
      tf.summary.scalar('Minimum Rewards', data=np.min(rewards), step=step)
      tf.summary.scalar('Maximum Rewards', data=np.max(rewards), step=step)

    if step % config.eval_every == 0:
      rollout_ids = [
          worker.eval_policy.remote(step, es.mean, eval=True)
          for worker in workers
      ]
      rewards = np.array(ray.get(rollout_ids))
      logz.log_tabular("Time", time.time() - start_time)
      logz.log_tabular("Iteration", step)
      logz.log_tabular("AverageReward", np.mean(rewards))
      logz.log_tabular("StdRewards", np.std(rewards))
      logz.log_tabular("MaxRewardRollout", np.max(rewards))
      logz.log_tabular("MinRewardRollout", np.min(rewards))
      logz.dump_tabular()
      with tf.name_scope('eval'):
        tf.summary.scalar('Average Rewards', data=np.mean(rewards), step=step)
        tf.summary.scalar('Minimum Rewards', data=np.min(rewards), step=step)
        tf.summary.scalar('Maximum Rewards', data=np.max(rewards), step=step)
      np.savez(
          open(
              os.path.join(config.logdir,
                           'lin_policy_plus_{}.npz'.format(step)), 'wb'),
          ray.get(workers[0].get_weights_plus_stats.remote()))


if __name__ == "__main__":
  app.run(main)
