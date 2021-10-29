"""The main learning class for ARS."""
import numpy as np
import ray
import tensorflow as tf
import time

from locomotion.agents.ars import logz
from locomotion.agents.ars import optimizers
from locomotion.agents.ars.policies import LinearPolicy, GaitChangePolicy
from locomotion.agents.ars.shared_noise import create_shared_noise, SharedNoiseTable
from locomotion.agents.ars.rollout_worker import Worker
from locomotion.agents.ars import utils


class ARSLearner(object):
  """
    Object class implementing the ARS algorithm.
    """
  def __init__(self, config):
    self.config = config
    logz.configure_output_dir(config.logdir)
    logz.save_params(config)
    self.writer = tf.summary.create_file_writer(config.logdir)
    self.writer.set_as_default()

    env = config.env_constructor(**config.env_args)

    self.timesteps = 0
    self.action_size = env.action_space.shape[0]
    self.ob_size = env.observation_space.shape[0]
    self.num_deltas = config.num_deltas
    self.deltas_used = config.deltas_used
    self.rollout_length = config.rollout_length
    self.step_size = config.step_size
    self.delta_std = config.delta_std
    self.logdir = config.logdir
    self.shift = config.shift
    self.max_past_avg_reward = float('-inf')
    self.num_episodes_used = float('inf')

    policy_params = {
        'ob_filter': config.filter,
        'ob_dim': self.ob_size,
        'ac_dim': self.action_size
    }

    # create shared table for storing noise
    print("Creating deltas table.")
    deltas_id = create_shared_noise.remote()
    self.deltas = SharedNoiseTable(ray.get(deltas_id), seed=config.seed + 3)
    print('Created deltas table.')

    # initialize workers with different random seeds
    print('Initializing workers.')
    self.num_workers = config.num_workers
    self.workers = [
        Worker.remote(config.seed + 7 * i,
                      config=config,
                      policy_params=policy_params,
                      deltas=deltas_id,
                      rollout_length=config.rollout_length,
                      delta_std=config.delta_std)
        for i in range(config.num_workers)
    ]

    # initialize policy
    self.policy = config.policy_constructor(config, policy_params)

    init_policy_path = self.config.get('policy_init', None)
    if init_policy_path:
      lin_policy = dict(np.load(init_policy_path, allow_pickle=True))
      lin_policy = list(lin_policy.items())[0][1]
      M, mean, std = lin_policy[0], lin_policy[1], lin_policy[2]
      self.policy.update_weights(M)
      if config.filter != 'NoFilter':
        self.policy.observation_filter.mean = mean
        self.policy.observation_filter.std = std
      # Sync filter among all workers
      filter_id = ray.put(self.policy.observation_filter)
      setting_filters_ids = [
          worker.sync_filter.remote(filter_id) for worker in self.workers
      ]
      # waiting for sync of all workers
      ray.get(setting_filters_ids)

    self.w_policy = self.policy.get_weights()

    # initialize optimization algorithm
    self.optimizer = optimizers.SGD(self.w_policy, self.step_size)
    print("Initialization of ARS complete.")

  def aggregate_rollouts(self, num_rollouts=None, evaluate=False):
    """
        Aggregate update step from rollouts generated in parallel.
        """

    if num_rollouts is None:
      num_deltas = self.num_deltas
    else:
      num_deltas = num_rollouts

    # put policy weights in the object store
    policy_id = ray.put(self.w_policy)

    t1 = time.time()
    num_rollouts = int(num_deltas / self.num_workers)

    # parallel generation of rollouts
    rollout_ids_one = [
        worker.do_rollouts.remote(policy_id,
                                  num_rollouts=num_rollouts,
                                  shift=self.shift,
                                  evaluate=evaluate) for worker in self.workers
    ]

    rollout_ids_two = [
        worker.do_rollouts.remote(policy_id,
                                  num_rollouts=1,
                                  shift=self.shift,
                                  evaluate=evaluate)
        for worker in self.workers[:(num_deltas % self.num_workers)]
    ]

    # gather results
    results_one = ray.get(rollout_ids_one)
    results_two = ray.get(rollout_ids_two)

    rollout_rewards, deltas_idx = [], []

    for result in results_one:
      if not evaluate:
        self.timesteps += result["steps"]
      deltas_idx += result['deltas_idx']
      rollout_rewards += result['rollout_rewards']

    for result in results_two:
      if not evaluate:
        self.timesteps += result["steps"]
      deltas_idx += result['deltas_idx']
      rollout_rewards += result['rollout_rewards']

    deltas_idx = np.array(deltas_idx)
    rollout_rewards = np.array(rollout_rewards, dtype=np.float64)

    print('Maximum reward of collected rollouts:', rollout_rewards.max())
    t2 = time.time()

    print('Time to generate rollouts:', t2 - t1)

    if evaluate:
      return rollout_rewards

    # select top performing directions if deltas_used < num_deltas
    original_rewards = rollout_rewards.copy()
    max_rewards = np.max(rollout_rewards, axis=1)
    if self.deltas_used > self.num_deltas:
      self.deltas_used = self.num_deltas

    idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(
        max_rewards, 100 * (1 - (self.deltas_used / self.num_deltas)))]
    deltas_idx = deltas_idx[idx]
    rollout_rewards = rollout_rewards[idx, :]

    # normalize rewards by their standard deviation
    rollout_rewards /= np.std(rollout_rewards)

    t1 = time.time()
    # aggregate rollouts to form g_hat, the gradient used to compute SGD step
    g_hat, count = utils.batched_weighted_sum(
        rollout_rewards[:, 0] - rollout_rewards[:, 1],
        (self.deltas.get(idx, self.w_policy.size) for idx in deltas_idx),
        batch_size=500)
    g_hat /= deltas_idx.size
    t2 = time.time()
    print('time to aggregate rollouts', t2 - t1)
    return g_hat, original_rewards

  def train_step(self, step):
    """
        Perform one update step of the policy weights.
        """

    g_hat, rollout_rewards = self.aggregate_rollouts()
    with tf.name_scope('train'):
      tf.summary.scalar("Euclidean norm of update step",
                        data=np.linalg.norm(g_hat),
                        step=step)
      tf.summary.scalar("Maximum Rewards",
                        data=np.max(rollout_rewards),
                        step=step)
      tf.summary.scalar("Minimum Rewards",
                        data=np.min(rollout_rewards),
                        step=step)
      tf.summary.scalar("Average Rewards",
                        data=np.mean(rollout_rewards),
                        step=step)
    self.w_policy -= self.optimizer._compute_step(g_hat).reshape(
        self.w_policy.shape)

  def train(self, num_iter):

    start = time.time()
    for i in range(num_iter):

      t1 = time.time()
      self.train_step(i)
      t2 = time.time()
      print('total time of one step', t2 - t1)
      print('iter ', i, ' done')

      t1 = time.time()
      # get statistics from all workers
      for j in range(self.num_workers):
        self.policy.observation_filter.update(
            ray.get(self.workers[j].get_filter.remote()))
      self.policy.observation_filter.stats_increment()

      # make sure master filter buffer is clear
      self.policy.observation_filter.clear_buffer()
      # sync all workers
      filter_id = ray.put(self.policy.observation_filter)
      setting_filters_ids = [
          worker.sync_filter.remote(filter_id) for worker in self.workers
      ]
      # waiting for sync of all workers
      ray.get(setting_filters_ids)

      increment_filters_ids = [
          worker.stats_increment.remote() for worker in self.workers
      ]
      # waiting for increment of all workers
      ray.get(increment_filters_ids)
      t2 = time.time()
      print('Time to sync statistics:', t2 - t1)

      # record statistics every 10 iterations
      if ((i) % 10 == 0):
        rewards = self.aggregate_rollouts(num_rollouts=10, evaluate=True)
        w = ray.get(self.workers[0].get_weights_plus_stats.remote())
        np.savez(self.logdir + "/lin_policy_plus_{}".format(i), w)

        # print(sorted(self.params.items()))
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", i + 1)
        logz.log_tabular("AverageReward", np.mean(rewards))
        logz.log_tabular("StdRewards", np.std(rewards))
        logz.log_tabular("MaxRewardRollout", np.max(rewards))
        logz.log_tabular("MinRewardRollout", np.min(rewards))
        logz.log_tabular("timesteps", self.timesteps)
        logz.dump_tabular()
        with tf.name_scope('eval'):
          tf.summary.scalar("Maximum Rewards", data=np.max(rewards), step=i)
          tf.summary.scalar("Minimum Rewards", data=np.min(rewards), step=i)
          tf.summary.scalar("Average Rewards", data=np.mean(rewards), step=i)
