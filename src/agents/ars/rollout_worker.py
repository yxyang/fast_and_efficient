"""Distributed worker for evaluating parameters in ARS."""
import numpy as np
import ray

from locomotion.agents.ars.policies import LinearPolicy
from locomotion.agents.ars.shared_noise import SharedNoiseTable


@ray.remote
class Worker(object):
  """
    Object class for parallel rollout generation.
    """
  def __init__(self,
               env_seed,
               config=None,
               policy_params=None,
               deltas=None,
               rollout_length=1000,
               delta_std=0.02):

    # initialize OpenAI environment for each worker
    self.config = config
    self.env = config.env_constructor(**config.env_args)
    self.env.seed(env_seed)

    # each worker gets access to the shared noise table
    # with independent random streams for sampling
    # from the shared noise table.
    self.deltas = SharedNoiseTable(deltas, env_seed + 7)
    self.policy_params = policy_params
    self.policy = self.config.policy_constructor(self.config, policy_params,
                                                 self.env.action_space)

    self.delta_std = delta_std
    self.rollout_length = rollout_length

  def get_weights_plus_stats(self):
    """
        Get current policy weights and current statistics of past states.
        """
    return self.policy.get_weights_plus_stats()

  def rollout(self, shift=0., rollout_length=None):
    """
        Performs one rollout of maximum length rollout_length.
        At each time-step it substracts shift from the reward.
        """

    if rollout_length is None:
      rollout_length = self.rollout_length

    total_reward = 0.
    steps = 0

    ob = self.env.reset()
    for i in range(rollout_length):
      action = self.policy.act(ob)
      ob, reward, done, _ = self.env.step(action)
      steps += 1
      total_reward += (reward - shift)
      if done:
        break

    print('Total_reward: {}'.format(total_reward))

    return total_reward, steps

  def do_rollouts(self, w_policy, num_rollouts=1, shift=1, evaluate=False):
    """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

    rollout_rewards, deltas_idx = [], []
    steps = 0

    for i in range(num_rollouts):

      if evaluate:
        self.policy.update_weights(w_policy)
        deltas_idx.append(-1)

        # set to false so that evaluation rollouts are not used for updating state statistics
        self.policy.update_filter = False

        # for evaluation we do not shift the rewards (shift = 0) and we use the
        # default rollout length (1000 for the MuJoCo locomotion tasks)
        reward, r_steps = self.rollout(shift=0.,
                                       rollout_length=self.rollout_length)
        rollout_rewards.append(reward)

      else:
        idx, delta = self.deltas.get_delta(w_policy.size)

        delta = (self.delta_std * delta).reshape(w_policy.shape)
        deltas_idx.append(idx)

        # set to true so that state statistics are updated
        self.policy.update_filter = True

        # compute reward and number of timesteps used for positive perturbation rollout
        self.policy.update_weights(w_policy + delta)
        pos_reward, pos_steps = self.rollout(shift=shift)

        # compute reward and number of timesteps used for negative pertubation rollout
        self.policy.update_weights(w_policy - delta)
        neg_reward, neg_steps = self.rollout(shift=shift)
        steps += pos_steps + neg_steps

        rollout_rewards.append([pos_reward, neg_reward])

    # if evaluate:
    #   print("Rollout Rewards from worker: {}".format(rollout_rewards))
    #   print("Filter from worker: {}".format(self.get_filter().std))

    return {
        'deltas_idx': deltas_idx,
        'rollout_rewards': rollout_rewards,
        "steps": steps
    }

  def stats_increment(self):
    self.policy.observation_filter.stats_increment()
    return

  def get_weights(self):
    return self.policy.get_weights()

  def get_filter(self):
    return self.policy.observation_filter

  def sync_filter(self, other):
    self.policy.observation_filter.sync(other)
    return
