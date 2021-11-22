'''
Policy class for computing action from weights and observation vector.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht
'''
from absl import logging
import numpy as np
from src.agents.cmaes.filter import get_filter
from src.intermediate_envs import gait_change_env


def relu(x):
  return x * (x > 0)


class Policy(object):
  def __init__(self, config, policy_params, observation_space, action_space):

    self.ob_dim = policy_params['ob_dim']
    self.ac_dim = policy_params['ac_dim']
    self.weights = np.empty(0)
    self.action_space = action_space
    self.observation_space = observation_space

    # a filter for updating statistics of the observations and normalizing inputs to the policies
    self.observation_filter = get_filter(policy_params['ob_filter'],
                                         shape=(self.ob_dim, ))
    self.update_filter = True

  def update_weights(self, new_weights):
    self.weights[:] = new_weights[:]
    return

  def get_weights(self):
    return self.weights

  def get_observation_filter(self):
    return self.observation_filter

  def act(self, ob):
    raise NotImplementedError

  def copy(self):
    raise NotImplementedError


class LinearPolicy(Policy):
  """Linear policy class that computes action as <w, ob>."""
  def __init__(self, config, policy_params, observation_space, action_space):
    self.config = config
    Policy.__init__(self, config, policy_params, action_space)
    self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float64)

  def act(self, ob, verbose=False):
    ob = self.observation_filter(ob, update=self.update_filter)
    action = np.dot(self.weights, ob)
    if self.config.action_limit_method == 'tanh':
      action = np.tanh(action)
    elif self.config.action_limit_method == 'clip':
      action = np.clip(action, -1, 1)
    else:
      raise ValueError("Unknown action limiting method.")
    action_mid = (self.action_space.low + self.action_space.high) / 2
    action_range = (self.action_space.high - self.action_space.low) / 2
    action = action * action_range + action_mid
    return action

  def get_weights_plus_stats(self):
    mu, std = self.observation_filter.get_stats()
    aux = np.asarray([self.weights, mu, std])
    return aux


class NNPolicy(Policy):
  def __init__(self, config, policy_params, observation_space, action_space):
    Policy.__init__(self, config, policy_params, observation_space,
                    action_space)
    self.config = config
    self.original_weights = []

    curr = self.ob_dim
    for _ in range(self.config.get('num_hidden_layers', 0)):
      self.original_weights.append(
          np.random.normal(size=(self.config.hidden_layer_size, curr)) * 0.01)
      self.original_weights.append(np.zeros(self.config.hidden_layer_size))
      curr = self.config.hidden_layer_size
    self.original_weights.append(np.zeros((self.ac_dim, curr)))
    self.original_weights.append(np.zeros(self.ac_dim))

    self.weights = np.concatenate([w.flatten() for w in self.original_weights])

  def _split_weights(self, weights):
    weights = weights.copy()
    splitted_weights = []
    for w in self.original_weights:
      length = np.prod(w.shape)
      splitted_weights.append(weights[:length].reshape(w.shape))
      weights = weights[length:]
    return splitted_weights

  def act(self, ob, verbose=False):
    ob = self.observation_filter(ob, update=self.update_filter)
    splitted_weights = self._split_weights(self.weights)

    curr = ob.copy()
    for _ in range(self.config.get('num_hidden_layers', 0)):
      # Hidden layers of NN
      weight, bias = splitted_weights[0], splitted_weights[1]
      splitted_weights = splitted_weights[2:]
      curr = relu(weight.dot(curr) + bias)
    # Output layers of NN
    weight, bias = splitted_weights[0], splitted_weights[1]
    action = weight.dot(curr) + bias

    if self.action_space:
      if self.config.action_limit_method == 'tanh':
        action = np.tanh(action)
      elif self.config.action_limit_method == 'clip':
        action = np.clip(action, -1, 1)
      else:
        raise ValueError("Unknown action limiting method.")
      action_mid = (self.action_space.low + self.action_space.high) / 2
      action_range = (self.action_space.high - self.action_space.low) / 2
      action = action * action_range + action_mid
    return action

  def get_weights_plus_stats(self):
    mu, std = self.observation_filter.get_stats()
    aux = np.asarray([self.weights, mu, std])
    return aux
