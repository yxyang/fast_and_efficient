# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrappers for OpenAI Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import atexit
import functools
import multiprocessing
import sys
import traceback

import gym
import gym.spaces
import numpy as np


class AttributeModifier(object):
  """Provides getter and setter functions to access wrapped environments."""

  def __getattr__(self, name):
    return getattr(self._env, name)

  def set_attribute(self, name, value):
    """Set an attribute in the wrapped environment.

    Args:
      name: Attribute to access.
      value: New attribute value.
    """
    set_attr = getattr(self._env, 'set_attribute', None)
    if callable(set_attr):
      self._env.set_attribute(name, value)
    else:
      setattr(self._env, name, value)


class RangeNormalize(AttributeModifier):
  """Normalize the specialized observation and action ranges to [-1, 1]."""

  def __init__(self, env):
    self._env = env
    self._should_normalize_observ = self._is_finite(self._env.observation_space)
    if not self._should_normalize_observ:
      logging.info('Not normalizing infinite observation range.')
    self._should_normalize_action = self._is_finite(self._env.action_space)
    if not self._should_normalize_action:
      logging.info('Not normalizing infinite action range.')

  @property
  def observation_space(self):
    space = self._env.observation_space
    if not self._should_normalize_observ:
      return space
    return gym.spaces.Box(
        -np.ones(space.shape), np.ones(space.shape), dtype=np.float32)

  @property
  def action_space(self):
    space = self._env.action_space
    if not self._should_normalize_action:
      return space
    return gym.spaces.Box(
        -np.ones(space.shape), np.ones(space.shape), dtype=np.float32)

  def step(self, action):
    if self._should_normalize_action:
      action = self._denormalize_action(action)
    observ, reward, done, info = self._env.step(action)
    if self._should_normalize_observ:
      observ = self._normalize_observ(observ)
    return observ, reward, done, info

  def reset(self):
    observ = self._env.reset()
    if self._should_normalize_observ:
      observ = self._normalize_observ(observ)
    return observ

  def _denormalize_action(self, action):
    min_ = self._env.action_space.low
    max_ = self._env.action_space.high
    action = (action + 1) / 2 * (max_ - min_) + min_
    return action

  def _normalize_observ(self, observ):
    min_ = self._env.observation_space.low
    max_ = self._env.observation_space.high
    observ = 2 * (observ - min_) / (max_ - min_) - 1
    return observ

  def _is_finite(self, space):
    return np.isfinite(space.low).all() and np.isfinite(space.high).all()


class ClipAction(AttributeModifier):
  """Clip out of range actions to the action space of the environment."""

  def __init__(self, env):
    self._env = env

  @property
  def action_space(self):
    shape = self._env.action_space.shape
    return gym.spaces.Box(
        -np.inf * np.ones(shape), np.inf * np.ones(shape), dtype=np.float32)

  def step(self, action):
    action_space = self._env.action_space
    action = np.clip(action, action_space.low, action_space.high)
    return self._env.step(action)


class LimitDuration(AttributeModifier):
  """End episodes after specified number of steps."""

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def step(self, action):
    if self._step is None:
      raise RuntimeError('Must reset environment.')
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()

