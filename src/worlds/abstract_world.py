"""Abstract base class for world builder."""


class AbstractWorld:
  def __init__(self, pybullet_client):
    self.pybullet_client = pybullet_client

  def build_world(self):
    raise NotImplementedError(
        "You are calling the abstract class of world builder. "
        "Please implement this function in a subclass.")
