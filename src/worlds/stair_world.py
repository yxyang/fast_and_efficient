"""Build a simple world with plane only."""


class StairWorld:
  """Builds a simple world with a plane only."""
  def __init__(self,
               pybullet_client,
               num_steps: int = 30,
               stair_height: float = 0.08,
               stair_length: float = 0.25,
               first_step_at: float = 1.):
    self._pybullet_client = pybullet_client
    self._num_steps = num_steps
    self._stair_height = stair_height
    self._stair_length = stair_length
    self._first_step_at = first_step_at

  def build_world(self):
    """Builds world with stairs."""
    p = self._pybullet_client
    ground_id = self._pybullet_client.loadURDF('plane.urdf')
    stair_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[self._stair_length / 2, 1, self._stair_height / 2])
    stair_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[self._stair_length / 2, 1, self._stair_height / 2])

    curr_x, curr_z = self._first_step_at, self._stair_height / 2
    for _ in range(self._num_steps):
      stair_id = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=stair_collision_id,
                                   baseVisualShapeIndex=stair_visual_id,
                                   basePosition=[curr_x, 0, curr_z])
      # Change friction coef to 1 (or anything you want). I think 1 is default
      # but you can play with this value.
      p.changeDynamics(stair_id, -1, lateralFriction=10.)
      curr_x += self._stair_length
      curr_z += self._stair_height

    return ground_id
