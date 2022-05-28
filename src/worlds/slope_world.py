"""Build a simple world with plane only."""


class SlopeWorld:
  """Builds a simple world with a plane only."""
  def __init__(self,
               pybullet_client,
               slope_center_x=1.5,
               slope_center_z=0.5,
               slope_angle=-0.4,
               slope_length=20,
               slope_width=2):
    self._pybullet_client = pybullet_client
    self._slope_center_x = slope_center_x
    self._slope_center_z = slope_center_z
    self._slope_angle = slope_angle
    self._slope_length = slope_length
    self._slope_width = slope_width

  def build_world(self):
    """Constructs the world with a single slope."""
    p = self._pybullet_client
    ground_id = self._pybullet_client.loadURDF('plane.urdf')

    slope_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[self._slope_length / 2, self._slope_width / 2, 0.05])
    slope_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[self._slope_length / 2, self._slope_width / 2, 0.05])

    slope_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=slope_collision_id,
        baseVisualShapeIndex=slope_visual_id,
        basePosition=[self._slope_center_x, 0, self._slope_center_z],
        baseOrientation=p.getQuaternionFromEuler((0., self._slope_angle, 0.)))
    # Change friction coef to 1 (or anything you want). I think 1 is default
    # but you can play with this value.
    p.changeDynamics(slope_id, -1, lateralFriction=10.)
    return ground_id
