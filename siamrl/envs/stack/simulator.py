import functools

import numpy as np
import pybullet as pb

class Simulator(object):
  def __init__(
    self,
    use_gui=False,
    time_step=1/240.,
    gravity=9.8,
    spawn_position=(0,0,2),
    spawn_orientation=(0,0,0,1),
    num_steps=None,
    velocity_threshold=0.01,
  ):
    """
    Args:
      use_gui: whether to run the physics server in grafical 
        mode.
      connection_mode: physics server mode (pybullet.GUI for
        graphical version)
      time_step: physics engine time step in fraction of seconds.
      gravity: gravity force, applied along the -Z axis.
      spawn_position: coordinates [x,y,z]. Position where a 
        new object is spawned.
      spawn_orientation: quaternion [x,y,z,w]. Orientation at 
        which a new object is spawned.
      num_steps: number of steps the physics engine takes on 
        each call of step. If None, a stop criterion is used.
      velocity_threshold: simulation stops when all objects'
        velocities are bellow this value. Only used if num_steps 
        is None.
    """
    self._connection_mode = pb.GUI if use_gui else pb.DIRECT
    self._time_step = time_step
    self._gravity = gravity
    self._spawn_position = tuple(spawn_position)
    self._spawn_orientation = tuple(spawn_orientation)
    self._num_steps = num_steps
    if not num_steps:
      self._velocity_threshold = velocity_threshold

    self._id = -1
    self._new = None
    self._objects = []
    self._steps_counter = np.zeros(2, dtype='uint16')

  def __call__(self, *args, **kwargs):
    """Calls step"""
    return self.step(*args, **kwargs)

  def __getattr__(self, name):
    return functools.partial(
      getattr(pb, name), 
      physicsClientId=self._id
    )

  def __del__(self):
    self.disconnect()

  @property
  def new_pose(self):
    return self._spawn_position, self._spawn_orientation

  @property
  def has_new_object(self):
    """True if any new object was added since last check of this property."""
    if hasattr(self, '_has_new_object') and self._has_new_object:  # pylint: disable=access-member-before-definition
      self._has_new_object = False
      return True
    else:
      return False

  @property
  def n_steps(self):
    """Number of physics engine steps taken on last iteration,
      before and after drop."""
    return tuple(self._steps_counter)

  @property
  def positions(self):
    """List of objects' positions."""
    return [p[0] for p in self._final_poses] if hasattr(self, '_final_poses') else []

  @property
  def poses(self):
    """List of objects' poses (position, orientation)."""
    return self._final_poses if hasattr(self, '_final_poses') else []

  @property
  def distances(self):
    """List of distances (translation, rotation) between the positions of
      each object at the beginning and at the end of the last simulator 
      step."""
    if hasattr(self, '_distances') and self._distances is not None:
      return self._distances
    elif hasattr(self, '_initial_poses') and hasattr(self, '_final_poses'):
      self._distances = []
      for (ip,io), (fp,fo) in zip(self._initial_poses, self._final_poses):
        dp = np.linalg.norm(np.subtract(ip, fp))
        do = 2*np.arccos(min(self.getDifferenceQuaternion(io, fo)[-1], 1.))
        self._distances.append((dp, do))

      return self._distances
    else:
      return []

  @property
  def distances_from_place(self):
    """List of distances (translation, rotation) between the original 
      placing positions of each object and at their current position."""
    if hasattr(self, '_distances_from_place') and self._distances_from_place is not None:
      return self._distances_from_place
    elif hasattr(self, '_place_poses') and hasattr(self, '_final_poses'):
      self._distances_from_place = []
      for (ip,io), (fp,fo) in zip(self._place_poses, self._final_poses):
        dp = np.linalg.norm(np.subtract(ip, fp))
        do = 2*np.arccos(min(self.getDifferenceQuaternion(io, fo)[-1], 1.))
        self._distances_from_place.append((dp, do))

      return self._distances_from_place
    else:
      return []

  @property
  def contact_points(self):
    """List of contact points of the last placed object."""
    return self.getContactPoints(self.objects[-1])

  @property
  def state(self):
    """Simulation state"""
    return self.n_steps, self.positions, self.contact_points

  def connect(self):
    """Connect to the physics server."""
    self._id = pb.connect(self._connection_mode)
    self.setTimeStep(self._time_step)
  
  def disconnect(self):
    """Disconnect from the physics server."""
    try:
      pb.disconnect(self._id)
    except:
      pass
    finally:
      self._id = -1

  def reset(self, urdf=None):
    """Reset simulation.
    Args:
      urdf: file of the first object to be loaded.
    """
    if self.isConnected():
      self.resetSimulation()
    else:
      self.connect()

    self.setGravity(0, 0, -self._gravity)
    self.createMultiBody(
      0,
      baseCollisionShapeIndex=self.createCollisionShape(
        pb.GEOM_BOX,
        halfExtents=[10,10,1],
        collisionFramePosition=[0,0,-1]
      ),
      baseVisualShapeIndex=self.createVisualShape(
        pb.GEOM_BOX,
        halfExtents=[10,10,0],
        rgbaColor=[0.5,0.5,0.5,0.1]
      )
    )

    self._objects = []

    self._place_poses = []
    self._initial_poses = []
    self._final_poses = []
    self._distances = None
    self._distances_from_place = None
    self._load(urdf)

  def step(
    self, 
    position, 
    orientation=(0,0,0,1), 
    urdf=None, 
    smooth_placing=False
  ):
    """Step the simulation.
    Args:
      position: catesian position [x,y,z] to place the new object.
      orientation: quaternion [x,y,z,w] of the orientation for the new 
        object.
      urdf: file of the next object to be loaded.
      smooth_placing: whether objects are droped smoothly 
        (velocity is controled until three contact points are
        achieved).      
    """
    assert self.isConnected()

    self._place(position, orientation)

    # Store the pose where the object was originally placed
    self._place_poses.append(self.getBasePositionAndOrientation(self._objects[-1]))

    self._steps_counter[0] = 1
    if smooth_placing:
      while not self._drop():
        self.resetBaseVelocity(
          self._objects[-1], 
          [0, 0, 0], 
          [0, 0, 0]
        )
        self.stepSimulation()
        self._steps_counter[0] +=1

    # Get poses of all objects at the beggining of the simulator step.
    self._initial_poses = [
      self.getBasePositionAndOrientation(i)
      for i in self._objects
    ]

    self._steps_counter[1] = 0
    while not self._stop():
      self.stepSimulation()
      self._steps_counter[1] += 1

    # Get poses of all objects at the end of the simulator step.
    self._final_poses = [
      self.getBasePositionAndOrientation(i)
      for i in self._objects
    ]
    # Set distances as None so that they are recalculated.
    self._distances = None
    self._distances_from_place = None

    self._load(urdf)
  
  def remove(self):
    """Remove the last placed object"""
    try:
      self.removeBody(self._objects.pop())
      return True
    except:
      return False

  def draw_rectangle(self, size, offset=(0, 0), rgb=(1, 1, 1)):
    """Create a visual rectangle on the ground plane.
    Args:
      size:  size [length, width] of the rectangle.
      offset: coordinates [x, y] of the corner with lowest coordinates of 
        the rectangle.
      rgba: color [red, green, blue] of the rectangle.
    """
    rgba = tuple(rgb) + (0.5,)

    return self.createMultiBody(
      0,
      baseVisualShapeIndex=self.createVisualShape(
        pb.GEOM_BOX,
        halfExtents=(size[0]/2, size[1]/2, 0),
        rgbaColor=rgba,
        visualFramePosition=(size[0]/2, size[1]/2, 0)
      ),
      basePosition=(offset[0], offset[1], 0),
    )

  def _load(self, urdf):
    """Load the object at urdf in the spawn pose."""
    if urdf:
      self._new = self.loadURDF(
        urdf, 
        self._spawn_position, 
        self._spawn_orientation
      )
      self._has_new_object = True

  def _place(self, position, orientation):
    """Reset the new object's pose to given position and orientation."""
    if self._new:
      self.resetBasePositionAndOrientation(
        self._new, 
        position, 
        orientation
      )
      self._objects.append(self._new)
      self._new = None
      self.stepSimulation()
    
  def _stop(self):
    """Returns True if the simulation stop criterion is
      achieved"""
    if self._num_steps:
      return self._step_counter >= self._num_steps
    else:
      for obj in self._objects[::-1]:
        # As the last object is the most likely to be moving,
        # iterating the list backwards assures only one object
        # is checked for most steps.
        v,_ = self.getBaseVelocity(obj)        
        if np.linalg.norm(v) > self._velocity_threshold:
          return False
      return True

  def _drop(self):
    """Returns True when the new object has at least 3 contact
      points, or the stop criterion is achieved"""
    return len(self.getContactPoints(self._objects[-1])) >= 3 \
      or self._stop()

class TestSimulator(Simulator):
  def __init__(self, **kwargs):
    super(TestSimulator, self).__init__(**kwargs)
    
    self._spawn_offset = np.array((self._spawn_position[-1], 0, 0))
    self._spawn_position = np.array(self._spawn_position)
    self._spawn_positions = []
    self._news = []
    self._dynamics = []

  @property
  def new_pose(self):
    return [(pos, self._spawn_orientation) for pos in self._spawn_positions]

  def reset(self, urdfs):
    super(TestSimulator, self).reset()

    self._spawn_positions = []
    self._news = []
    self._dynamics = []

    for i,urdf in enumerate(urdfs):
      pos = tuple(self._spawn_position + i*self._spawn_offset)
      self._spawn_positions.append(pos)
      new = self.loadURDF(urdf, pos, self._spawn_orientation)
      self._dynamics.append(self.getDynamicsInfo(new,-1))
      self.changeDynamics(new, -1, mass=0, localInertiaDiagonal=(0,0,0))
      self._news.append(new)
    self._has_new_object = True

  def step(self, index, **kwargs):
    self._new = self._news.pop(index)
    self._spawn_positions.pop(index)
    dynamics = self._dynamics.pop(index)
    self.changeDynamics(self._new, -1, mass=dynamics[0], localInertiaDiagonal=dynamics[2])
    super(TestSimulator, self).step(**kwargs)