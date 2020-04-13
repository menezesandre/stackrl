import functools as _ft

import pybullet as _pb
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs import registry

import gin

from siamrl.envs import data

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None

DEFAULT_EPISODE_LENGTH = 24

class Simulator(object):
  def __init__(self,
    use_gui=False,
    time_step=1/240.,
    gravity=9.8,
    spawn_position=[0,0,2],
    spawn_orientation=[0,0,0,1],
    num_steps=None,
    velocity_threshold=0.01,
    smooth_placing=True
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
      smooth_placing: whether objects are droped smoothly 
        (velocity is controled until three contact points are
        achieved).
    """
    self._connection_mode = _pb.GUI if use_gui else _pb.DIRECT
    self._time_step = time_step
    self._gravity = gravity
    self._spawn_position = spawn_position
    self._spawn_orientation = spawn_orientation
    self._num_steps = num_steps
    if not num_steps:
      self._velocity_threshold = velocity_threshold
    self._smooth = smooth_placing

    self._id = -1
    self._new = None
    self._objects = []
    self._steps_counter = np.zeros(2, dtype='uint16')

  def __call__(self, *args, **kwargs):
    """Calls step"""
    return self.step(*args, **kwargs)

  def __getattr__(self, name):
    return _ft.partial(
      getattr(_pb, name), 
      physicsClientId=self._id
    )

  def __del__(self):
    self.disconnect()

  @property
  def new_pose(self):
    return self._spawn_position, self._spawn_orientation

  @property
  def n_steps(self):
    """Number of physics engine steps taken on last iteration,
      before and after drop."""
    return tuple(self._steps_counter)

  @property
  def positions(self):
    """List of objects' positions."""
    return [self.getBasePositionAndOrientation(i)[0] 
      for i in self._objects]

  @property
  def poses(self):
    """List of objects' poses (position, orientation)."""
    return [self.getBasePositionAndOrientation(i)
      for i in self._objects]

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
    self._id = _pb.connect(self._connection_mode)
    self.setTimeStep(self._time_step)
  
  def disconnect(self):
    """Disconnect from the physics server."""
    try:
      _pb.disconnect(self._id)
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
        _pb.GEOM_BOX,
        halfExtents=[10,10,1],
        collisionFramePosition=[0,0,-1]
      ),
      baseVisualShapeIndex=self.createVisualShape(
        _pb.GEOM_BOX,
        halfExtents=[10,10,0],
        rgbaColor=[0.5,0.5,0.5,0.1]
      )
    )

    self._objects = []
    self._load(urdf)

  def step(self, position, orientation=[0,0,0,1], urdf=None):
    """Step the simulation.
    Args:
      position: catesian position [x,y,z] to place the new object.
      orientation: quaternion [x,y,z,w] of the orientation for the new 
        object.
      urdf: file of the next object to be loaded.
    """
    assert self.isConnected()

    self._place(position, orientation)

    self._steps_counter[0] = 1
    if self._smooth:
      while not self._drop():
        self.resetBaseVelocity(
          self._objects[-1], 
          [0, 0, 0], 
          [0, 0, 0]
        )
        self.stepSimulation()
        self._steps_counter[0] +=1

    self._steps_counter[1] = 0
    while not self._stop():
      self.stepSimulation()
      self._steps_counter[1] += 1

    self._load(urdf)
  
  def remove(self):
    """Remove the last placed object"""
    try:
      self.removeBody(self._objects.pop())
      return True
    except:
      return False

  def draw_rectangle(self, size, offset=[0, 0], rgb=[1, 1, 1]):
    """Create a visual rectangle on the ground plane.
    Args:
      size:  size [length, width] of the rectangle.
      offset: coordinates [x, y] of the corner with lowest coordinates of 
        the rectangle.
      rgba: color [red, green, blue] of the rectangle.
    """
    rgba=list(rgb)
    rgba.append(0.5)

    return self.createMultiBody(
      0,
      baseVisualShapeIndex=self.createVisualShape(
        _pb.GEOM_BOX,
        halfExtents=[size[0]/2, size[1]/2, 0],
        rgbaColor=rgba,
        visualFramePosition=[size[0]/2, size[1]/2, 0]
      ),
      basePosition=[offset[0], offset[1], 0]
    )

  def _load(self, urdf):
    """Load the object at urdf in the spawn pose."""
    if urdf:
      self._new = self.loadURDF(
        urdf, 
        self._spawn_position, 
        self._spawn_orientation
      )

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
  
class Observer(object):
  far = 10**3

  def __init__(self,
    simulator,
    overhead_resolution=192,
    object_resolution=32,
    pixel_size=2.**(-8),
    max_z=1
  ):
    """
    Args:
      simulator: instance of the simulator to be observed. Must support the 
        pybullet interface (used methods: computeViewMatrix,
        computeProjectionMatrix, getCameraImage).
      overhead_resolution: size of the overhead image, in pixels. Either a
        scalar for square image, or a list with [height, width]. 
      object_resolution: size of the object image, in pixels. Either a
        scalar for square image, or a list with [height, width]. 
      pixel_size: real size of a pixel in simulator units. Either a scalar 
        for square pixels, or a list with [height, width].
      max_z: maximum observable z coordinate.
    """
    # Pixel dimensions
    if np.isscalar(pixel_size):
      self._pixel_h = pixel_size
      self._pixel_w = pixel_size
    else:
      self._pixel_h = pixel_size[0]
      self._pixel_w = pixel_size[1]    

    # Overhead image dimensions
    if np.isscalar(overhead_resolution):
      self._overhead_h = int(overhead_resolution)
      self._overhead_w = int(overhead_resolution)
    else:
      self._overhead_h = int(overhead_resolution[0])
      self._overhead_w = int(overhead_resolution[1])
    # Real dimensions
    self._overhead_x = self._overhead_h*self._pixel_h
    self._overhead_y = self._overhead_w*self._pixel_w
    self._overhead_z = max_z

    # Object image dimensions
    if np.isscalar(object_resolution):
      self._object_h = int(object_resolution)
      self._object_w = int(object_resolution)
    else:
      self._object_h = int(object_resolution[0])
      self._object_w = int(object_resolution[1])
    # Real dimensions
    self._object_x = self._object_h*self._pixel_h
    self._object_y = self._object_w*self._pixel_w
    self._object_z = max(self._object_x, self._object_y)

    # Overhead camera
    self._overhead_cam = lambda: simulator.getCameraImage(
      width=self._overhead_w,
      height=self._overhead_h,
      viewMatrix=simulator.computeViewMatrix(
        cameraEyePosition=[
          self._overhead_x/2, 
          self._overhead_y/2, 
          self.far
        ],
        cameraTargetPosition=[
          self._overhead_x/2, 
          self._overhead_y/2, 
          0
        ],
        cameraUpVector=[-1,0,0]
      ),
      projectionMatrix=simulator.computeProjectionMatrix(
        left=-self._overhead_y/2,
        right=self._overhead_y/2,
        bottom=-self._overhead_x/2,
        top=self._overhead_x/2,
        nearVal=self.far-self._overhead_z,
        farVal=self.far
      )
    )

    # Object camera
    spawn_position, spawn_orientation = simulator.new_pose
    target = spawn_position
    eye, _ = simulator.multiplyTransforms(
      spawn_position,
      spawn_orientation,
      [0,0,-self.far],
      [0,0,0,1]
    )
    up, _ = simulator.multiplyTransforms(
      [0,0,0],
      spawn_orientation,
      [-1,0,0],
      [0,0,0,1]
    )
    self._object_cam = lambda: simulator.getCameraImage(
      width=self._object_w,
      height=self._object_h,
      viewMatrix=simulator.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=up
      ),
      projectionMatrix=simulator.computeProjectionMatrix(
        left=-self._object_y/2,
        right=self._object_y/2,
        bottom=-self._object_x/2,
        top=self._object_x/2,
        nearVal=self.far-self._object_z/2,
        farVal=self.far+self._object_z/2
      )
    )

    self._overhead_map = np.zeros(
      (self._overhead_h, self._overhead_w), 
      dtype='float32'
    )
    self._object_map = np.zeros(
      (self._object_h, self._object_w), 
      dtype='float32')

    # For visualization
    self.visualize = lambda: simulator.draw_rectangle(self.size)

  def __call__(self):
    """Get new depth images and convert to elevation maps"""
    # Get overhead camera depth image
    _,_,_,d,_ = self._overhead_cam()
    # Convert to elevation
    self._overhead_map = self.far - \
      self.far*(self.far-self._overhead_z)/(self.far-self._overhead_z*d)

    # Get object camera depth image
    _,_,_,d,_ = self._object_cam()
    # Convert to elevation
    self._object_map = self.far+self._object_z/2 - \
      (self.far**2-(self._object_z/2)**2)/(self.far+self._object_z*(1/2-d))
    # Flip so that the axis directions are the same as in the overhead map
    self._object_map = self._object_map[:,::-1]

  @property
  def size(self):
    """World dimensions of the observable space."""
    return self._overhead_x, self._overhead_y, self._overhead_z

  @property
  def shape(self):
    """Shape of the maps"""
    return (self._overhead_h, self._overhead_w), \
      (self._object_h, self._object_w)

  @property
  def state(self):
    """Last captured overhead map"""
    return self._overhead_map, self._object_map

  @property
  def max_z(self):
    """Maximum z coordinate of an object so that it is completely visible
      on the overhead map."""
    return self._overhead_z - self._object_z

  def pixel_to_xy(self, pixel):
    """Converts a pixel on the overhead map to a position on the XY plane."""
    return pixel[0]*self._pixel_h, pixel[1]*self._pixel_w

  def xy_to_pixel(self, position):
    """Converts a position to a pixel on the overhead map."""
    return position[0]//self._pixel_h, position[1]//self._pixel_w 

  def position(self, pixel):
    """Returns the position of the object for which the object's map is 
      overlapped with the overhead map with an offset given by pixel."""
    x,y = self.pixel_to_xy(pixel)

    z = self._overhead_map[
      pixel[0]:pixel[0]+self._object_h, 
      pixel[1]:pixel[1]+self._object_w
    ] + self._object_map 
    z = np.max(z[self._object_map>10**(-4)])
    # Correction for the object position inside the object map frame
    x += self._object_x/2
    y += self._object_y/2
    z -= self._object_z/2

    return x,y,z

class Rewarder(object):
  margin_factor = 8

  def __init__(self,
    simulator,
    observer,
    goal_size_ratio=0.5,
    occupation_ratio_weight=0.,
    occupation_ratio_param=False,
    positions_weight=0.,
    positions_param=0.,
    n_steps_weight=0.,
    n_steps_param=0.,
    contact_points_weight=0.,
    contact_points_param=0.,
    differential=True,
    seed=None
  ):
    """
    Args:
      simulator: instance of Simulator, from wich to get the simulator 
        state.
      observer: instance of Observer, from which to get the observed state
        as well as spatial information to define the goal.
      goal_size_ratio: size of the goal, given in fractions of the 
        observed space. Either None for completely random dimensions, a 
        scalar for  constant area or a list with [height, width] for 
        fixed dimensions.
      random: random number generator (used method: randint(low, high)).
        If None, numpy.random is used, no seed defined.
    """
    self._sim = simulator
    self._obs = observer

    self._shape, (self._goal_min_h,self._goal_min_w) = self._obs.shape
    self._goal_z = self._obs.max_z
    
    # Set target size
    if not goal_size_ratio:
      self._goal_size = None
    elif np.isscalar(goal_size_ratio) and \
      goal_size_ratio > 0 and goal_size_ratio <= 1\
    :
      self._goal_size = int(goal_size_ratio*self._shape[0]*self._shape[1])
    elif len(goal_size_ratio) == 2 and \
      goal_size_ratio[0] > 0 and goal_size_ratio[0] <= 1 and \
      goal_size_ratio[1] > 0 and goal_size_ratio[1] <= 1\
    :
      self._goal_size = [
        int(goal_size_ratio[0]*self._shape[0]),
        int(goal_size_ratio[1]*self._shape[1])
      ]
    else:
      raise ValueError('Invalid value for argument goal_size_ratio')

    # Initialize goal
    self._goal = np.zeros(self._shape, dtype='float32')
    self._goal_params = [[0,0],[0,0]]
    self._goal_v = 0.

    # Set reward weights and parameters
    self._or = 0.
    self._or_w = max(occupation_ratio_weight,0.)
    self._or_p = 1 if occupation_ratio_param else 0
    self._pr = 0.
    self._pr_w = max(positions_weight, 0.)
    self._pr_p = max(positions_param, 0.)
    self._sp_w = max(n_steps_weight, 0.)
    self._sp_p = max(n_steps_param, 0.)
    self._cp_w = max(contact_points_weight, 0.)
    self._cp_p = contact_points_param
    self._diff = differential

    # Set the random number generator
    self._random, seed = seeding.np_random(seed)
    self._randint = self._random.randint


  def __call__(self):
    """Returns the reward computed from the current state"""
    reward = 0
    # Ocupation ratio reward
    if self._or_w != 0:
      if self._diff:
        reward -= self._or_w*self._or
      self._or = np.sum(np.minimum(
        self._obs.state[0][self._goal_b],
        self._goal_z
      ))/self._goal_v - self._or_p
      reward += self._or_w*self._or
    # Positions reward
    if self._pr_w != 0.:
      if self._diff:
        reward -= self._pr_w*self._pr
      self._pr = 0
      for p in self._sim.positions:
        u,v = self._obs.xy_to_pixel(p[:2])
        if u >= self._goal_params[1][0] and v >= self._goal_params[1][1] \
          and u < self._goal_params[1][0]+self._goal_params[0][0] \
          and v < self._goal_params[1][1]+self._goal_params[0][1] \
        :
          self._pr += 1
        else:
          self._pr -= self._pr_p
      reward += self._pr_w*self._pr

    if self._sp_w != 0:
      reward += self._sp_w*(2**(-self._sim.n_steps[1]/self._sp_p)-1)

    if self._cp_w != 0:
      raise NotImplementedError('Contact points based reward not implemented.')

    return reward

  @property
  def goal(self):
    """Current goal map"""
    return self._goal

  @property
  def boolean_goal(self):
    return self._goal_b

  def reset(self):
    """Reset the reward memory and the goal"""
    self._or = 0.
    self._pr = 0.
    self._reset_goal()

  def seed(self, seed=None):
    """Set the seed for the random number
      generator"""
    seed = seeding.create_seed(seed)
    self._random.seed(seed)
    return [seed]

  def visualize(self):
    """Visualize the target as a green rectangle in the
    simulator"""
    size = self._obs.pixel_to_xy(self._goal_params[0])
    offset = self._obs.pixel_to_xy(self._goal_params[1])
    return self._sim.draw_rectangle(size, offset, [0,1,0])    

  def _reset_goal(self):
    """Create new goal"""
    # Target dimensions
    if not self._goal_size:
      h = self._randint(self._goal_min_h, self._shape[0]+1)
      w = self._randint(self._goal_min_w, self._shape[1]+1)
    elif np.isscalar(self._goal_size):
      h = self._randint(
        max(self._goal_min_h, self._goal_size//self._shape[1]),
        min(self._shape[0]+1, self._goal_size//self._goal_min_w)
      )
      w = min(max(
        self._goal_min_w,
        self._goal_size//h),
        self._shape[1])
    else:
      h = self._goal_size[0]
      w = self._goal_size[1]

    # Target offset
    u_max = self._shape[0] - h
    u = self._randint(
      u_max//self.margin_factor,
      (self.margin_factor-1)*u_max//self.margin_factor+1)
    v_max = self._shape[1] - w
    v = self._randint(
      v_max//self.margin_factor,
      (self.margin_factor-1)*v_max//self.margin_factor+1)

    self._goal = np.zeros(self._shape, dtype='float32')
    self._goal[u:u+h,v:v+w] = self._goal_z
    self._goal_params = [[h,w],[u,v]]
    self._goal_v = np.sum(self._goal)
    self._goal_b = self._goal != 0

class StackEnv(gym.Env):
  metadata = {
    'dtypes': ['uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64'],
    'render.modes': ['human', 'rgb_array']
  }

  def __init__(self,
    episode_length=DEFAULT_EPISODE_LENGTH,
    urdfs='train',
    object_max_dimension=0.125,
    use_gui=False,
    sim_time_step=1/60.,
    gravity=9.8,
    num_sim_steps=None,
    velocity_threshold=0.01,
    smooth_placing = True,
    observable_size_ratio=(4,6),
    resolution_factor=5,
    max_z=1,
    goal_size_ratio=1/3.,
    occupation_ratio_weight=0.,
    occupation_ratio_param=False,
    positions_weight=0.,
    positions_param=0.,
    n_steps_weight=0.,
    n_steps_param=0.,
    contact_points_weight=0.,
    contact_points_param=0.,
    differential=True,
    flat_action=True,
    dtype='float32',
    seed=None
  ):
    """
    Args:
      episode_length: Number of steps per episode (corresponds to the 
      number of objects used on each episode).
      urdfs: list of files (urdf format) that describe the objects to be 
        used on the environment. A name string can be provided to use 
        objects from the 'siamrl/envs/data/generated' directory. On each 
        episode, a fixed number of files is randomly choosen from this list.
      object_max_dimension: maximum dimension of all objects in the list.
        All objects should be completely visible within a square with this
        value as side length.
      use_gui: whether to use physics engine gafical interface. 
      sim_time_step, gravity, num_sim_steps, velocity_threshold, smooth_placing:
        see Simulator.
      observable_size_ratio: size of the observable space as a multiple of
        object_max_dimension. Either a scalar for square space or a list with
        [height, width] (as seen in the observation).
      resolution_factor: resolution is such that the number of pixels along
        object_max_dimensions is two to the power of this factor.
      goal_size_ratio, occupation_ratio_weight, occupation_ratio_param,
        positions_weight, positions_param, n_steps_weight, n_steps_param,
        contact_points_weight, contact_points_param, differential: see 
        Rewarder.
      flat_action: whether to receive action as a flat index or a pair of
        indexes [h, w].
      dtype: data type of the returned observation. Must be one of 'uint8',
        float16', 'float32' or 'float64'. Internaly, float32 is used.
      seed: Seed for the env's random number generator.
    """
    self._length = episode_length
    # Set the files list
    if isinstance(urdfs, str):
      self._list = data.generated(urdfs)
    else:
      self._list = list(urdfs)
    # Assert the list is not empty
    assert self._list, 'List of object descriptor files is empty.'
    # If the list is smaller than the episode length, files have to be
    # sampled with replacement.
    self._replace = len(self._list) < self._length

    # Set the random number generator
    self._random, seed = seeding.np_random(seed)

    # Used only when render is called in human mode.
    self._fig = None

    # Set simulator
    self._gui = use_gui
    self._sim = Simulator(
      use_gui=use_gui, 
      time_step=sim_time_step, 
      gravity=gravity, 
      spawn_position=[0,0,2*max_z], 
      spawn_orientation=[0,0,0,1],
      num_steps=num_sim_steps,
      velocity_threshold=velocity_threshold,
      smooth_placing=smooth_placing
    )

    # Set observer
    object_resolution = 2**resolution_factor
    if np.isscalar(observable_size_ratio):
      overhead_resolution = object_resolution*observable_size_ratio
    else:
      overhead_resolution = [
        object_resolution*observable_size_ratio[0],
        object_resolution*observable_size_ratio[1]
      ]
    pixel_size = object_max_dimension/object_resolution

    self._obs = Observer(
      self._sim,
      overhead_resolution=overhead_resolution,
      object_resolution=object_resolution,
      pixel_size=pixel_size,
      max_z=max_z
    )

    # Set the rewarder.
    self._rew = Rewarder(
      self._sim, 
      self._obs,
      goal_size_ratio=goal_size_ratio,
      occupation_ratio_weight=occupation_ratio_weight,
      occupation_ratio_param=occupation_ratio_param,
      positions_weight=positions_weight,
      positions_param=positions_param,
      n_steps_weight=n_steps_weight,
      n_steps_param=n_steps_param,
      contact_points_weight=contact_points_weight,
      contact_points_param=contact_points_param,
      differential=differential,
      seed=self._random.randint(2**32)
    )

    # Set the return wrapper for the observations.
    if dtype not in self.metadata['dtypes']:
      raise ValueError('Invalid value {} for argument dtype.'.format(dtype))
    if dtype == 'uint8':
      self._return = lambda x: np.array(x*(2**8-1)/max_z, dtype=dtype)
    elif dtype == 'uint16':
      self._return = lambda x: np.array(x*(2**16-1)/max_z, dtype=dtype)
    elif dtype == 'uint32':
      self._return = lambda x: np.array(x*(2**32-1)/max_z, dtype=dtype)
    elif dtype == 'uint64':
      self._return = lambda x: np.array(x*(2**64-1)/max_z, dtype=dtype)
    else:
      self._return = lambda x: np.array(x, dtype=dtype)

    obs_shape = self._obs.shape
    # Set observation space
    self.observation_space = spaces.Tuple((
      spaces.Box(
        low=0, 
        high=max_z if dtype!='uint8' else 255, 
        dtype=np.dtype(dtype),
        shape=(
          obs_shape[0][0],
          obs_shape[0][1],
          2
        )
      ),
      spaces.Box(
        low=0, 
        high=object_max_dimension if dtype!='uint8' else 255, 
        dtype=np.dtype(dtype), 
        shape=(
          obs_shape[1][0],
          obs_shape[1][1],
          1
        )
      )
    ))
    # Set action space
    if flat_action:
      self._action_width = obs_shape[0][1] - obs_shape[1][1] + 1
      self.action_space = spaces.Discrete(
        (obs_shape[0][0] - obs_shape[1][0] + 1)*self._action_width
      )
    else:
      self._action_width = None
      self.action_space = spaces.MultiDiscrete([
        obs_shape[0][1] - obs_shape[1][1] + 1, 
        obs_shape[0][0] - obs_shape[1][0] + 1
      ])

    # Set the flag to true to ensure environment is reset before step.
    self._done = True

  def __del__(self):
    self.close()

  @property
  def observation(self):
    """Agent's observation of the environment state."""
    m,n = self._obs.state
    g = self._rew.goal
    return self._return(np.stack([m,g],axis=-1)), \
      self._return(n[:,:,np.newaxis])

  def step(self, action):
    # Reset if necessary.
    if self._done:
      return self.reset(), 0, False, {}
    # Assert action is valid
    assert self.action_space.contains(action), 'Invalid action.'
    # Unflatten action if necessary.
    if self._action_width:
      action = [action//self._action_width, action%self._action_width]
    # Pop next urdf if list isn't empty, otherwise episode is done.
    if self._episode_list:
      urdf = self._episode_list.pop()
    else:
      urdf = None
      self._done = True

    # Run simulation with given action.
    self._sim(position=self._obs.position(action), urdf=urdf)
    # Get observation of the new state.
    self._obs()
    # Compute reward.
    reward = self._rew()

    return self.observation, reward, self._done, {}

  def reset(self):
    # Get episode's list of urdfs
    self._episode_list = list(self._random.choice(
      self._list, 
      size=self._length,
      replace=self._replace
    ))
    # Reset simulator.
    self._sim.reset(self._episode_list.pop())
    # Reset rewarder.
    self._rew.reset()
    # Get first observation.
    self._obs()

    # If using GUI, mark observable space and goal.
    if self._gui:
      self._obs.visualize()
      self._rew.visualize()

    self._done = False
    return self.observation

  def render(self, mode='human'):
    if mode not in self.metadata['render.modes']:
      return super(StackEnv, self).render(mode=mode)

    m, n = self._obs.state
    _max = np.max(m)
    r = m/_max if _max!=0 else m
    b = 1 - r
    g = np.ones(r.shape)*0.5
    g[self._rew.boolean_goal] += 0.1
    rgb0 = np.stack([r,g,b], axis=-1)

    _max = np.max(n)
    r = n/_max if _max!=0 else n
    b = 1 - r
    g = np.ones(r.shape)*0.5
    rgb1 = np.stack([r,g,b], axis=-1)
    
    if mode == 'human':
      if plt is None:
        raise ImportError("'render' requires matplotlib.pyplot to run in 'human' mode.")
        

      if not (self._fig and plt.fignum_exists(self._fig.number)):
        width_ratio = rgb0.shape[1]//rgb1.shape[1]
        self._fig, self._axs = plt.subplots(
          1, 2, 
          gridspec_kw={'width_ratios':[width_ratio, 1]}
        )

      self._axs[0].cla()
      self._axs[0].imshow(rgb0)
      self._axs[1].cla()
      self._axs[1].imshow(rgb1)
      self._fig.show()
      
    elif mode == 'rgb_array':
      return rgb0, rgb1

  def close(self):
    if plt and self._fig and plt.fignum_exists(self._fig.number):
      plt.close(self._fig)
      self._fig = None
    self._sim.disconnect()
    self._done = True

  def seed(self, seed=None):
    """Set the seed for the env's random number
      generator"""
    seed = seeding.create_seed(seed)
    self._random.seed(seed)
    return [seed]+self._rew.seed(self._random.randint(2**32))

@gin.configurable
def register(env_id=None, **kwargs):
  """Register StackEnv in the gym registry.
  Args:
    env_id: id with which the environment will be registered. If None,
      id is of the form 'Stack-v%d', using the lowest unregistered
      version number.
    kwargs: key word arguments for StackEnv.
  Returns:
    Registered environment's id.
  Raises:
    gym.error.Error: if provided env_id is already registered.  
  """
  if not env_id:
    ids = [key for key in registry.env_specs if 'Stack-v' in key]
    i = 0
    while 'Stack-v%d'%i in ids:
      i+=1
    env_id = 'Stack-v%d'%i
  if 'episode_length' in kwargs:
    max_episode_steps = kwargs['episode_length']
  else:
    max_episode_steps = DEFAULT_EPISODE_LENGTH
  gym.register(
    id=env_id,
    max_episode_steps=max_episode_steps,
    entry_point=StackEnv,
    kwargs = kwargs
  )

  return env_id

@gin.configurable
def curriculum(goals=[], ckwargs={}, **kwargs):
  """Registers a set of environments to form a curriculum.
  Args:
    goals: list of goal reward for each environment.
    ckwargs: changing keyword arguments. Dictionary of lists.
      Each list has the values of the corresponding keyword
      argument for each environment.
    kwargs: constant keyword aguments (for all environments
      registered).
  Returns:
    Dictionary with registered environment ids as keys and 
      respective goals as values.
  """
  ids = {}
  # Turn dict of lists to list of dicts
  ckwargs = [dict(zip(ckwargs,values)) for values in zip(*ckwargs.values())]
  for g, a in zip(goals, ckwargs):
    env_id = register(**a, **kwargs)
    ids[env_id] = g
  return ids
