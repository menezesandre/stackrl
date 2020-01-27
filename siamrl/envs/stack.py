import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as pb
from pybullet_utils import bullet_client

import numpy as np

import types

import siamrl as s
from siamrl.envs.utils import ElevationCamera
from siamrl.envs.data import getGeneratedURDF

MIN_DIV = 2.**5 #image dimentions must be divisible by this
CAMERA_DISTANCE = 10.**3
GRAVITY = 9.8
OBJ_MAX_DEPTH = 2.**(-3)
OBJ_MAX_SIDE = 2.**(-3)
MAX_ELEVATION = 0.5
VELOCITY_THRESHOLD = 0.01

def avg_occupied(img):
  ocupied = np.sum(img != 0)
  if ocupied != 0:
    return np.sum(img)/ocupied
  else:
    return 0.

def max_occupied(img):
  ocupied = np.sum(img != 0)
  return np.max(img)/np.sqrt(max(ocupied,6144))

class BaseStackEnv(gym.Env):
  """Pybullet implementation of an abstract gym environment whose goal is to stack 
  the models of a given urdf list on a base rectangle of given size.

  - observation - overhead elevation image of the stack and elevation image of the 
  downside of the new object:
    Tuple(Box(<overhead image height>, <overhead image width>, 1),
          Box(<object image height>, <object image width>, 1))

  - action - Converted to the new object pose. Format to be chosen with 
    flat_action argument on __init__:
    True (flat)- Linear index of the point on overhead image where to place the 
    new object:
      Discrete((<overhead image height> - <object image height> + 1)
              *(<overhead image width> - <object image width> + 1))
    False (grid) - Pair of coordinates representing the point on overhead image
    where to place the new object:
      Multidiscrete(<overhead image height> - <object image height> + 1,
                    <overhead image width> - <object image width> + 1)

  - reward - To be chosen with state_reward, drop_penalty and settle_penalty arguments
  on __init__. Differential reward computed from the overhead image, 
  penalized by droped objects (fallen off the environment limits) and time taken for the 
  structure to stabilize after a placed object

  Subclasses can override the following methods:
    _observation() - to change the observation returned by step() (set 
      observation_space acordingly) 
    _action_to_pose() - to change the way actions are intrepreted (set
      action_space acordingly)
    _reward() - to change the reward returned by step()
    _get_data_path() - to use models from a different path

  Subclasses must override the following methods:
    _get_urdf_list() - Returns a list of (urdf) file names of the models to be used,
      located at the path returned from _get_data_path(). Should use num_objects as the
      lenght of the returned list
  """
  metadata = {'render.modes': ['depth_array']}

  def __init__(self,
              base_size=[0.3125, 0.4375],
              resolution=2.**(-9),
              num_objects=100,
              gui=False,
              flat_action=True,
              state_reward=None,
              differential_reward=True,
              settle_penalty=None,
              drop_penalty=0.,
              reward_scale=1.,
              info = False,
              seed = None):
    """
    Args:
      base_size: dimensions of the base of the structure, in m.
      resolution: size of a pixel of the observation images, in m.
      num_objects: number of objects used in an episode
      gui: whether to use the pybullet GUI.
      flat_action: defines the format of actions (True - flat, 
        False - grid).
      state_reward: Either a callable that computes the reward
        from the overhead observation, or a string if one of the
        built-in functions is to be used ('avg', 'avg_occ', 'max',
        'max_occ'). If None or invalid, no state reward is used.
      differential_reward: Whether the reward is the difference 
        between current state and last state rewards or absolute
        current state reward.
      settle_penalty: callable that returns a penalty given the 
        number of simulation steps taken for the structure to 
        settle.
      drop_penalty: penalty aplied to the reward when an object 
        falls of the environment.
      reward_scale: scaling factor by which returned reward is
        multiplied
      info: Whether to return info from step
      seed: seed for the random generator.
    """
    # Make image dimensions divisible by MIN_DIV
    self.size = np.round(np.array(base_size)/(resolution*MIN_DIV))*resolution*MIN_DIV
    self.resolution = resolution    
    self.spawn_z = 2*MAX_ELEVATION
    self.num_objects = num_objects
    self._info = info

    # Connect to the physics server
    self.connection_mode = pb.GUI if gui else pb.DIRECT
    self.sim = BulletClient(connection_mode=self.connection_mode)
    self.sim.setAdditionalSearchPath(self._get_data_path())

    # Set the cameras for observation
    self.overhead_cam = ElevationCamera(client=self.sim, cameraPos=[0,0,CAMERA_DISTANCE],
      width=self.size[1], height=self.size[0], depthRange=[-MAX_ELEVATION - 2*OBJ_MAX_DEPTH,
      0], resolution=resolution)
    self.object_cam = ElevationCamera(client=self.sim, targetPos=[0, 0, self.spawn_z],
      cameraPos=[0,0,self.spawn_z-CAMERA_DISTANCE], width=OBJ_MAX_SIDE, height=OBJ_MAX_SIDE, 
      depthRange=[-OBJ_MAX_DEPTH, OBJ_MAX_DEPTH], resolution=resolution)
    self._overhead_img = np.array([])
    self._object_img = np.array([])

    # Set observation space
    self.observation_space = spaces.Tuple((
      spaces.Box(low=0.0, high=MAX_ELEVATION, shape=(self.overhead_cam.height,
        self.overhead_cam.width, self.overhead_cam.channels), dtype=np.float32),
      spaces.Box(low=0.0, high=2*OBJ_MAX_DEPTH, shape=(self.object_cam.height, 
        self.object_cam.width, self.object_cam.channels), dtype=np.float32)
      ))

    # Set action space
    if flat_action:
      self.flat_action = True
      self.action_width = self.overhead_cam.width - self.object_cam.width + 1
      self.action_space = spaces.Discrete((self.overhead_cam.height - 
        self.object_cam.height + 1)*self.action_width)
    else:
      self.flat_action = False
      self.action_space = spaces.MultiDiscrete(
        [self.overhead_cam.height - self.object_cam.height + 1, 
        self.overhead_cam.width - self.object_cam.width + 1])

    # Set reward operations
    if isinstance(state_reward, str):
      if state_reward in ['mean', 'avg']:
        self._reward_op = np.mean
      elif state_reward == 'avg_occ':
        self._reward_op = avg_occupied
      elif state_reward == 'max':
        self._reward_op = np.max
      elif state_reward == 'max_occ':
        self._reward_op = max_occupied
      else:
        self._reward_op = lambda x: 0.
    elif isinstance(state_reward, types.FunctionType):
      try:
        dummy = 0. + state_reward(np.zeros(
            (self.overhead_cam.height, self.overhead_cam.width)))
      except:
        self._reward_op = lambda x: 0.        
      else:
        if np.isscalar(dummy):
          self._reward_op = state_reward
        else:
          self._reward_op = lambda x: 0.
    else:
      self._reward_op = lambda x: 0.

    if isinstance(settle_penalty, types.FunctionType):
      try:
        dummy = 0. + settle_penalty(0.)
      except:
        self._settle_penalty = lambda x: 0.
      else:
        if np.isscalar(dummy):
          self._settle_penalty = settle_penalty
        else:
          self._settle_penalty = lambda x: 0.
    else:
      self._settle_penalty = lambda x: 0.

    self._differential_reward = differential_reward
    self._drop_penalty = drop_penalty
    self._reward_scale = reward_scale

    # Seed the random generator
    self.seed(seed)
    # Set this to true to ensure no step before the first reset
    self._done = True

  def step(self, action):
    # Reset if necessary
    if self._done:
      return self.reset()
    # Assert action is in the action space
    assert self.action_space.contains(action)

    # Set last object's position according to given action
    position, orientation = self._action_to_pose(action)
    self.sim.resetBasePositionAndOrientation(self._object_ids[-1], position, orientation)
    self.sim.stepSimulation()
    counter = 0
    # Step simulation until all objects settle
    while not self._step_done():
      self.sim.stepSimulation()
      counter +=1

    # Add the penalty for the time taken to settle
    self._penalty += self._settle_penalty(counter)

    # Spawn the object to be placed on next step
    self._new_state()

    if self._info:
      info = {'max': np.max(self._overhead_img),
              'mean': np.mean(self._overhead_img),
              'steps': counter,
              'n_obj': len(self._object_ids)}
    else:
      info = {}

    return self._observation(), self._reward(), self._done, info

  def reset(self):
    # Connect to physics server if disconnected, otherwise reset simulation
    if self.sim._client < 0:
      self.sim = BulletClient(connection_mode=self.connection_mode)
      self.sim.setAdditionalSearchPath(self._get_data_path())
      self.overhead_cam.setClient(self.sim)
      self.object_cam.setClient(self.sim)
    else:
      self.sim.resetSimulation()
    # Get new list of urdf models
    self._urdf_list = self._get_urdf_list()
    # Reset current accumulated penalty
    self._penalty = 0.
    # Empty object ids list
    self._object_ids = []
    # Set end of episode flag to false
    self._done = False
    # Set gravity
    self.sim.setGravity(0, 0, -GRAVITY)
    # Load the ground plane
    self.sim.loadURDF('plane.urdf')
    # Spawn the first object to be placed
    self._new_state()
    return self._observation()

  def render(self, mode='depth_array'):
    if mode == 'depth_array':
      return self._observation()
    else:
      return super(BaseStackEnv, self).render(mode = mode)

  def close(self):
    self.sim.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _observation(self):
    return (self._overhead_img.copy(), self._object_img.copy())

  def _reward(self, **kwargs):
    """
    Computes this step's reward as the difference between state's
      reward and the acumulated penalty
    """
    reward = self._reward_op(self._overhead_img) - self._penalty

    # If using diffetential rewards, set the penalty of the next
    # step as the current state reward.
    # Note: this operation is assigning 
    # penalty + reward_op(state) - penalty = reward_op(state)
    if self._differential_reward:
      self._penalty += reward
    else:
      self._penalty = 0.
    return reward*self._reward_scale

  def _new_state(self):
    """
    Prepare next state of the environment: Spawn new object, store
      new observations
    """
    # Pop a file from the urdf list and load it into the simulator
    if len(self._urdf_list) > 0:
      filename = self._urdf_list.pop()
      self._object_ids.append(self.sim.loadURDF(filename, 
          basePosition=[0, 0, self.spawn_z]))
    else:
      # The episode ends when the list is empty 
      self._done = True

    # Take new observations
    self._overhead_img = self.overhead_cam()
    self._object_img = self.object_cam(flip='w')

    # This is unnecessary if MAX_ELEVATION is big enough for the 
    # number of objects
#    if np.max(self._overhead_img) >= MAX_ELEVATION:
#      self._done = True

  def _step_done(self):
    """
    Check if this step's simulation is done. The condition is that
      the velocity of every object is bellow some threshold (meaning
      that the structure has settled).
    Objects off the limits get removed.

    Returns:
      True if all objects' velocities are bellow threshold,
      False otherwise.
    """
    # As the last object is the most likely to be moving, iterating
    # the list backwards assures only one object is checked for most
    # steps
    for obj in self._object_ids[::-1]:
      pos, _ = self.sim.getBasePositionAndOrientation(obj)
      # If object fell off the base, remove it and add the penalty 
      if pos[2] < 0:
        self.sim.removeBody(obj)
        self._object_ids.remove(obj)
        self._penalty += self._drop_penalty
        continue
      if np.linalg.norm(np.array(self.sim.getBaseVelocity(obj))[0,:]
          ) > VELOCITY_THRESHOLD:
        return False
    return True

  def _action_to_pose(self, action):
    """
    Converts the index(es) from action into a pose on the physical
      environment
    """
    if self.flat_action:
      action = [action//self.action_width, action%self.action_width]
    # Convert the pixel indexes to a (x,y) position
    x = (action[0]+self.object_cam.height/2)*self.resolution - self.size[0]/2
    y = (action[1]+self.object_cam.width/2)*self.resolution - self.size[1]/2
    # Calculate z as the lowest point before colision between object and structure
    elevation = self._overhead_img[action[0]:action[0]+self.object_cam.height, 
      action[1]:action[1]+self.object_cam.width] + self._object_img 
    z = np.max(elevation[self._object_img>10**(-3)]) - OBJ_MAX_DEPTH
    return [x, y, z], [0,0,0,1]

  def _get_data_path(self):
    return s.envs.getDataPath()

  def _get_urdf_list(self):
    raise NotImplementedError

class CubeStackEnv(BaseStackEnv):
  """
  Stack environment where the objects to stack are cubes of the same size
  """
  def _get_urdf_list(self):
    return ['cube.urdf']*self.num_objects

class GeneratedStackEnv(BaseStackEnv):
  """
  Stack environment where the objects to stack are models with a
  given common name from the 'data/generated' directory
  """
  def __init__(self,
               model_name='ic',
               **kwargs):
    """
    Args:
      model_name: Common name of the models to be used
        (i.e. '<model_name>*.urdf')
    """
    self._full_list = getGeneratedURDF(model_name)
    assert len(self._full_list) > 0
    super(GeneratedStackEnv, self).__init__(**kwargs)


  def _get_urdf_list(self):
    """
    Sample the required number of objects from the full list of
    available objects. If required number is greater or equal to
    the available number, sample with replacement. Otherwise,
    objects are unique in each episode

    Returns:
      The list of sampled file names
    """
    replace = len(self._full_list) <= self.num_objects
    return list(self.np_random.choice(self._full_list, 
        size=self.num_objects, replace=replace))

class GeneratingStackEnv(BaseStackEnv):
  """
  Stack environment where the objects to stack are temporary 
  models generated at the beginning of each episode with a given
  function
  """
  def __init__(self,
               generator, 
               generator_args,
               **kwargs):
    raise NotImplementedError
    super(GeneratingStackEnv, self).__init__(**kwargs)
    
  # TODO

class BulletClient(bullet_client.BulletClient):
  """
  Overrides the __del__ method of BulletClient from pybullet_utils,
  The original method catches a pybullet.error exception that does
  not inherit from BaseException, which is not allowed and raises
  a TypeError.
  """
  def __del__(self):
    # Disconnect if still connected.
    # Verify if pb hasn't been cleared before when program exits
    if self._client>=0 and pb:
      pb.disconnect(physicsClientId=self._client)
      self._client = -1
