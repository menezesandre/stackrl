import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import numpy as np

import siamrl as s
from siamrl.envs.utils import ElevationCamera

MIN_DIV = 2.**5 
# as the network may perform polling, transpose convolutions, and concatenation with
# previous layers, the images' dimensions must be divisible by this number to avoid
# dim mismatch on concat.
CAMERA_DISTANCE = 10.**3
GRAVITY = 9.8
OBJ_MAX_DEPTH = 2.**(-3)
OBJ_MAX_SIDE = 2.**(-3)
MAX_ELEVATION = 1.0
MAX_EPISODE_STEPS = 1000
VELOCITY_THRESHOLD = 0.01

class BaseStackEnv(gym.Env):
  """ 
  Pybullet implementation of an abstract gym environment whose goal is to reach a 
  target height by stacking models from a given urdf list on a base rectangle of 
  given size.

- observation - overhead elevation image of the stack and elevation image of the 
downside of the new object:
  Tuple(Box(<overhead image height>, <overhead image width>, 1),
        Box(<object image height>, <object image width>, 1))
- action - Pair of coordinates representing the point on overhead image where to 
place the new object (converted to the new object pose):
  Multidiscrete(<overhead image height> - <object image height> + 1,
                <overhead image width> - <object image width> + 1)
- reward - Differencial maximum elevation on the overhead image

  Subclasses can override the following methods:
    _observation() - to change the observation returned by step() 
    _action_to_pose() - to change the way actions are intrepreted (set action_space
      acordingly)
    _reward() - to change the reward returned by step()
    _get_data_path() - to use models from a different path

  Subclasses must override the following methods:
    _get_urdf_list() - Should return a list of file names of the models to be used,
      located at the data path
  """
  metadata = {'render.modes': ['human', 'depth_array']}

  def __init__(self, baseSize=[.5, .5], targetHeight=1.0, resolution = 2.**(-9),
               gui=False, seed = None):
    #assert image dimensions are divisible by MIN_DIV
    self.size = np.round(np.array(baseSize)/(resolution*MIN_DIV))*resolution*MIN_DIV
    assert targetHeight > 0.
    self.max_elevation = targetHeight
    self.resolution = resolution
    self.gui = gui
    self.seed(seed)
    
    self.spawn_z = 2*self.max_elevation
    self.overhead_cam = ElevationCamera(cameraPos=[0,0,CAMERA_DISTANCE], width=
      self.size[1], height=self.size[0], depthRange=[-self.max_elevation - 
      2*OBJ_MAX_DEPTH, 0], resolution=resolution)
    self.object_cam = ElevationCamera(targetPos=[0, 0, self.spawn_z], cameraPos=
      [0,0,self.spawn_z-CAMERA_DISTANCE], width=OBJ_MAX_SIDE, height=OBJ_MAX_SIDE, 
      depthRange=[-OBJ_MAX_DEPTH, OBJ_MAX_DEPTH], resolution=resolution)

    self._overhead_img = np.array([])
    self._object_img = np.array([])

    self.physicsClient = -1
    self._urdf_list = self._get_urdf_list()

    self.observation_space = spaces.Tuple((
      spaces.Box(low=0.0, high=1.0, shape=(self.overhead_cam.height,
        self.overhead_cam.width, self.overhead_cam.channels), dtype=np.float32),
      spaces.Box(low=0.0, high=1.0, shape=(self.object_cam.height, 
        self.object_cam.width, self.object_cam.channels), dtype=np.float32)
      ))
    self.action_space = spaces.MultiDiscrete(
      [self.overhead_cam.height - self.object_cam.height + 1, 
       self.overhead_cam.width - self.object_cam.width + 1])

  def step(self, action):
    self._step_count += 1

    position, orientation = self._action_to_pose(action)
    p.resetBasePositionAndOrientation(self._object_ids[-1], position, orientation, 
      physicsClientId=self.physicsClient)
    p.stepSimulation(physicsClientId=self.physicsClient)
    while not self._all_velocities_below(VELOCITY_THRESHOLD):
      p.stepSimulation(physicsClientId=self.physicsClient)
    self._new_object()

    return self._observation(), self._reward(), self._done(), {}

  def reset(self):
    if self.physicsClient >= 0:
      p.disconnect(self.physicsClient)

    self._reward_mem = 0.
    self._object_ids = []
    self._step_count = 0

    if self.gui:
      self.physicsClient = p.connect(p.GUI)
    else:
      self.physicsClient = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(self._get_data_path(), physicsClientId=self.physicsClient)
    p.setGravity(0, 0, -GRAVITY, physicsClientId=self.physicsClient)
    p.loadURDF('plane.urdf', physicsClientId=self.physicsClient)
    self._new_object()
    return self._observation()

  def render(self, mode='human'):
    if mode == 'human':
      self.gui = True
      return 
    elif mode == 'depth_array':
      return self._observation()
    else:
      return super(BaseStackEnv, self).render(mode = mode)

  def close(self):
    if self.physicsClient >= 0:
      p.disconnect(self.physicsClient)
      self.physicsClient = -1

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _observation(self):
    return (self._overhead_img, self._object_img)

  def _reward(self):
    current = np.max(self._overhead_img)
    reward = current - self._reward_mem
    self._reward_mem = current
    return reward

  def _done(self):
    return (self._step_count > MAX_EPISODE_STEPS or 
      np.max(self._overhead_img) >= self.max_elevation)
  
  def _new_object(self):
    filename = self.np_random.choice(self._urdf_list)
    self._object_ids.append(p.loadURDF(filename, basePosition=[0, 0, self.spawn_z], 
      physicsClientId=self.physicsClient))
    self._overhead_img = self.overhead_cam(physicsClient=self.physicsClient)
    self._object_img = self.object_cam(physicsClient=self.physicsClient, flip='w')

  def _all_velocities_below(self, threshold):
    for obj in self._object_ids[::-1]:
      # as the last object is the most likely to be moving, iterating the 
      # list backwards assures only one object is checked for most steps
      if np.linalg.norm(np.array(p.getBaseVelocity(obj, physicsClientId=
        self.physicsClient))[0,:]) > threshold:
        return False
    return True

  def _action_to_pose(self, action):
    x = (action[0]+self.object_cam.height/2)*self.resolution - self.size[0]/2
    y = (action[1]+self.object_cam.width/2)*self.resolution - self.size[1]/2
    elevation = self._overhead_img[action[0]:action[0]+self.object_cam.height, 
      action[1]:action[1]+self.object_cam.width] + self._object_img 
    z = np.max(elevation[self._object_img>10**(-3)]) - OBJ_MAX_DEPTH
    print(z)
    return [x, y, z], [0,0,0,1]

  def _get_data_path(self):
    return s.envs.getDataPath()

  def _get_urdf_list( self):
    raise NotImplementedError

class BaseStackEnvFlatAction(BaseStackEnv):
  """
  Inherits BaseStackEnv, changing action format

- action - Linear index of the point on overhead image where to place the new object
(converted to the new object pose):
  Discrete((<overhead image height> - <object image height> + 1)
          *(<overhead image width> - <object image width> + 1))
  """
  def __init__(self, **kwargs):
    super(BaseStackEnvFlatAction, self).__init__(**kwargs)
    self.action_width = self.overhead_cam.width - self.object_cam.width + 1
    self.action_space = spaces.Discrete((self.overhead_cam.height - 
      self.object_cam.height + 1)*self.action_width)

  def _action_to_pose(self, action):
    action = [action//self.action_width, action%self.action_width]
    x = (action[0]+self.object_cam.height/2)*self.resolution - self.size[0]/2
    y = (action[1]+self.object_cam.width/2)*self.resolution - self.size[1]/2
    elevation = self._overhead_img[action[0]:action[0]+self.object_cam.height, 
      action[1]:action[1]+self.object_cam.width] + self._object_img 
    z = np.max(elevation[self._object_img>10**(-3)]) - OBJ_MAX_DEPTH
    return [x, y, z], [0,0,0,1]

