import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import numpy as np

import siamrl as s
from siamrl.envs.utils import ElevationCamera

MIN_DIV = 2.**5
NEW_OBJ_POS = [0,0,2]
GRAVITY = 9.8
NUM_URDF_FILES = 1
OBJ_MAX_DEPTH = 0.25
OBJ_MAX_SIDE = 2**(-3)
MAX_NUM_OBJ = 1000
VELOCITY_THRESHOLD = 0.01


class RocksEnv(gym.Env):
  metadata = {'render.modes': ['human', 'depth_array'],
              'reward.modes': ['max_elevation', 'diff_max_elevation', 'max_min_elevation', 'diff_max_min_elevation'],
              'reward.modes.max': ['max_elevation', 'diff_max_elevation', 'max_min_elevation', 'diff_max_min_elevation'],
              'reward.modes.min': ['max_min_elevation', 'diff_max_min_elevation'],
              'reward.modes.diff': ['diff_max_elevation', 'diff_max_min_elevation']}

  def __init__(self, size=[.5, .5], resolution = 2.**(-9), gui=False):
    self.size = np.round(np.array(size)/(resolution*MIN_DIV))*resolution*MIN_DIV #assert dimensions are divisivle by MIN_DIV
    self.resolution = resolution
    self.overhead_cam = ElevationCamera(width=self.size[1], height=self.size[0], resolution=resolution)
    self.object_cam = ElevationCamera(targetPos=NEW_OBJ_POS, cameraPos=[0,0,-10**6+2], width=OBJ_MAX_SIDE, height=OBJ_MAX_SIDE,
                                      depthRange=[-OBJ_MAX_DEPTH,OBJ_MAX_DEPTH], resolution=resolution)
    self.gui = gui

    self._overhead_img = np.array([])
    self._object_img = np.array([])

    self.physicsClient = -1

    self.seed()

    self.action_space = spaces.MultiDiscrete(
      [self.overhead_cam.height - self.object_cam.height + 1, 
       self.overhead_cam.width - self.object_cam.width + 1])
    self.observation_space = spaces.Tuple(
      (spaces.Box(low=0.0, high=1.0, shape=(self.overhead_cam.height, self.overhead_cam.width), dtype=np.float32),
       spaces.Box(low=0.0, high=1.0, shape=(self.object_cam.height, self.object_cam.width), dtype=np.float32)))

  def step(self, action):
    x = (action[0]+self.object_cam.height/2)*self.resolution - self.size[0]/2
    y = (action[1]+self.object_cam.width/2)*self.resolution - self.size[1]/2
    elevation = self._overhead_img[action[0]:action[0]+self.object_cam.height,
                                   action[1]:action[1]+self.object_cam.width] + self._object_img 
    z = np.max(elevation[self._object_img>0.]) - OBJ_MAX_DEPTH
    p.resetBasePositionAndOrientation(self._object_ids[-1], [x,y,z], [0,0,0,1])

    p.stepSimulation()
    while self._max_velocity() >= VELOCITY_THRESHOLD:
      p.stepSimulation()
    self._new_object()

    return self.observation(), self.reward(), self.done(), {}

  def reset(self):
    if self.physicsClient >= 0:
      p.disconnect(self.physicsClient)

    self._reward = 0.
    self._object_ids = []
    self._object_count = 0

    if self.render:
      self.physicsClient = p.connect(p.GUI)
    else:
      self.physicsClient = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(s.envs.getDataPath())
    p.setGravity(0, 0, -GRAVITY)
    p.loadURDF('plane.urdf')
    self._new_object()
    return (self._overhead_img, self._object_img)

  def render(self, mode='human'):
    if mode == 'human':
      self.gui = True
      return 
    elif mode == 'depth_array':
      return self.observation()
    else:
      return super(RocksEnv, self).render(mode = mode)

  def close(self):
    if self.physicsClient >= 0:
      p.disconnect(self.physicsClient)
      self.physicsClient = -1

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def observation(self):
    return (self._overhead_img, self._object_img)

  def reward(self, mode='diff_max_elevation', discount = 0.):
    assert mode in self.metadata['reward.modes']
    current_reward = 0.
    if mode in self.metadata['reward.modes.max']:
      current_reward += np.max(self._overhead_img)
    if mode in self.metadata['reward.modes.min']:
      current_reward += np.min(self._overhead_img)
    ret = current_reward - discount
    if mode in self.metadata['reward.modes.diff']:
      ret -= self._reward
    self._reward = current_reward
    return ret

    self._reward = current_reward
    return current_reward - discount

  def done(self):
    return self._object_count > MAX_NUM_OBJ or np.max(self._overhead_img) >= 1.0
  
  def _new_object(self):
    filename = "rocks/rock{}.urdf".format(self.np_random.randint(NUM_URDF_FILES))
    self._object_ids.append(p.loadURDF(filename, basePosition = NEW_OBJ_POS))
    self._object_count += 1
    self._overhead_img = self.overhead_cam()
    self._object_img = self.object_cam(flip='w')

  def _max_velocity(self):
    v = 0.
    for obj in self._object_ids:
      v = max(v, np.linalg.norm(np.array(p.getBaseVelocity(obj))[0,:]))
    return v
