import functools

import numpy as np

class Observer(object):
  far = 10**3

  def __init__(
    self,
    simulator,
    overhead_resolution=192,
    object_resolution=32,
    pixel_size=2.**(-8),
    max_z=1,
    object_pose=None,
    orientation_freedom=0,
  ):
    """
    Args:
      simulator: instance of the simulator to be observed. Must support the 
        pybullet interface. Used methods: 
          computeViewMatrix,
          computeProjectionMatrix, 
          getCameraImage, 
          multiplyTransforms.
      overhead_resolution: size of the overhead image, in pixels. Either a
        scalar for square image, or a list with [height, width]. 
      object_resolution: size of the object image, in pixels. Either a
        scalar for square image, or a list with [height, width]. 
      pixel_size: real size of a pixel in simulator units. Either a scalar 
        for square pixels, or a list with [height, width].
      max_z: maximum observable z coordinate.
      object_pose: tuple with position and orientation of the object to be
        observed. Ignored if simulator has attribute 'new_pose'.
      orientation_freedom: integer exponent for the number of orientaions of
        the object to be captured (2^n).
    Raises:
      AttributeError: if simulator misses any of the used methods.
    """
    # Check if simulator has necessary attributes
    if (
      hasattr(simulator,'getCameraImage') and
      hasattr(simulator,'computeViewMatrix') and
      hasattr(simulator,'computeProjectionMatrix') and
      hasattr(simulator,'multiplyTransforms')
    ):
      self._sim = simulator
    else:
      raise TypeError("simulator must provide the pybullet API")

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
    self._overhead_view = simulator.computeViewMatrix(
      cameraEyePosition=(
          self._overhead_x/2, 
          self._overhead_y/2, 
          self.far
      ),
      cameraTargetPosition=(
          self._overhead_x/2, 
          self._overhead_y/2, 
          0
      ),
      cameraUpVector=(-1,0,0),
    ) 
    self._overhead_projection = simulator.computeProjectionMatrix(
      left=-self._overhead_y/2,
      right=self._overhead_y/2,
      bottom=-self._overhead_x/2,
      top=self._overhead_x/2,
      nearVal=self.far-self._overhead_z,
      farVal=self.far
    )

    self._overhead_map = np.zeros(
      (self._overhead_h, self._overhead_w), 
      dtype='float32'
    )

    # Object camera
    self._object_projection = simulator.computeProjectionMatrix(
        left=-self._object_y/2,
        right=self._object_y/2,
        bottom=-self._object_x/2,
        top=self._object_x/2,
        nearVal=self.far-self._object_z/2,
        farVal=self.far+self._object_z/2
      )

    if object_pose is None:
      if hasattr(simulator, 'new_pose'):
        if not isinstance(simulator.new_pose, list):
          object_pose = simulator.new_pose
      else:
        raise ValueError("If object_pose is not provided, simulator must have 'new_pose' attribute.")

    n_orientations = 2**orientation_freedom
    up_vectors = []
    for i in range(n_orientations):
      quaternion = simulator.getQuaternionFromEuler([0,0,i*2*np.pi/n_orientations])

      up,_ = simulator.multiplyTransforms(
        (0,0,0),
        quaternion,
        (-1,0,0),
        (0,0,0,1),
      )
      # orientation of the object relative to the view
      _,orientation = simulator.invertTransform((0,0,0), quaternion)
      up_vectors.append((up, orientation))

    if object_pose is not None and len(up_vectors) == 1:
      self._up_vectors = None
      self._object_orientations = None
      self._object_indexes = None

      eye, _ = simulator.multiplyTransforms(
        object_pose[0],
        object_pose[1],
        (0,0,-self.far),
        (0,0,0,1)
      )
      up,_ = simulator.multiplyTransforms(
        (0,0,0),
        object_pose[1],
        up_vectors[0][0],
        (0,0,0,1),
      )
      self._object_view = simulator.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=object_pose[0],
        cameraUpVector=up
      )
    elif object_pose is not None and len(up_vectors) > 1:
      self._up_vectors = up_vectors
      self._object_orientations = []
      self._object_indexes = None

      eye, _ = simulator.multiplyTransforms(
        object_pose[0],
        object_pose[1],
        (0,0,-self.far),
        (0,0,0,1)
      )

      def obj_view(up):
        up,_ = simulator.multiplyTransforms(
          (0,0,0),
          object_pose[1],
          up,
          (0,0,0,1),
        )
        return simulator.computeViewMatrix(
          cameraEyePosition=eye,
          cameraTargetPosition=object_pose[0],
          cameraUpVector=up
        )
      self._object_view = obj_view
    elif object_pose is None and len(up_vectors) == 1:
      self._up_vectors = None
      self._object_orientations = None
      self._object_indexes = []

      def obj_view(pose):
        eye, _ = simulator.multiplyTransforms(
          pose[0],
          pose[1],
          (0,0,-self.far),
          (0,0,0,1)
        )
        up,_ = simulator.multiplyTransforms(
          (0,0,0),
          pose[1],
          up_vectors[0][0],
          (0,0,0,1),
        )
        return simulator.computeViewMatrix(
          cameraEyePosition=eye,
          cameraTargetPosition=pose[0],
          cameraUpVector=up
        )
      
      self._object_view = obj_view
    elif object_pose is None and len(up_vectors) > 1:
      self._up_vectors = up_vectors
      self._object_orientations = []
      self._object_indexes = []

      def obj_view(up, pose):
        eye, _ = simulator.multiplyTransforms(
          pose[0],
          pose[1],
          (0,0,-self.far),
          (0,0,0,1)
        )
        up,_ = simulator.multiplyTransforms(
          (0,0,0),
          pose[1],
          up,
          (0,0,0,1),
        )
        return simulator.computeViewMatrix(
          cameraEyePosition=eye,
          cameraTargetPosition=pose[0],
          cameraUpVector=up
        )
    
      self._object_view = obj_view

    if callable(self._object_view):
      self._object_map = []
    else:
      self._object_map = np.zeros(
        (self._object_h, self._object_w), 
        dtype='float32'
      )

  def __call__(self):
    """Get new depth images and convert to elevation maps"""
    # Get overhead camera depth image
    _,_,_,d,_ = self._sim.getCameraImage(
      width=self._overhead_w,
      height=self._overhead_h,
      viewMatrix=self._overhead_view,
      projectionMatrix=self._overhead_projection,
    )
    # Convert to elevation
    self._overhead_map = self.far - \
      self.far*(self.far-self._overhead_z)/(self.far-self._overhead_z*d)

    if self._sim.has_new_object or not (
      hasattr(self, '_last_new_poses') and self._last_new_poses  # pylint: disable=access-member-before-definition
    ):
      if not callable(self._object_view):
        # Get object camera depth image
        _,_,_,d,_ = self._sim.getCameraImage(
          width=self._object_w,
          height=self._object_h,
          viewMatrix=self._object_view,
          projectionMatrix=self._object_projection,
        )
        # Convert to elevation
        d = self.far+self._object_z/2 - \
          (self.far**2-(self._object_z/2)**2)/(self.far+self._object_z*(1/2-d))
        # Flip so that the axis directions are the same as in the overhead map
        self._object_map = d[:,::-1]
      elif self._object_orientations is not None and self._object_indexes is None:
        self._object_map = []
        self._object_orientations = []

        for up, orientation in self._up_vectors:
          _,_,_,d,_ = self._sim.getCameraImage(
            width=self._object_w,
            height=self._object_h,
            viewMatrix=self._object_view(up),
            projectionMatrix=self._object_projection,
          )

          d = self.far+self._object_z/2 - \
            (self.far**2-(self._object_z/2)**2)/(self.far+self._object_z*(1/2-d))
          self._object_map.append(d[:,::-1])
          self._object_orientations.append(orientation)
      elif self._object_orientations is None and self._object_indexes is not None:
        self._object_map = []
        self._object_indexes = []

        self._last_new_poses = self._sim.new_pose
        for i, pose in enumerate(self._last_new_poses):
          _,_,_,d,_ = self._sim.getCameraImage(
            width=self._object_w,
            height=self._object_h,
            viewMatrix=self._object_view(pose),
            projectionMatrix=self._object_projection,
          )
          d = self.far+self._object_z/2 - \
            (self.far**2-(self._object_z/2)**2)/(self.far+self._object_z*(1/2-d))
          self._object_map.append(d[:,::-1])
          self._object_indexes.append(i)
      elif self._object_orientations is not None and self._object_indexes is not None:
        self._object_map = []
        self._object_orientations = []
        self._object_indexes = []

        self._last_new_poses = self._sim.new_pose
        for i, pose in enumerate(self._last_new_poses):
          for up, orientation in self._up_vectors:
            _,_,_,d,_ = self._sim.getCameraImage(
              width=self._object_w,
              height=self._object_h,
              viewMatrix=self._object_view(up, pose),
              projectionMatrix=self._object_projection,
            )
            d = self.far+self._object_z/2 - \
              (self.far**2-(self._object_z/2)**2)/(self.far+self._object_z*(1/2-d))
            self._object_map.append(d[:,::-1])
            self._object_orientations.append(orientation)
            self._object_indexes.append(i)
    else:
      # If no objects were added, don't recollect all observations.
      new_poses = self._sim.new_pose
      new_indexes = [None]*len(self._last_new_poses)
      to_remove = []
      for i, pose in enumerate(self._last_new_poses):
        if pose not in new_poses:
          for j in range(len(self._object_map)):
            if self._object_indexes[j] == i:
              to_remove.append(j)
        else:
          new_indexes[i] = new_poses.index(pose)
      # Remove the observations of used objects.
      while to_remove:
        i = to_remove.pop()
        self._object_map.pop(i)
        self._object_indexes.pop(i)
        if self._object_orientations is not None:
          self._object_orientations.pop(i)
      # Update indexes
      for i, old in enumerate(self._object_indexes):
        self._object_indexes[i] = new_indexes[old]

      self._last_new_poses = new_poses

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
  def num_objects(self):
    """Number of observed objects"""
    if isinstance(self._object_map, list):
      return len(self._object_map)
    else:
      return int(np.any(self._object_map))

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

  def pose(self, pixel, index=None):
    """Returns the position of the object for which the object's map is 
      overlapped with the overhead map with an offset given by pixel. If
      there is more than one object observation, also returns the 
      orientation and/or the object index corresponding to the given 
      observation index."""
    x,y = self.pixel_to_xy(pixel)

    if index is None:
      object_map = self._object_map
    else:
      object_map = self._object_map[index]

    z = self._overhead_map[
      pixel[0]:pixel[0]+self._object_h, 
      pixel[1]:pixel[1]+self._object_w
    ] + object_map 
    z = np.max(z[object_map>10**(-4)])
    # Correction for the object position inside the object map frame
    x += self._object_x/2
    y += self._object_y/2
    z -= self._object_z/2

    ret = {'position': (x,y,z)}
    if index is not None:
      if self._object_orientations:
        ret['orientation'] = self._object_orientations[index]
      if self._object_indexes:
        ret['index'] = self._object_indexes[index]
    return ret

  def visualize(self, **kwargs):
    """Visualize the observable space on the simulator."""
    if hasattr(self._sim, 'draw_rectangle'):
      self._sim.draw_rectangle(self.size, **kwargs)

