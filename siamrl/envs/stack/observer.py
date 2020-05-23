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
    object_pose=([0.,0.,1.], [0.,0.,0.,1.])
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
    Raises:
      AttributeError: if simulator misses any of the used methods.
    """
    # Check if simulator has necessary attributes
    getattr(simulator,'getCameraImage')
    getattr(simulator,'computeViewMatrix')
    getattr(simulator,'computeProjectionMatrix')
    getattr(simulator,'multiplyTransforms')

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
    oh_view = simulator.computeViewMatrix(
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
    ) 
    oh_projection = simulator.computeProjectionMatrix(
      left=-self._overhead_y/2,
      right=self._overhead_y/2,
      bottom=-self._overhead_x/2,
      top=self._overhead_x/2,
      nearVal=self.far-self._overhead_z,
      farVal=self.far
    )
    self._overhead_cam = lambda: simulator.getCameraImage(
      width=self._overhead_w,
      height=self._overhead_h,
      viewMatrix=oh_view,
      projectionMatrix=oh_projection
    )

    # Object camera
    if hasattr(simulator, 'new_pose'):
      object_position, object_orientation = simulator.new_pose
    else:
      object_position, object_orientation = object_pose
    eye, _ = simulator.multiplyTransforms(
      object_position,
      object_orientation,
      [0,0,-self.far],
      [0,0,0,1]
    )
    up, _ = simulator.multiplyTransforms(
      [0,0,0],
      object_orientation,
      [-1,0,0],
      [0,0,0,1]
    )
    obj_view = simulator.computeViewMatrix(
      cameraEyePosition=eye,
      cameraTargetPosition=object_position,
      cameraUpVector=up
    )
    obj_projection = simulator.computeProjectionMatrix(
      left=-self._object_y/2,
      right=self._object_y/2,
      bottom=-self._object_x/2,
      top=self._object_x/2,
      nearVal=self.far-self._object_z/2,
      farVal=self.far+self._object_z/2
    )
    self._object_cam = lambda: simulator.getCameraImage(
      width=self._object_w,
      height=self._object_h,
      viewMatrix=obj_view,
      projectionMatrix=obj_projection
    )

    self._overhead_map = np.zeros(
      (self._overhead_h, self._overhead_w), 
      dtype='float32'
    )
    self._object_map = np.zeros(
      (self._object_h, self._object_w), 
      dtype='float32')

    # For visualization
    if hasattr(simulator, 'draw_rectangle'):
      self._visualize = lambda: simulator.draw_rectangle(self.size)
    else:
      self._visualize = None

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

  def visualize(self):
    """Visualize the observable space on the simulator."""
    if self._visualize:
      return self._visualize()
