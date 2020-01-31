import numpy as np
import pybullet as p

class Camera(object):
  def __init__(self, client, targetPos = [0,0,0], cameraPos = [0,0,10**3], upVector = 
    [-1,0,0], width = 1., height = 1., depthRange = [-1., 0], resolution = 2.**(-9)):

    self.bc = client
    self.width = int(width/resolution)
    self.height = int(height/resolution)
    d = np.linalg.norm(np.array(targetPos)-np.array(cameraPos))
    self.n = d+depthRange[0]
    self.f = d+depthRange[1]
    self.viewMatrix = p.computeViewMatrix(cameraPos,targetPos,upVector)
    self.projectionMatrix = p.computeProjectionMatrix(-width*self.n/(2*self.f), 
      width*self.n/(2*self.f), -height*self.n/(2*self.f), height*self.n/(2*self.f), self.n, self.f)

  def getImage(self):
    return self.bc.getCameraImage(self.width, self.height, viewMatrix = self.viewMatrix, 
      projectionMatrix = self.projectionMatrix)
  
  def setClient(self, client):
    self.bc = client

class ElevationCamera(Camera):
  def __init__(self, **kwargs):
    super(ElevationCamera, self).__init__(**kwargs)
    self.channels = 1

  def __call__(self, flip = None):
    _,_,_,d,_ = self.getImage()
    d = self.f - self.f*self.n/(self.f - (self.f-self.n)*d)
    if flip is not None:
      if flip == 'w' or flip == 'width':
        d = d[:,::-1]
      if flip == 'h' or flip == 'height':
        d = d[::-1,:]
    return d[:,:,np.newaxis]
