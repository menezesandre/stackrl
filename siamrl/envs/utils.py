import numpy as np
import pybullet as p

class Camera(object):
    def __init__(self, targetPos = [0,0,0], cameraPos = [0,0,10**6], upVector = [-1,0,0], width = 1., height = 1., depthRange = [-1., 0], resolution = 2.**(-9)):
        self.width = int(width/resolution)
        self.height = int(height/resolution)
        self.range = depthRange[1] - depthRange[0]

        d = np.linalg.norm(np.array(targetPos)-np.array(cameraPos))
        d = [d+depthRange[0], d+depthRange[1]]
        self.viewMatrix = p.computeViewMatrix(cameraPos,targetPos,upVector)
        self.projectionMatrix = p.computeProjectionMatrix(-width*d[0]/(2*d[1]), width*d[0]/(2*d[1]), -height*d[0]/(2*d[1]), height*d[0]/(2*d[1]), d[0], d[1])

    def getImage(self):
        return p.getCameraImage(self.width, self.height, viewMatrix = self.viewMatrix, projectionMatrix = self.projectionMatrix)

class ElevationCamera(Camera):
    def __init__(self, **kwargs):
        super(ElevationCamera, self).__init__(**kwargs)

    def __call__(self, flip = None):
        _,_,_,d,_ = self.getImage()
        d = (1.-d)*self.range
        if flip is not None:
            if flip == 'w' or flip == 'width':
                d = d[:,::-1]
            if flip == 'h' or flip == 'height':
                d = d[::-1,:]
        return d
