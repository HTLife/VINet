"""
SE3 quaternion tools
"""


import sys
sys.path
sys.path.append('/notebooks/Sophus/py')

from sophus import *
import numpy as np
from sympy import *

from pyquaternion import Quaternion as Qua


## xyz quaternion ==> se(3)
def normalize(ww,wx,wy,wz):# make first number positive
    q = [ww, wx, wy, wz]
    ## Find first negative
    idx = -1
    for i in range(len(q)):
        if q[i] < 0:
            idx = i
            break
        elif q[i] > 0:
            break
    # -1 if should not filp, >=0  flipping index
    if idx >= 0:
        ww = ww * -1
        wx = wx * -1
        wy = wy * -1
        wz = wz * -1
    return ww, wx, wy, wz  
    

def xyzQuaternion2se3_(arr):
    x,y,z,ww,wx,wy,wz = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]
    trans = Matrix([x,y,z])
    ww, wx, wy, wz = normalize(ww, wx, wy, wz)
        
    q_real = ww
    q_img = Matrix([wx, wy, wz])
    q = Quaternion(q_real,q_img)
    R = So3(q)
    
    RT = Se3(R, trans)
    numpy_vec = np.array(RT.log()).astype(float)  # SE3 to se3
    return np.concatenate(numpy_vec)

def xyzQ2se3(arr):
    result = []
    
    if len(arr.shape) == 1:
        return xyzQuaternion2se3_(arr)
    else:
        arr_size = arr.shape[0]
        for i in range(arr_size):
            result.append(xyzQuaternion2se3_(arr[i]))
    return np.array(result)

## SE3 to x y z quaternion
def SE3toXYZQuaternion(matrix):
    trans = np.array([matrix[0,3], matrix[1,3], matrix[2,3]]).astype(float)

    q = np.array(matrix[0:3,0:3]).astype(float)
    
    if not np.allclose(np.dot(q, q.conj().transpose()), np.eye(3)):
        print(q)
    
    q8d = Qua(matrix=q)
    r = np.array([q8d.real]).astype(float)
    v = np.array(q8d.imaginary).astype(float)
    
    q_con = np.concatenate([trans, r, v])
    
    return q_con

def accu(lastxyzQuaternion, new_se3r6):
    """
    lastxyzQuaternion: np.array([x y z ww wx wy wz])
    new_se3r6: np.array([se3R^6])
    """
    new_se3r6 = Matrix(np.transpose(new_se3r6)) #numpy array to sympy matrix
    M_SE3 = Se3.exp(new_se3r6)
    M_SE3 = M_SE3.matrix()
    
    last = xyzQ2se3(lastxyzQuaternion)
    last = Matrix(np.transpose(last))
    M_SE3_last = Se3.exp(last)
    M_SE3_last = M_SE3_last.matrix()
    
    accu = M_SE3_last * M_SE3
    xyzq = SE3toXYZQuaternion(accu)
    
    return xyzq
    
def se3R6toxyzQ(se3r6):
    se3r6 = Matrix(np.transpose(se3r6)) #numpy array to sympy matrix    
    M_SE3 = Se3.exp(se3r6)
    M_SE3 = M_SE3.matrix()
    return SE3toXYZQuaternion(M_SE3)
    
    
    
    