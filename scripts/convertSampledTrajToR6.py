#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Convert sampled VICON ground truth(/vicon0/sampled.csv) to relative pose

The format is in se3 R6 = (x,x,x,x,x,x)

Relative pose transition is simply calculated by:
    previous position P = (x, y, z)  
    current position P'=(x', y', z')
    relative pose transition R = (x' - x, y'-y, z'-z)

Relative rotation is a little bit complicated:
    'Difference' between two quaternions
    Q = Q2*Q1^{-1}.
    (https://stackoverflow.com/questions/1755631/difference-between-two-quaternions)

"""


from PIL import Image
import os
import sys
import errno
from subprocess import call
import csv



import decimal


from sophus import *
import quaternion #pip install numpy numpy-quaternion numba
import numpy as np
from sympy import *

from pyquaternion import Quaternion as Qua

# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 15

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')
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
    #print(RT.log())
    numpy_vec = np.array(RT.log()).astype(float)  # SE3 to se3
    
    return np.concatenate(numpy_vec)

def _get_filenames_and_classes(dataset_dir):
    
    
    trajectory_abs = []  #abosolute camera pose
    
    with open(dataset_dir + '/vicon0/sampled_relative.csv') as csvfile:
        count = 0
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if count == 0:
                count = 1
                continue
            trajectory_abs.append(row)
            
    print('Total data: '+ str(len(trajectory_abs)))
    
    
    # Calculate relative pose
    trajectory_relative = []
    for i in range(len(trajectory_abs)-1):
        #timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []
        timestamp = trajectory_abs[i+1][0]
        X, Y, Z = np.array(trajectory_abs[i+1][1:4]).astype(float) - np.array(trajectory_abs[i][1:4]).astype(float)
        
        ww0,wx0,wy0,wz0 = np.array(trajectory_abs[i][4:]).astype(float)
        ww1,wx1,wy1,wz1 = np.array(trajectory_abs[i+1][4:]).astype(float)
        q0 = np.quaternion(ww0,wx0,wy0,wz0)
        q1 = np.quaternion(ww1,wx1,wy1,wz1)
        relative_rot = quaternion.as_float_array(q1 * q0.inverse())
        
        
        relative_pose = [timestamp, X, Y, Z, relative_rot[0], relative_rot[1], relative_rot[2], relative_rot[3]]
        se3R6 = xyzQuaternion2se3_([relative_pose[1],\
                                  relative_pose[2],\
                                  relative_pose[3],\
                                  relative_pose[4],\
                                  relative_pose[5],\
                                  relative_pose[6],\
                                  relative_pose[7]])
        trajectory_relative.append(se3R6)
        #print(i)
        
    

    with open(dataset_dir + '/vicon0/sampled_relative_R6.csv', 'w+') as f:
        tmpStr = ",".join(trajectory_abs[0])
        f.write(tmpStr + '\n')        
        
        for i in range(len(trajectory_relative)):
            r1 = float_to_str(trajectory_relative[i][0])
            r2 = float_to_str(trajectory_relative[i][1])
            r3 = float_to_str(trajectory_relative[i][2])
            r4 = float_to_str(trajectory_relative[i][3])
            r5 = float_to_str(trajectory_relative[i][4])
            r6 = float_to_str(trajectory_relative[i][5])
            tmpStr = str(trajectory_relative[i][0]) + ',' + r1 + ',' + r2 + ',' + r3 + ',' + r4 + ',' + r5 + ',' + r6
            f.write(tmpStr + '\n')        

    return
                

def main():
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_01_easy')
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_02_easy')
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_03_medium')
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_04_difficult')
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_05_difficult')

    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V1_01_easy')
   # _get_filenames_and_classes('/notebooks/EuRoC_modify/V1_02_medium')
#     _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V1_03_difficult')
     _get_filenames_and_classes('/notebooks/EuRoC_modify/V2_01_easy')
#     _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V2_02_medium')
#     _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V2_03_difficult')

if __name__ == "__main__":
    main()
    
    
