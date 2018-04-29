#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Convert sampled VICON ground truth(/vicon0/sampled.csv) to relative pose

The format is in x y z ww wx wy wz

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

import numpy as np
import quaternion #pip install numpy numpy-quaternion numba

import decimal

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

def _get_filenames_and_classes(dataset_dir):
    
    
    trajectory_abs = []  #abosolute camera pose
    with open(dataset_dir + '/vicon0/sampled.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
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
        trajectory_relative.append(relative_pose)
        
    with open(dataset_dir + '/vicon0/sampled_relative.csv', 'w+') as f:
        tmpStr = ",".join(trajectory_abs[0])
        f.write(tmpStr + '\n')        
        
        for i in range(len(trajectory_relative)):
            #tmpStr = ",".join(np.array(trajectory_relative[i]).astype(str))
            tmpStr = trajectory_relative[i][0] + ',' +\
                     float_to_str(trajectory_relative[i][1]) + ',' +\
                     float_to_str(trajectory_relative[i][2]) + ',' +\
                     float_to_str(trajectory_relative[i][3]) + ',' +\
                     float_to_str(trajectory_relative[i][4]) + ',' +\
                     float_to_str(trajectory_relative[i][5]) + ',' +\
                     float_to_str(trajectory_relative[i][6]) + ',' +\
                     float_to_str(trajectory_relative[i][7])
            f.write(tmpStr + '\n')        
            
            
    return
                

def main():
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_01_easy')
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_02_easy')
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_03_medium')
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_04_difficult')
    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/MH_05_difficult')

    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V1_01_easy')
    _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V1_02_medium')
    _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V1_03_difficult')
    _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V2_01_easy')
    _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V2_02_medium')
    _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V2_03_difficult')
       
 

    
    
if __name__ == "__main__":
    main()
    
    
