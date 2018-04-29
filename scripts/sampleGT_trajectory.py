#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sampling VICON ground truth camera pose. (absolute pose)

In EuRoC MAV dataset, the vicon motion capture system (Leica MS50) record 
data with 100Hz.  (All pose in vicon seems to be global pose, which is
the pose related to first camera pose.)

Because VINet prediction trajectory  with the frequency equal to image 
frame rate, the "answer" of the training need to be in the same frequency.

My quick workaround is to find the nearest timestamp in vicon/data.csv based 
on the timestamp of cam0/.

"""


from PIL import Image
import os
import sys
import errno
from subprocess import call
import csv



def getMidiumIndex(startTime, endTime, imu_index):
    startIndex = 0
    endIndex = 0
    for i in range(len(imu_index)):
        if imu_index[i] >= startTime:
            startIndex = i
            break
            
    for i in range(len(imu_index)):            
        if imu_index[i] >= endTime:
            endIndex = i
            break
     
    return int((endIndex - startIndex) / 2) + startIndex
        
def getClosestIndex(searchTime, searchStartIndex, timeList):
    foundIdx = 0
    for i in range(searchStartIndex, len(timeList)):
        if timeList[i] >= searchTime:
            foundIdx = i
            break
    
    return foundIdx

def _get_filenames_and_classes(dataset_dir):


    ## Get image list
    fileList = os.listdir(dataset_dir + '/cam0/data')  
    fileList.sort()

    for i in range(len(fileList)):
        fileList[i] = fileList[i][0:-4]
    
    
    ## Get IMU original data
    myTime = []
    timeList = []
    with open(dataset_dir + '/vicon0/data.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            myTime.append(row)
            
    myTime = myTime[1:]
    for i in range(len(myTime)):
        timeList.append( int( myTime[i][0] ) )
    
    
    sampledRow = []
    searchStartIndex = 0
    for i in range(len(fileList)):
        searchTime = int(fileList[i])
        foundIdx = getClosestIndex(searchTime, searchStartIndex, timeList)
        sampledRow.append(myTime[foundIdx])
        
    with open(dataset_dir + '/vicon0/sampled.csv', 'w+') as f:
        for i in range(len(sampledRow)):
            tmpStr = ",".join(sampledRow[i])
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
    
    
