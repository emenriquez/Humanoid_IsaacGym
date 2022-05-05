import os
import json

from envs.utils.interpolateMotion import HumanoidPoseInterpolator
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import math


test = HumanoidPoseInterpolator('humanoid3d_run.txt')

# Test looping through frames
step = 0
epsilon = 1E-4
while(step <= 8.0):
    pose = test.slerpFromTime(step)
    basePos1 = pose[26:29]
    baseOrn1 = pose[29:]
    # Print the frame data here
    print(f'step: {step}\t pos: {basePos1}\t orn: {baseOrn1}')
    #step
    step += 1/4

print(step)

