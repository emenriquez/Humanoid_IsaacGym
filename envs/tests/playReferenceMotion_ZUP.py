# imports
import math
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
from time import sleep
import os, sys
import json
from math import floor



# Add to path so that we can import from utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.interpolateMotion import HumanoidPoseInterpolator

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

file_name = "mjcf/newHumanoid_minitest.xml"


# parse arguments
args = gymutil.parse_arguments(
    description="Reference Motion Player: Test and view mocap file motions",
    custom_parameters=[
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "../../assets"
asset_file = file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
    # set DOF position to default
    dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

# Print DOF properties
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# set up the env grid
num_envs = 36
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(17.2, 16, 2)
cam_target = gymapi.Vec3(5, 13, -2.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

# Load motion file and initialize interpolator
interpolator = HumanoidPoseInterpolator('../mocapData/humanoid3d_run.txt')
motionFile = '../mocapData/humanoid3d_run.txt'
motionFile = os.path.join(os.path.dirname(__file__), motionFile)

with open(motionFile,'r') as motion_file:
    mocap_data = json.load(motion_file)

total_frames = len(mocap_data['Frames'])
epsilon = 1E-4


print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.32)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


# acquire root state
_actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(_actor_root_state)
root_positions = actor_root_state[:, 0:3]
root_orientations = actor_root_state[:, 3:7]
root_linvels = actor_root_state[:, 7:10]
root_angvels = actor_root_state[:, 10:13]

gym.simulate(sim)
gym.refresh_actor_root_state_tensor(sim)
original_state = actor_root_state.clone()



# Main motion replay loop
currTime = 0

while not gym.query_viewer_has_closed(viewer):
  
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    

    '''
    Joint DOF indices should correspond to the following:
        0, 1, 2 -       neck (xyz)           
        3, 4, 5 -       abdomen **(zyx)** # Note order    
        6, 7, 8 -       right_hip (xyz)  
        9 -             right_knee 
        10, 11 -        right_ankle (xy)
        12, 13, 14 -    left_hip (xyz)    
        15 -            left_knee 
        16, 17 -        left_ankle (xy)
        18, 19, 20 -    right_shoulder (xyz)
        21 -            right_elbow
        22, 23, 24 -    left_shoulder (xyz)
        25 -            left_elbow 
        **Note: This should match the return from pose interpolator using Slerp() method
        ***Slerp() additionally returns base position and orientation (indices: 26-32)
    '''
    # Fetch Frame
    frameTime = mocap_data['Frames'][0][0]+epsilon
    currentFrame = mocap_data['Frames'][floor(currTime/frameTime) % total_frames] 
    nextFrame = mocap_data['Frames'][floor(currTime/frameTime+1) % total_frames] 
    fraction = (currTime / frameTime) % 1
    frame_pose = interpolator.Slerp(currentFrame, nextFrame, fraction)

    # Set position and orientation
    pos_offset = torch.tensor([0, 1, 0])
    orn_offset = torch.tensor(frame_pose[29:])

    # for i in range(num_envs):
        # root_positions[i] += pos_offset
        # root_orientations[i] += orn_offset
        # root_linvels[i] += torch.tensor([0, 0, 0])
        # root_angvels[i] += torch.tensor([0, 0, 0])

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)

    # Set joint positions
    for i in range(num_dofs):
        dof_positions[i] = frame_pose[i]
    
    currTime += dt



    if args.show_axis:
        gym.clear_lines(viewer)

    # Set the actor position on each frame
    # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(original_state))
    # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(actor_root_state))
    # gym.refresh_actor_root_state_tensor(sim)


    # clone actor state in all of the environments
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

        if args.show_axis:
            # get the DOF frame (origin and axis)
            dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], current_dof)
            frame = gym.get_dof_frame(envs[i], dof_handle)

            # draw a line from DOF origin along the DOF axis
            p1 = frame.origin
            p2 = frame.origin + frame.axis * 0.7
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

















    '''
    TESTS
    '''
    # actor_root_state = gym.acquire_actor_root_state_tensor(sim) # JUST RETURNS ZEROES!!! FIX THIS <-------
    # print(gymtorch.wrap_tensor(actor_root_state))
    # break















    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)