#!/usr/bin/env python
import roslib
import rospy
from linguamechanica.inference import setup_inference

from lingua_mechanica_kinematics_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from sensor_msgs.msg import JointState, MultiDOFJointState
from pytransform3d.transformations import transform_from_pq, exponential_coordinates_from_transform
import torch
import numpy as np

def solve_ik(positon_ik: GetPositionIK):
    pose = positon_ik.ik_request.pose_stamped.pose
    pq = np.array([pose.position.x, pose.position.y, pose.position.z, 
         pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    transformation = transform_from_pq(pq)
    pose = exponential_coordinates_from_transform(transformation)
    pose = torch.from_numpy(pose)
    # NOTE: in lingua mechanica, poses are inverted compared to pytransform3d
    pose = torch.cat([pose[3:], pose[:3]])
    environment, agent, state, initial_reward = setup_inference("/home/ros/linguamechanica/urdf/cr5.urdf", 806000, 3200, None, target_pose=pose)
    thetas_sorted, reward_sorted = agent.inference(
        100, state, environment, 1
    )
    print(thetas_sorted, reward_sorted)
    robot_states = []
    for idx in range(thetas_sorted.shape[0]):
        joint_state = JointState()
        joint_state.header = positon_ik.ik_request.robot_state.joint_state.header
        joint_state.name = positon_ik.ik_request.robot_state.joint_state.name    
        joint_state.position  = thetas_sorted[idx, :].tolist()
        joint_state.effort   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        joint_state.velocity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        robot_state = RobotState()
        robot_state.joint_state = joint_state
        robot_states.append(robot_state)
    success = MoveItErrorCodes()
    success.val = 1    
    return robot_states, success

def get_position_ik_server():
    rospy.init_node('solve_ik')
    service = rospy.Service('solve_ik', GetPositionIK, solve_ik)
    rospy.spin()

if __name__ == "__main__":
    get_position_ik_server()
    
