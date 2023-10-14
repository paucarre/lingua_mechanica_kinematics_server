#!/usr/bin/env python
import roslib
import rospy

from lingua_mechanica_kinematics_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from sensor_msgs.msg import JointState, MultiDOFJointState
from pytransform3d.transformations import transform_from_pq, exponential_coordinates_from_transform
import torch
import numpy as np
'''
ik_request: 
  group_name: "cr5_arm"
  robot_state: 
    joint_state: 
      header: 
        seq: 0
        stamp: 
          secs: 0
          nsecs:         0
        frame_id: "dummy_link"
      name: 
        - joint1
        - joint2
        - joint3
        - joint4
        - joint5
        - joint6
      position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      velocity: []
      effort: []
    multi_dof_joint_state: 
      header: 
        seq: 0
        stamp: 
          secs: 0
          nsecs:         0
        frame_id: "dummy_link"
      joint_names: []
      transforms: []
      twist: []
      wrench: []
    attached_collision_objects: []
    is_diff: False
  constraints: 
    name: ''
    joint_constraints: []
    position_constraints: []
    orientation_constraints: []
    visibility_constraints: []
  avoid_collisions: True
  ik_link_name: "Link6"
  pose_stamped: 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "base_link"
    pose: 
      position: 
        x: 1.3270787349028978e-06
        y: -0.2460000365972519
        z: 0.8392711877822876
      orientation: 
        x: 1.298674078498152e-06
        y: -0.7071067690849304
        z: 0.7071067690849304
        w: 3.896022462868132e-06
  ik_link_names: []
  pose_stamped_vector: []
  timeout: 
    secs: 0
    nsecs:         0
'''

def solve_ik(positon_ik: GetPositionIK):
    # positon_ik.robot_state
    print(positon_ik)
    pose = positon_ik.ik_request.pose_stamped.pose
    pq = np.array([pose.position.x, pose.position.y, pose.position.z, 
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    transformation = transform_from_pq(pq)
    pose = exponential_coordinates_from_transform(transformation)
    pose = torch.from_numpy(pose)
    # NOTE: in lingua mechanica, poses are inverted compared to pytransform3d
    pose = torch.cat([pose[3:], pose[:3]]).unsqueeze(0)
    joint_state = JointState()
    joint_state.header = positon_ik.ik_request.robot_state.joint_state.header
    joint_state.name = positon_ik.ik_request.robot_state.joint_state.name    
    joint_state.position  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]    
    joint_state.effort   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    joint_state.velocity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    robot_state = RobotState()
    robot_state.joint_state = joint_state
    success = MoveItErrorCodes()
    success.val = 1
    return [robot_state], success

def get_position_ik_server():
    rospy.init_node('solve_ik')
    service = rospy.Service('solve_ik', GetPositionIK, solve_ik)
    rospy.spin()

if __name__ == "__main__":
    get_position_ik_server()