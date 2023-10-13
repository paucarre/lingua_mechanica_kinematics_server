#!/usr/bin/env python
import roslib
import rospy

from lingua_mechanica_kinematics_msgs.srv import GetPositionIK
from lingua_mechanica_kinematics_msgs.msg import RobotStates

def solve_ik(request: GetPositionIK):
    print("Returning [%s + %s = %s]" % (req.a, req.b, (req.a + req.b)))
    return RobotStates(req.a + req.b)

def get_position_ik_server():
    rospy.init_node('solve_ik')
    service = rospy.Service('solve_ik', GetPositionIK, solve_ik)
    rospy.spin()

if __name__ == "__main__":
    print("Running stuff")
    get_position_ik_server()