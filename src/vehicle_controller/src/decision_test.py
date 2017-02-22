#!/usr/bin/env python
# license removed for brevity

import rospy
import numpy
import math
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# OVERALL DRIVE PARAMETERS
# CONFIGURE THIS TO MODULATE THROTTLE/BRAKING CHARACTERISTICS
gentle_throttle = 0.1  # ~ smooth slowdown
cruise_throttle = 0.2  # amount of throttle needed to maintain constant speed
accel_throttle = 0.5  # ~ for gradual acceleration

gentle_brake = 0.1  # amount of brake needed to gradual slowdown
normal_brake = 0.2  # ~ for normal stop
emergency_brake = 1  # ~ for emergency stop

# VEHICLE PHYSICAL PARAMETERS
# CONFIGURE THIS FOR CHANGES TO WHEELBASE, HEIGHT, TIRES, ETC.
wheelbase = 2


def decision_test():
    print("decision_test node started!")
    rospy.loginfo("decision_test node started!")

    # initialising my node
    rospy.init_node('decision_test', anonymous=True)

    # setting up publications
    lane_curvature = rospy.Publisher('lane_curvature', Float32, queue_size=10)
    lane_direction = rospy.Publisher('lane_direction', String, queue_size=10)
    # manual_throttle =  rospy.Publisher('manual_throttle', Float32, queue_size=10)
    # auto_toggle = rospy.Publisher('auto_toggle', Bool, queue_size=10)
    # actual_velocity = rospy.Publisher('actual_velocity', Float32, queue_size=10)

    # refresh rate of pubs/subs
    rate = rospy.Rate(10)  # 10hz

    # prevents rospy from stopping unless node is closed
    # rospy.spin()

    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        # pub.publish(hello_str)
        lane_crv = 2
        lane_dir = 'Right'
        # mnl_thrtl = 20.0
        # at_tgl = True
        # actl_vel = 20.0

        lane_curvature.publish(lane_crv)
        lane_direction.publish(lane_dir)

    rate.sleep()

if __name__ == '__main__':
    try:
        decision_test()
    except rospy.ROSInterruptException:
        pass
