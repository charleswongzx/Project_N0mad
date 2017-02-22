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

# GLOBAL VARS
lane_crv = 100
usound_apx = None
lane_dir = None


# callback functions, necessary for all subscribers
def callback1(data):
    # rospy.loginfo(rospy.get_caller_id() + "lane_curvature: %s", data.data)
    global lane_crv
    lane_crv = data.data


def callback2(data):
    rospy.loginfo(rospy.get_caller_id() + "usound_approx: %s", data.data)
    global usound_apx
    usound_apx = data.data

def callback3(data):
    # rospy.loginfo(rospy.get_caller_id() + "lane_direction: %s", data.data)
    global lane_dir
    lane_dir = data.data


def angleToVel(angl_str):
    if 0 < angl_str < 0.3:
        velocity = 30
    elif 0.3 < angl_str < 0.5:
        velocity = 20
    elif 0.5 < angl_str < 0.7:
        velocity = 10
    else:
        velocity = 0
    return velocity


def decision():
    print("decision node started!")
    rospy.loginfo("decision node started!")

    # initialising my node
    rospy.init_node('decision', anonymous=True)

    # setting up subscriptions
    lane_curvature = rospy.Subscriber('lane_curvature', Float32, callback1)
    usound_approx = rospy.Subscriber('usound_approx', Float32, callback2)
    lane_direction = rospy.Subscriber('lane_direction', String, callback3)

    # setting up publications
    angle_steer = rospy.Publisher('angle_steer', Float32, queue_size=10)
    auto_velocity = rospy.Publisher('auto_velocity', Float32, queue_size=10)
    # manual_throttle = rospy.Publisher('manual_throttle', Float32, queue_size=10)
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
        # mnl_thrtl = 20.0
        # at_tgl = True
        # actl_vel = 20.0

        # steering angle based on bicycle model

        if lane_crv > 100:
            angl_str = 0
        else:
            angl_str = math.atan(wheelbase / lane_crv)
            print(angl_str,lane_crv,lane_dir)

        if angl_str > 0.7:
            angl_str = 0.7

        at_vel = angleToVel(angl_str)
        auto_velocity.publish(at_vel)

        if lane_dir == 'Left':
            angl_str = -angl_str
        angle_steer.publish(angl_str)

        # manual_throttle.publish(mnl_thrtl)
        # auto_toggle.publish(at_tgl)
        # actual_velocity.publish(actl_vel)

        rate.sleep()


if __name__ == '__main__':
    try:
        decision()
    except rospy.ROSInterruptException:
        pass
