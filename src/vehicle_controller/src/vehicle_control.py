#!/usr/bin/env python
# license removed for brevity

import rospy
import math
from std_msgs.msg       import Float32
from std_msgs.msg       import Bool
from std_msgs.msg	import Int8
from geometry_msgs.msg  import Twist

# OVERALL DRIVE PARAMETERS
# CONFIGURE THIS TO MODULATE THROTTLE/BRAKING CHARACTERISTICS
gentle_throttle = 0.1 # ~ smooth slowdown
cruise_throttle = 0.2 # amount of throttle needed to maintain constant speed
accel_throttle = 0.5 # ~ for gradual acceleration

gentle_brake = 0.1 # amount of brake needed to gradual slowdown
normal_brake = 0.2 # ~ for normal stop
emergency_brake = 1 # ~ for emergency stop

# VEHICLE PHYSICAL PARAMETERS
# CONFIGURE THIS FOR CHANGES TO WHEELBASE, HEIGHT, TIRES, ETC.
wheelbase = 2

# GLOBAL VARS
angle_str = None
auto_vel = None
manual_ttl = None
auto_tgl = None
acual_vel = None

# callback functions, necessary for all subscribers
def callback1(data):
    rospy.loginfo(rospy.get_caller_id() + "angle_steer: %s", data.data)
    global angle_str
    angle_str = data.data


def callback2(data):
    rospy.loginfo(rospy.get_caller_id() + "auto_velocity: %s", data.data)
    global auto_vel
    auto_vel = data.data


def callback3(data):
    rospy.loginfo(rospy.get_caller_id() + "manual_throttle: %s", data.data)
    global manual_ttl
    manual_ttl = data.data


def callback4(data):
    rospy.loginfo(rospy.get_caller_id() + "auto_toggle: %s", data.data)
    global auto_tgl
    auto_tgl = data.data

def callback5(data):
    rospy.loginfo(rospy.get_caller_id() + "actual_velocity: %s", data.data)
    global actual_vel
    actual_vel = data.data


def vehicle_control():

    print("vehicle_control node started!")

    # initialising my node
    rospy.init_node('vehicle_control', anonymous=True)

    # setting up publications
    cmd_throttle = rospy.Publisher('cmd_throttle', Float32, queue_size=10)
    cmd_steer = rospy.Publisher('cmd_steer', Float32, queue_size=10)
    cmd_brake = rospy.Publisher('cmd_brake', Float32, queue_size=10)

    # setting up subscriptions
    angle_steer = rospy.Subscriber('angle_steer', Float32, callback1)
    auto_velocity = rospy.Subscriber('auto_velocity', Float32, callback2)
    manual_throttle =  rospy.Subscriber('manual_throttle', Float32, callback3)
    auto_toggle = rospy.Subscriber('auto_toggle', Bool, callback4)
    actual_velocity = rospy.Subscriber('actual_velocity', Float32, callback5)

    # refresh rate of pubs/subs
    rate = rospy.Rate(10) # 10hz

    # prevents rospy from stopping unless node is closed
    #rospy.spin()

    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        # pub.publish(hello_str)
        if auto_tgl == True:
            if auto_vel == -1.0:
                cmd_brake.publish(emergency_brake)
                cmd_throttle.publish(0)
                rospy.loginfo("Emergency brake deployed")
                print("Emergency brake deployed")

            elif actual_vel == auto_vel:
                cmd_throttle.publish(cruise_throttle)
                rospy.loginfo("Cruising")
                print("Cruising")

            elif actual_vel < auto_vel:
                cmd_throttle.publish(accel_throttle)
                rospy.loginfo("Accelerating")
                print("Accelerating")

            elif actual_vel > auto_vel:
                cmd_brake.publish(gentle_brake)
                cmd_throttle.publish(0)
                rospy.loginfo("Braking")
                print("Braking")

            # publishes steering input based on upcoming lane angle, converted using
            # standard bicycle model
            steering_angle = angle_str
            cmd_steer.publish(steering_angle)
            rospy.loginfo("Steering angle: {:.4f}".format(steering_angle))

        elif auto_tgl == False:
            cmd_throttle.publish(manual_ttl)
            cmd_steer.publish(-999)
            rospy.loginfo("Manual mode engaged")
            print("Manual mode engaged")

        rate.sleep()


# def carModelConversion(angle_steer,actual_velocity):
#     distance = actual_velocity*3 # arbitrary 3 seconds
#
#     if angle_steer > 0.01: # steering deadzone, interpreted as straight motion
#         steering_input = math.atan((angle_steer*wheelbase)/distance)
#     else:
#         steering_input = 0
#
#     return steering_input


if __name__ == '__main__':
    try:
        vehicle_control()
    except rospy.ROSInterruptException:
        pass
