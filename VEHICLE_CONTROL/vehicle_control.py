#!/usr/bin/env python
# license removed for brevity

import rospy
import numpy
from std_msgs.msg       import Float32
from std_msgs.msg       import Bool
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
wheelbase =
width =



# callback functions, necessary for all subscribers
def callback1(data):
    rospy.loginfo(rospy.get_caller_id() + "angle_steer: %s", data.data)


def callback2(data):
    rospy.loginfo(rospy.get_caller_id() + "auto_velocity: %s", data.data)


def callback3(data):
    rospy.loginfo(rospy.get_caller_id() + "manual_throttle: %s", data.data)


def callback4(data):
    rospy.loginfo(rospy.get_caller_id() + "auto_toggle: %s", data.data)


def callback5(data):
    rospy.loginfo(rospy.get_caller_id() + "actual_velocity: %s", data.data)


def vehicle_control():

    # initialising my node
    rospy.init_node('vehicle_control', anonymous=True)

    # setting up publications
    cmd_throttle = rospy.Publisher('cmd_throttle', Float32, queue_size=10)
    cmd_steer = rospy.Publisher('cmd_steer', Float32, queue_size=10)
    cmd_brake = rospy.Publisher('cmd_brake', Float32, queue_size=10)

    # setting up subscriptions
    angle_steer = rospy.Subscriber('angle_steer', Twist, callback1)
    auto_velocity = rospy.Subscriber('auto_velocity', Twist, callback2)
    manual_throttle =  rospy.Subscriber('manual_throttle', Float32, callback3)
    auto_toggle = rospy.Subscriber('auto_toggle', Bool, callback4)
    actual_velocity = rospy.Subscriber('actual_velocity', Twist, callback5)

    # refresh rate of pubs/subs
    rate = rospy.Rate(10) # 10hz

    # prevents rospy from stopping unless node is closed
    rospy.spin()

    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        # pub.publish(hello_str)
        if auto_toggle == True:
            if auto_velocity == -1.0:
                cmd_brake.publish(emergency_brake)
                cmd_throttle.publish(0)

            elif actual_velocity == auto_velocity:
                cmd_throttle.publish(cruise_throttle)

            elif actual_velocity < auto_velocity:
                cmd_throttle.publish(accel_throttle)

            elif actual_velocity > auto_velocity:
                cmd_brake.publish(gentle_brake)
                cmd_throttle.publish(0)

            # publishes steering input based on upcoming lane angle, converted using
            # standard bicycle model
            cmd_steer.publish(carModelConversion(angle_steer,actual_velocity))

        else:
            cmd_throttle.publish(manual_throttle)
            cmd_steer.publish(-1)
            cmd_steer.publish(-999)

        rate.sleep()


def carModelConversion(angle_steer,actual_velocity):
    distance = actual_velocity*3 # arbitrary 3 seconds

    if angle_steer > 0.01: # steering deadzone, interpreted as straight motion
        steering_input = numpy.arctan((angle_steer*wheelbase)/distance)
    else:
        steering_input = 0

    return steering_input


if __name__ == '__main__':
    try:
        vehicle_control()
    except rospy.ROSInterruptException:
        pass