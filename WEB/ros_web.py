#!/usr/bin/env python
# license removed for brevity

import rospy

from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist

subscriptions = {
    'angle_steer': Twist,
    'auto_velocity': Twist,
    'manual_throttle': Float32,
    'actual_velocity': Twist,
    'auto_toggle': Bool
}

publications = {
    'cmd_log': Bool
}


def callback(node):
    def cb(data):
        rospy.loginfo("%s - %s: %s", rospy.get_caller_id(), node, data.data)
    return cb


def web_hooks():

    rospy.init_node('web_hooks', anonymous=True)

    publishers = { k: rospy.Publisher(k, v, queue_size=10) for k, v in publications.items() }
    subscribers = { k: rospy.Subscriber(k, v, callback(k)) for k, v in subscriptions.items() }

    # refresh rate of pubs/subs
    rate = rospy.Rate(10) # 10hz

    # prevents rospy from stopping unless node is closed
    rospy.spin()

    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        # pub.publish(hello_str)

        rate.sleep()


if __name__ == '__main__':
    try:
        web_hooks()
    except rospy.ROSInterruptException:
        pass