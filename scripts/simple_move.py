#!/usr/bin/env python

import time
from math import sin

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

class SimpleMover():

    def __init__(self):
        rospy.init_node('simple_mover', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber("diff_drive_robot/camera1/image_raw", Image, self.camera_cb)
        self.rate = rospy.Rate(30)

        self.cv_bridge = CvBridge()


    def camera_cb(self, msg):

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        except CvBridgeError, e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        self.show_image(cv_image)


    def show_image(self, img):
        cv2.imshow("Camera 1 from Robot", img)
        cv2.waitKey(3)


    def spin(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            twist_msg = Twist()
            t = time.time() - start_time
            twist_msg.linear.x  = 0.1
            twist_msg.angular.z = 0.4 * sin(0.3 * t)
            self.cmd_vel_pub.publish(twist_msg)
            self.rate.sleep()


    def shutdown(self):
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)


simple_mover = SimpleMover()
simple_mover.spin()
