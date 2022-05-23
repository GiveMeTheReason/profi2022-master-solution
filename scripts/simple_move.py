#!/usr/bin/env python

import time
from math import sin
import numpy as np
import cv2

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import tf
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion

class SimpleMover():

    def __init__(self):
        
        rospy.init_node('simple_mover', anonymous=True)
        rospy.on_shutdown(self.shutdown)
        self.listener = tf.TransformListener()

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber("diff_drive_robot/camera1/image_raw", Image, self.camera_cb)
        rospy.Subscriber('odom', Odometry, self.odometry_callback )
        self.rate = rospy.Rate(30)
        self.cv_bridge = CvBridge()
        

        self.pos = np.array([0,0,0])

    def odometry_callback(self, msg) :

        # try:
        #     (trans,rot) = self.listener.lookupTransform( '/base', '/odom', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     continue

        self.pos = np.zeros(3)
        self.pos[0] = msg.pose.pose.position.x * 1
        self.pos[1] = msg.pose.pose.position.y * 1
        quater = msg.pose.pose.orientation
        orientation_list = [quater.x, quater.y, quater.z, quater.w]
        self.pos[2] = euler_from_quaternion(orientation_list)[2]
        print self.pos 
        
    def camera_cb(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        except CvBridgeError, e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        arrow_len = 30
        pt = self.pos.copy()
        pt[2] = -pt[2]
        pt2 = pt[:-1] + arrow_len * np.array(np.cos(pt[2]), np.sin(pt[2]))
        cv2.arrowedLine(cv_image, tuple(pt[:-1].astype(np.int32)), tuple(pt2.astype(np.int32)), (0,255,0), 3)
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
