#!/usr/bin/env python

import time
from math import sin
import numpy as np
import cv2
import json
from global_planner import Global_Planner

import rospy
from geometry_msgs.msg import Twist, Bool
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import tf
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion

from mapper import Mapper
from sock_chooser import Sock_Chooser
from mpc import MPC
from robot import Robot

from utils import wrap_angle

json_file = open(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "config/config.json")))
layers = json.load(json_file)["model"]
color = json.load(json_file)["colors"]
color = json.load(json_file)["mpc"]
json_file.close()

class SimpleMover():

    def __init__(self):
        
        rospy.init_node('simple_mover', anonymous=True)
        rospy.on_shutdown(self.shutdown)
        self.listener = tf.TransformListener()

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber("diff_drive_robot/camera1/image_raw", Image, self.camera_cb)
        rospy.Subscriber('odom', Odometry, self.odometry_callback )
        rospy.Subscriber('monitor/is_sock_taken', Bool, self.sock_taken_callback)

        self.cv_bridge = CvBridge()
        
        self.sock_is_taken = False
        self.pos = np.array([0,0,0])

    def odometry_callback(self, msg) :

        # try:
        #     (trans,rot) = self.listener.lookupTransform( '/base', '/odom', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     continue
        x_scale = 480/(16.4*2)
        y_scale = 499/(16.72*2)
        img_sz = 600

        self.pos = np.zeros(3)
        self.pos[0] = msg.pose.pose.position.x * 1
        self.pos[1] = msg.pose.pose.position.y * 1
        quater = msg.pose.pose.orientation
        orientation_list = [quater.x, quater.y, quater.z, quater.w]
        self.pos[2] = euler_from_quaternion(orientation_list)[2]
        
        self.pos[0] *= x_scale 
        self.pos[1] *= y_scale
        buf = self.pos[0]
        self.pos[0] = -self.pos[1] + img_sz//2
        self.pos[1] = -buf + img_sz//2

        n_vec = np.array([np.cos(self.pos[2]), np.sin(self.pos[2])])
        n_vec[0] *= x_scale 
        n_vec[1] *= y_scale
        
        buf = n_vec[0]
        n_vec[0] = -n_vec[1] + img_sz//2 
        n_vec[1] = -buf + img_sz//2

        self.pos[2] = wrap_angle(np.arctan2(n_vec[1], n_vec[0]))
         
        print self.pos

    def sock_taken_callback(self, msg) :
        self.sock_is_taken = msg.data


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

if __name__ == 'main' :
    
    simple_mover = SimpleMover()
    mapper = Mapper()
    sock_chooser = Sock_Chooser(mapper.targets) # ???
    robot = Robot(dim_observation = 3, dim_action = 2, dim_hidden = 10)
    
    mpc = MPC( robot, params)

    mpc_rate = rospy.Rate(1/sampling_time)

    while not mapper.target:
        # Get statis map
        map_msg = rospy.wait_for_message("diff_drive_robot/camera1/image_raw", Image)
        mapper.static_map(map_msg)

    global_planner = Global_Planner(  )
    control = np.zeros(2)

    while not np.all(sock_chooser.socks) :

        cur_pose = simple_mover.pos
        cur_sock = sock_chooser.next_sock( cur_pose )
        # we get it extrinsically from topic

        # Make a path to choosen sock
        # get bunch of refference points
        path_cost, local_path = global_planner.find_route( cur_pose, cur_sock )
        mpc.set_route( local_path )
        
        while not simple_mover.sock_is_taken :
            # - get robot position
            # - Choose sock
            cur_pose = simple_mover.pos
            robot.set_pose( cur_pose )
            robot.update_model( robot.predictor(robot.prev_pose, control), cur_pose )
            
            # Now MPC do things into it
            # Do ros synchronisation here
            control = mpc.calc_next_control()
            control_msg = Twist()
            
            control_msg.linear.x = control[0]
            control_msg.angular.z = control[1]
            
            # gen messages Twist() from control
            simple_mover.cmd_vel_pub.publish( control_msg )
            # ros sleep
        
        cur_pose = simple_mover.pos
        # make current sock taken
        sock_chooser.take_sock(cur_pose)


    simple_mover.spin()
