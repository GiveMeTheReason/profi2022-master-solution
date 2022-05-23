import cv2

class Mapper():
    """Creates static map and estimates robot position"""
    def __init__(self):
        rp = rospkg.RosPack()
        self.cv_bridge = CvBridge()

        self.cv_robot_pose_pub = rospy.Publisher('cv/robot_pose', Pose, queue_size=1 )

        json_file = open(rp.get_path("profi2022_master_solution") + "/config/colors.json")
        self.colors = json.load(json_file)
        json_file.close()

        self.unpack_colors(self.colors)
        self.target = {}
        self.pos = np.array([0, 0, np.pi])


    def unpack_colors(self, colors):
        """Color settings"""
        self.tolerance = np.array(self.colors["tolerance"][::-1])
        self.robot = np.array(self.colors["robot"][::-1])
        self.sock = np.array(self.colors["sock"]["main"][::-1])
        self.sock_center = np.array(self.colors["sock"]["center"][::-1])
        self.wall = np.array(self.colors["wall"][::-1])
        self.obstacle = np.array(self.colors["obstacle"][::-1])
        self.floor = np.array(self.colors["floor"][::-1])


    def static_map(self, msg):
        
        """Create static map"""
        img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        self.map = np.zeros_like(img[:, :, 0])

        # walls
        wall_mask = np.all(img > self.wall - self.tolerance, axis=2)
        wall_mask += np.all((img > self.obstacle - self.tolerance) * \
            (img < self.obstacle + self.tolerance), axis=2)
        self.map[wall_mask] = 255

        # robot
        robot_mask = np.all((img > self.robot - 2*self.tolerance) * \
            (img < self.robot + 2*self.tolerance), axis=2)

        # white wheels detected as wall, extend robot a little to fix this
        ext = 2
        d = 2 * ext + 1
        robot_ext_mask = cv2.dilate(robot_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (d, d)))
        self.map[robot_ext_mask>0] = 0

        # find robot sides
        hull = cv2.findContours(robot_ext_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0][:,-1]
        _, sides, _ = cv2.minAreaRect(hull)

        # extend obstacles by robot side
        d = int(min(sides) + ext * 2 + 1)
        self.map = cv2.dilate(self.map, cv2.getStructuringElement(cv2.MORPH_RECT, (d, d)))

        # find socks
        sock_main_mask = np.all((img > self.sock - self.tolerance) * \
            (img < self.sock + self.tolerance), axis=2)
        sock_main_mask = cv2.dilate(sock_main_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        self.map[sock_main_mask>0] = 255

        sock_center_mask = np.all((img > self.sock_center - 2*self.tolerance) * \
            (img < self.sock_center + 2*self.tolerance), axis=2)
        self.map[sock_center_mask] = 255

        self.find_targets(sock_center_mask)
        self.map_targets()
        self.make_cost_map(self.map)
        self.robot_position(img)


    def make_cost_map(self, mp, n=2, c=7):
        """Creates cost fine if robot near obstacles
        n: border size (more is longer for Planner)
        c: fine step (equals to diagonal cost gives best result)
        """
        self.cost_map = np.zeros_like(mp)

        for i in range(3, 3+2*n, 2):
            self.cost_map[cv2.morphologyEx(mp.astype(np.uint8), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (i, i))) > 0] += c


    def find_targets(self, sock_mask):
        sock_centers = cv2.findContours(sock_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        self.target = {}
        target_mean_diam = 0
        for i, center in enumerate(sock_centers):
            mom = cv2.moments(center)
            if mom['m00'] == 0 :
                mom['m00'] += 1e-5
            cy, cx = round(mom['m10'] / mom['m00']), round(mom['m01'] / mom['m00'])
            _, rad = cv2.minEnclosingCircle(center)
            target_mean_diam += rad
            self.target[str(i)] = [int(cx), int(cy)]
        self.target_mean_diam = int(round(2 * target_mean_diam / len(self.target.keys())))

        self.passed_targets = {}
        for key in self.target.keys():
            self.passed_targets[key] = False


    def pass_targets(self, pos):
        min_dist = 1e9
        for key in self.target.keys():
            dist = abs(pos[0] - self.target[key][0]) + abs(pos[1] - self.target[key][1])
            if dist < min_dist:
                min_dist = dist
                near_key = key
        self.passed_targets[near_key] = True


    def map_targets(self):
        self.map_target = np.zeros_like(self.map)
        for key in self.target.keys():
            if self.passed_targets[key] == False:
                self.map_target[self.target[key][0], self.target[key][1]] = 255
        d = 2 * self.target_mean_diam + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        self.map_target[(cv2.dilate(self.map_target.astype(np.uint8), kernel)>0) * (self.map==0)] = 255

    # make publishing out of func. Create subscriber and publisher here, 
    # and manually publish robot position after getting robot position

    def robot_position(self, img):

        robot_pose_msg = Pose()

        robot_mask = np.all((img > self.robot - self.tolerance) * \
            (img < self.robot + self.tolerance), axis=2)

        hull = cv2.findContours(robot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0][:,-1]
    
        mom = cv2.moments(hull)
        if mom['m00'] == 0 :
            mom['m00'] += 1e-5
        x_pos, y_pos = round(mom['m10'] / mom['m00']), round(mom['m01'] / mom['m00'])

        # print( 'robot pose : (', x_pos, y_pos, angle,')' )
        self.pos[:-1] = np.array([x_pos, y_pos])

        robot_pose_msg.position.x = x_pos
        robot_pose_msg.position.y = y_pos
        robot_pose_msg.position.z = 0
        robot_pose_msg.orientation = Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0,self.pos[-1]))

        self.cv_robot_pose_pub.publish(robot_pose_msg)
        return self.pos.copy()

    def robot_position_with_angle(self, img):  
        
        robot_pose_msg = Pose()
        angle_win = np.pi/6
        prev_pos = self.pos.copy()

        robot_mask = np.all((img > self.robot - self.tolerance) * \
            (img < self.robot + self.tolerance), axis=2)

        hull = cv2.findContours(robot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0][:,-1]
        _, (height, width), angle = cv2.minAreaRect(hull)
        angle = np.deg2rad(angle)
        if height < width:
            angle += np.pi/2
            angle = wrap_angle(angle)

        mom = cv2.moments(hull)
        if mom['m00'] == 0 :
            mom['m00'] += 1e-5
        x_pos, y_pos = round(mom['m10'] / mom['m00']), round(mom['m01'] / mom['m00'])

        d_angle = abs(self.pos[2] - angle) 
        if d_angle > np.pi :
            d_angle = 2*np.pi - d_angle
        if d_angle > angle_win :
            angle += np.pi
            angle = wrap_angle(angle)
            d_angle = abs(self.pos[2] - angle)
            if d_angle > np.pi :
                d_angle = 2*np.pi - d_angle
            # print ('angle = ', angle, 'init_angle = ', self.pos[2], 'd_angle = ', d_angle)
        # print('final angle = ', angle)
            

        # print( 'robot pose : (', x_pos, y_pos, angle,')' )
        self.pos = np.array([x_pos, y_pos, angle])

        robot_pose_msg.position.x = x_pos
        robot_pose_msg.position.y = y_pos
        robot_pose_msg.position.z = 0
        robot_pose_msg.orientation = Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0, angle))

        self.cv_robot_pose_pub.publish(robot_pose_msg)
        return self.pos.copy()