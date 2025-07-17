import rospy 
import tf
import numpy as np
import math


from trajectory_msgs.msg import JointTrajectory,  JointTrajectoryPoint
from utils.utils import normalize_angle

class gaze:
    def __init__(self):
        self.head_goal_point = JointTrajectoryPoint()
        self.head_goal_point.time_from_start = rospy.Duration(0.3)
        self.head_joint_trajectory_goal = JointTrajectory()
        self.head_joint_trajectory_goal.joint_names = ['head_pan_joint', 'head_tilt_joint']
        
        self.HEAD_POSE_TIME = 2.0
        self.HEAD_PAN_BASE_OFFSET = 1.132

        
        self.headtj_pub = rospy.Publisher('/head_traj_controller/command', JointTrajectory, queue_size=10)

    def rotate_head(self, pan_angle , tilt_angle):
        self.head_goal_point.positions = [pan_angle, tilt_angle]
        self.head_joint_trajectory_goal.points = [self.head_goal_point]
        self.head_joint_trajectory_goal.header.stamp = rospy.Time.now()
        self.headtj_pub.publish(self.head_joint_trajectory_goal)

    def look_at_path(self, planned_traj, robot_current_pose):
        point_found = False
        for data_ in planned_traj.points:
            if data_.time_from_start > rospy.Duration(self.HEAD_POSE_TIME + 1): 
                point_found = True
                break

        if not point_found:
            data_ = planned_traj.points[-1] 
        point_x = data_.pose.position.x
        point_y = data_.pose.position.y
        q = (
        data_.pose.orientation.x,
        data_.pose.orientation.y,
        data_.pose.orientation.z,
        data_.pose.orientation.w
        )

        m = tf.transformations.quaternion_matrix(q)
        point_theta = tf.transformations.euler_from_matrix(m)[2] #YET TO USE   

        point_pos = np.array([point_x , point_y])
        robot_pos = np.array([robot_current_pose.x , robot_current_pose.y])
        vector_to_goal = point_pos - robot_pos
        goal_yaw = math.atan2(vector_to_goal[1], vector_to_goal[0])
        goal_orientation = (goal_yaw - robot_current_pose.theta + math.pi) % (2 * math.pi) - math.pi 
        self.rotate_head(goal_orientation , 0.0)

    def look_at_agent(self, tracked_agents_data, robot_current_pose):
        agent_positions = []    
        for agent in tracked_agents_data.agents: 
            agent_positions.append([agent.segments[0].pose.pose.position.x, agent.segments[0].pose.pose.position.y])
        
        agent_distance_to_robot = []
        for agent in agent_positions: 
            agent_distance_to_robot.append(math.sqrt((agent[0] - robot_current_pose.x)**2 + (agent[1] - robot_current_pose.y)**2))
        
        if min(agent_distance_to_robot) < 3.0: 
            closest_agent_index = agent_distance_to_robot.index(min(agent_distance_to_robot))
            closest_agent = agent_positions[closest_agent_index]
            angle_to_closest_agent = math.atan2(closest_agent[1] - robot_current_pose.y, closest_agent[0] - robot_current_pose.x)
            robot_head_angle = normalize_angle(angle_to_closest_agent - robot_current_pose.theta)
            if robot_head_angle > math.pi / 3 or robot_head_angle < -math.pi / 3:
                robot_head_angle = 0.0
        else: 
            robot_head_angle = 0.0
        
        self.rotate_head(robot_head_angle, 0.0)



    def look_at_point(self, point_wrt_base_link):
        z_wrt_head_pan = point_wrt_base_link[2] - self.HEAD_PAN_BASE_OFFSET
        tilt_angle = np.arctan2(z_wrt_head_pan, point_wrt_base_link[0])
        pan_angle = np.arctan2(point_wrt_base_link[1], point_wrt_base_link[0])
        pan_angle = normalize_angle(pan_angle)
        tilt_angle = -normalize_angle(tilt_angle)
        self.rotate_head(pan_angle, tilt_angle)