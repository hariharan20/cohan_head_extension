#! /usr/bin/env python
import rospy 
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose2D
from cohan_msgs.msg import Trajectory , TrajectoryStamped , TrackedAgents
import tf
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from cohan_msg.msg import TrajectoryStamped
import math


class head_motion:
    def __init__(self):
        self.robot_pose = Pose2D()
        self.point = Pose2D()
        self.robot_head_angle = 0.0
        self.head_pose_time = 2.0 #in seconds (towards whose goal, the head is oriented)
        self.headtj_pub = rospy.Publisher('/head_traj_controller/command' , JointTrajectory , queue_size=10)
        self.agent_positions = []
        self.head_joint_trajectory_goal = JointTrajectory()
        self.head_goal_point = JointTrajectoryPoint()
        self.head_joint_trajectory_goal.joint_names = ['head_pan_joint' , 'head_tilt_joint']
        self.look_at_agent = False
        rospy.Subscriber('/move_base/HATebLocalPlannerROS/local_traj' , TrajectoryStamped , self.traj_callback)
        rospy.Subscriber('/base_pose_ground_truth' , Odometry , self.robot_pose_cb)
        rospy.Subscriber('/tracked_agents' , TrackedAgents , self.tracked_agents_cb)
    def robot_pose_cb(self, data):
        q = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        )
        
        m = tf.transformations.quaternion_matrix(q)
        
        self.robot_pose.x = data.pose.pose.position.x
        self.robot_pose.y = data.pose.pose.position.y
        self.robot_pose.theta = tf.transformations.euler_from_matrix(m)[2]

    def tracked_agents_cb(self ,data) : 
        if len(self.agent_positions ) == 0 :
            for agent in data.agents : 
                self.agent_positions.append([agent.segments[0].pose.pose.position.x , agent.segments[0].pose.pose.position.y])
        else :
            for i , agent in enumerate(data.agents) :
                self.agent_positions[i] = [agent.segments[0].pose.pose.position.x , agent.segments[0].pose.pose.position.y]

        agent_distance_to_robot = []
        for agent in self.agent_positions : 
            agent_distance_to_robot.append(math.sqrt((agent[0] - self.robot_pose.x)**2 + (agent[1] - self.robot_pose.y)**2))
        if min(agent_distance_to_robot) < 3.0: 
            closest_agent_index = agent_distance_to_robot.index(min(agent_distance_to_robot))
            closest_agent = self.agent_positions[closest_agent_index]
            angle_to_closest_agent = math.atan2(closest_agent[1] - self.robot_pose.y, closest_agent[0] - self.robot_pose.x)
            
            robot_head_angle = (angle_to_closest_agent - self.robot_pose.theta + math.pi) % (2 * math.pi) - math.pi
            if robot_head_angle > math.pi / 3 or robot_head_angle < -math.pi / 3:
                self.robot_head_angle = 0.0
                self.look_at_agent = False
            else: 
                self.robot_head_angle = robot_head_angle
                self.look_at_agent = True
        
        else :
            self.look_at_agent = False


    def path_angle_calculator(self, data) : 

        point_found = False
        for data_ in data.points:
            if data_.time_from_start > rospy.Duration(self.head_pose_time + 1): # Trial and Fix
                point_found = True
                break

        if not point_found:
            data_ = data.points[-1] 
        self.point.x = data_.pose.position.x
        self.point.y = data_.pose.position.y
        q = (
        data_.pose.orientation.x,
        data_.pose.orientation.y,
        data_.pose.orientation.z,
        data_.pose.orientation.w
        )

        m = tf.transformations.quaternion_matrix(q)
        self.point.theta = tf.transformations.euler_from_matrix(m)[2]   

        point_pos = np.array([self.point.x , self.point.y])
        robot_pos = np.array([self.robot_pose.x , self.robot_pose.y])
        vector_to_goal = point_pos - robot_pos
        goal_yaw = math.atan2(vector_to_goal[1], vector_to_goal[0])
        goal_orientation = (goal_yaw - self.robot_pose.theta + math.pi) % (2 * math.pi) - math.pi 
        return goal_orientation , 0

    def traj_callback(self, data):

        if self.look_at_agent == False:
            goal_yaw, goal_pitch = self.path_angle_calculator(data)
        else : 
            goal_yaw = self.robot_head_angle
            goal_pitch = 0.0

        self.head_goal_point.positions = [goal_yaw , goal_pitch]
        self.head_goal_point.time_from_start = rospy.Duration(0.1)
        self.head_joint_trajectory_goal.points.append(self.head_goal_point)
        self.head_joint_trajectory_goal.header.stamp = rospy.Time.now()
        self.headtj_pub.publish(self.head_joint_trajectory_goal)
        # rospy.loginfo("Head Command Sent")

if __name__ == "__main__":
    rospy.init_node('head_extension')
    obj = head_motion()
    rospy.spin()