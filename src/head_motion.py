#! /usr/bin/env python
import rospy 
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose2D
from cohan_msgs.msg import TrajectoryStamped
import tf
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cohan_msg.msg import TrajectoryStamped
import math


class head_motion:
    def __init__(self):
        self.robot_pose = Pose2D()
        self.point = Pose2D()
        self.robot_head_angle = 0.0
        self.head_joint_trajectory_goal = JointTrajectory()
        self.head_goal_point = JointTrajectoryPoint()
        self.head_joint_trajectory_goal.joint_names = ['head_pan_joint']
        self.head_pose_time = 2.0 #in seconds (towards whose goal, the head is oriented)
        self.headtj_pub = rospy.Publisher('/head_traj_controller/command' , JointTrajectory , queue_size=10)
        rospy.Subscriber('/base_pose_ground_truth' , Odometry , self.robot_pose_cb)
        rospy.Subscriber('/move_base/HATebLocalPlannerROS/local_traj' , TrajectoryStamped , self.angle_calculator)

    def robot_pose_cb(self, data):
        q = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        )
        
        m = tf.transformations.quaternion_matrix(q)
        
        self.robot_pose.x = r.pose.pose.position.x
        self.robot_pose.y = r.pose.pose.position.y
        self.robot_pose.theta = tf.transformations.euler_from_matrix(m)[2]

    def angle_calculator(self, data):
        
        for data_ in data.points:
            if data_.time_from_start > (self.head_pose_time + 1): # Trial and Fix
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
                self.head_goal_point.positions = [
                break
            else: 
                #TODO: for trajectories of less than 2 seconds 
                break
        point_pos = np.array([self.point.x , self.point.y])
        robot_pos = np.array([self.robot_pose.x , self.robot_pose.y])
        # print(person_pos)
        vector_to_goal = point_pos - robot_pos

        # Calculate the distance from the robot to the person
        # distance_to_goal = math.sqrt(vector_to_goal[0]**2 + vector_to_goal[1]**2)

        # Normalize the vector to the desired distance
        # normalized_vector = vector_to_goal/ distance_to_goal

        # Calculate the orientation needed to reach the person
        goal_orientation = (math.atan2(vector_to_goal[1], vector_to_goal[0]))- self.robot_pose.theta
        self.head_joint_trajectory_goal.points.position = [goal_orientation]
        self.head_joint_trajectory_goal.header.stamp = rospy.Time.now()
        self.headtj_pub.publish(self.head_joint_trajectory_goal)
