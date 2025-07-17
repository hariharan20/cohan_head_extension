import rospy 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from cohan_msgs.msg import Trajectory , TrajectoryStamped , TrackedAgents
import tf

class robot_state_manager:
    def __init__(self) : 
        super().__init__()
        self.robot_pose = Pose2D()  
        rospy.Subscriber('/base_pose_ground_truth' , Odometry , self.robot_pose_cb)
        rospy.Subscriber('/move_base/HATebLocalPlannerROS/local_traj', TrajectoryStamped, self.cohan_callback)
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
        pass

    def cohan_callback(self, data):
        pass