import rospy 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from cohan_msgs.msg import Trajectory , TrajectoryStamped , TrackedAgents
import tf

class robot_state_manager:
    def __init__(self) : 
        super().__init__()
        self.robot_pose = Pose2D() 
        self.robot_planned_trajectory = None 
        self.tracked_agents_data = TrackedAgents()
        self.last_traj_time = rospy.Time.now()
        self.last_tracked_agent_data = rospy.Time.now()
        rospy.Subscriber('/base_pose_ground_truth' , Odometry , self.robot_pose_cb)
        rospy.Subscriber('/move_base/HATebLocalPlannerROS/local_traj', TrajectoryStamped, self.cohan_callback)
        rospy.Subscriber('/tracked_agents' , TrackedAgents , self.tracked_agents_cb)
        rospy.Timer(rospy.Duration(0.1), self.check_trajectory_timeout)

    
    def robot_pose_cb(self, data):
        q = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        )
        
        m = tf.transformations.quaternion_matrix(q)
        robot_pose = Pose2D()
        robot_pose.x = data.pose.pose.position.x
        robot_pose.y = data.pose.pose.position.y
        robot_pose.theta = tf.transformations.euler_from_matrix(m)[2]
        self.robot_pose = robot_pose

    def tracked_agents_cb(self ,data) :
        self.tracked_agents_data = data
        self.last_tracked_agent_data = rospy.Time.now()

    def cohan_callback(self, data):
        self.robot_planned_trajectory = data
        self.last_traj_time = rospy.Time.now()

    def check_trajectory_timeout(self, _):
        if rospy.Time.now() - self.last_traj_time > rospy.Duration(1.0):  
            self.robot_planned_trajectory = None
        if rospy.Time.now() - self.last_tracked_agent_data > rospy.Duration(1.0):
            self.tracked_agents_data = None






