import rospy
from sensor_msgs.msg import Image , CameraInfo
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from utils.utils import normalize_angle
import tf
import math
import rospy 
from sensor_msgs.msg import Image, CameraInfo
import message_filters


class vision : 
    def __init__(self):
        super().__init__()
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.point_camera = PointStamped()
        self.head_goal_point = JointTrajectoryPoint()
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n-pose.pt')
        self.point_camera.header.frame_id = 'head_mount_l515_depth_optical_frame'
        self.head_pan_base_offset = 1.132
        self.camera_info = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo, timeout=1.0)


        
        self.pub = rospy.Publisher('/eye_center_image', Image, queue_size=1)
        self.aligned_depth_pub = rospy.Publisher('/aligned_depth_image', Image, queue_size=1)
        self.viz_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)

        
        self.listener = tf.TransformListener()
        self.listener.waitForTransform('base_link', 'head_mount_l515_depth_optical_frame', rospy.Time(0), rospy.Duration(1.0))

        
        
        self.image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_image_sub  = message_filters.Subscriber('/camera/depth/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_image_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        
    def image_callback(self, image, depth_image):
        self.color_image = image
        self.depth_image = depth_image


    def get_human_eye_coordinates(self):
        if self.color_image is None or self.depth_image is None:
            rospy.logwarn("Color or depth image not received yet.")
            return None
        img = self.bridge.imgmsg_to_cv2(self.color_image, desired_encoding="passthrough").astype(np.float32) 
        depth_np = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding="passthrough").astype(np.float32) 
        results = self.model(img , verbose=False)
        for r in results:
            for kp in r.keypoints.xy:
                x1, y1 = map(int, kp[1])
                x2, y2 = map(int, kp[2])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

        # point_wrt_base_link = depth_to_xyz(cx , cy, depth_np)
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        px = self.camera_info.K[2]
        py = self.camera_info.K[5]
        z = depth_np[cy, cx]  
        x = (cx - px) * z / fx
        y = (cy - py) * z / fy
        self.point_camera.point.x = x 
        self.point_camera.point.y = y
        self.point_camera.point.z = z
        self.point_camera.header.stamp = rospy.Time(0)
        point_body = self.listener.transformPoint('base_link', self.point_camera)
        if z == 0 or np.isnan(z): 
            return None


        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "points"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point_body.point.x
        marker.pose.position.y = point_body.point.y
        marker.pose.position.z = point_body.point.z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.viz_pub.publish(marker)    

        cv2.circle(depth_np, (cx, cy), 5, (0, 0, 255), -1) 
        aligned_depth_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding='passthrough')
        self.aligned_depth_pub.publish(aligned_depth_msg)

        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        eye_center_image = self.bridge.cv2_to_imgmsg(img, encoding='passthrough')
        self.pub.publish(eye_center_image)


        return point_body.point.x, point_body.point.y, point_body.point.z


    def get_human_gaze_on_map(self):
        