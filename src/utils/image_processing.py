import rospy
from sensor_msgs.msg import Image , CameraInfo
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf
import math

head_pan_base_offset = 1.132

rospy.init_node('eye_center_publisher')

point_camera = PointStamped()
head_goal_point = JointTrajectoryPoint()
listener = tf.TransformListener()
bridge = CvBridge()
model = YOLO('yolov8n-pose.pt')



point_camera.header.frame_id = 'head_mount_l515_depth_optical_frame'
listener.waitForTransform('base_link', 'head_mount_l515_depth_optical_frame', rospy.Time(0), rospy.Duration(1.0))







pub = rospy.Publisher('/eye_center_image', Image, queue_size=1)
aligned_depth_pub = rospy.Publisher('/aligned_depth_image', Image, queue_size=1)
viz_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
headtj_pub = rospy.Publisher('/head_traj_controller/command' , JointTrajectory , queue_size=10)

 


camera_info = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo, timeout=1.0)

last_sent_time = rospy.Time(0)



def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def image_cb(image):

    global last_sent_time
    if (rospy.Time.now() - last_sent_time).to_sec() < 0.2:
        return
    img = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    results = model(img , verbose=False)
    for r in results:
        for kp in r.keypoints.xy:
            x1, y1 = map(int, kp[1])
            x2, y2 = map(int, kp[2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
    depth_image = rospy.wait_for_message('/camera/depth/image_raw', Image , timeout=1.0)



    depth_np = bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough").astype(np.float32) 
    point_wrt_base_link = depth_to_xyz(cx , cy, depth_np)
    

    if point_wrt_base_link :

        z_wrt_head_pan = point_wrt_base_link[2] - head_pan_base_offset
        tilt_angle = np.arctan2(z_wrt_head_pan, point_wrt_base_link[0])
        pan_angle = np.arctan2(point_wrt_base_link[1], point_wrt_base_link[0])

        print("Pan Angle Calc : ", z_wrt_head_pan , point_wrt_base_link[0] , pan_angle) 
        print("Tilt Angle Calc : ", point_wrt_base_link[1] , point_wrt_base_link[0] , tilt_angle)

        pan_angle = normalize_angle(pan_angle)
        tilt_angle = -normalize_angle(tilt_angle)

        print(pan_angle, tilt_angle)

        head_goal_point.positions = [pan_angle ,  tilt_angle]
        head_goal_point.time_from_start = rospy.Duration(0.3)


        head_joint_trajectory_goal = JointTrajectory()
        head_joint_trajectory_goal.joint_names = ['head_pan_joint' , 'head_tilt_joint']
        head_joint_trajectory_goal.points.append(head_goal_point)
        head_joint_trajectory_goal.header.stamp = rospy.Time.now()
        headtj_pub.publish(head_joint_trajectory_goal)
        last_sent_time = rospy.Time.now()

    
    depth_draw = depth_np.copy()    
    cv2.circle(depth_draw, (cx, cy), 5, (0, 0, 255), -1) 
    aligned_depth_msg = bridge.cv2_to_imgmsg(depth_draw, encoding='passthrough')
    aligned_depth_pub.publish(aligned_depth_msg)







def depth_to_xyz(u, v, depth_image):
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]
    z = depth_image[v, u]  
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point_camera.point.x = x 
    point_camera.point.y = y
    point_camera.point.z = z
    point_camera.header.stamp = rospy.Time(0)
    point_body = listener.transformPoint('base_link', point_camera)

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
    viz_pub.publish(marker)
    return point_body.point.x, point_body.point.y, point_body.point.z






if __name__ == '__main__':
    rospy.Subscriber('/camera/color/image_raw', Image , image_cb)
    rospy.spin()
