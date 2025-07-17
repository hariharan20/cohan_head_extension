#! /usr/bin/env python
import rospy 
from sensor_msgs.msg import Image
import message_filters

from observations.vision import vision
from actions.gaze import gaze
from utils.utils import get_agent_id 
from observations.robot_state_manager import robot_state_manager


rospy.set_param('look_at_agent_eye' , True)
class social_signal_generator:
    def __init__(self):
        self.robot_states = robot_state_manager()
        self.vision_processor = vision()
        self.gaze_controller = gaze()
        self.last_image_sent_time = rospy.Time(0)
        rospy.Timer(rospy.Duration(0.1), self.machine)
        
    def machine(self, _):
        gaze_at_agent = rospy.get_param('look_at_agent' , False)
        gaze_at_path = rospy.get_param('look_at_path' , False)
        gaze_at_agent_eye = rospy.get_param('look_at_agent_eye' , False)

        if gaze_at_agent_eye: 
            if (rospy.Time.now() - self.last_image_sent_time).to_sec() > 0.2:
                point_to_look = self.vision_processor.get_human_eye_coordinates()
                if point_to_look is not None:            
                    self.gaze_controller.look_at_point(point_to_look)
                    self.last_image_sent_time = rospy.Time.now()

            
        if gaze_at_path:
            if self.cohan_plan is not None:
                self.gaze_controller.look_at_path(self.cohan_plan, self.color_image)

        if gaze_at_agent:
            if self.cohan_plan is not None : 
                self.gaze_controller.look_at_agent(self.cohan_plan, self.color_image)


if __name__ == "__main__":
    rospy.init_node('social_signal_generator', anonymous=True)
    social_signal_generator()
    rospy.spin()


