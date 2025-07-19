#! /usr/bin/env python
import rospy 
from utils.utils import is_internet_connected , quat_to_euler , rad_to_deg2
from gtts import gTTS
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
import numpy as np



template_speech = {
    "greeting": "Hello! How can I assist you today?",
    "farewell": "Goodbye! Have a great day!",
    "error": "An error has occurred. Please try again.",
    "welcome": "Welcome to our service! How can I help you?",
    "thank_you": "Thank you for your patience and understanding.",
    "help": "I'm here to assist you. What do you need help with?"
}

position_dict =  {
    ('left' , 'behind') : {
        1 : 'follow and pass by left,move right',
        90 : 'cross behind from left,pass first',
        180 : 'pass by left,move right',
        270 : 'cross behind from right,pass first',
        359 : 'follow and pass by left,move right',
    },
    ('right' , 'behind') : {
        1 : 'follow and pass by right,move left',
        90 : 'cross behind from left,pass first',
        180 : 'pass by right,move left',
        270 : 'cross behind from right,pass first',
        359 : 'follow and pass by right,move left',
    },
    ('left' , 'front') : {
        1 : 'follow and pass by left,move right',
        90 : 'cross in front from left,pass second',
        180 : 'pass by left,move right',
        270 : 'cross in front from right,pass second',
        359 : 'follow and pass by left,move right',

    },
    ('right' , 'front') : {
        0 : 'follow and pass by right,move left',
        90 : 'crossing in front from left,pass second',
        180 : 'pass by right,move left',
        270 : 'crossing in front from right,pass second',
        359 : 'follow and pass by right,move left',
    },
}



class speech:
    def __init__(self):
        self.current_audio=  None
        # self.engine = pyttsx3.init()
        # self.engine.setProperty('rate', 150)  

    def speak(self , text):
        if not is_internet_connected():
            rospy.logwarn("Internet connection is not available. Using offline TTS.")
            # self.engine.say(text)
            # self.engine.runAndWait()
        else:
            tts = gTTS(text=text, lang='en')
            tts.save("temp.mp3")
            # print("Saved ")
            self.current_audio = AudioSegment.from_mp3("temp.mp3")
            play(self.current_audio)
    
    def play_speech(self, audio_type):
        if audio_type in template_speech:
            text = template_speech[audio_type]
            self.speak(text)
        else:
            rospy.logwarn(f"Audio type '{audio_type}' not found in prerecorded speech.")
    
    def speak_direction_of_passby(self, crossing_info , agent_plan , robot_plan , from_llm = False , suggest_human = False):
        crossing_index = crossing_info.indices[0]
        human_poses = []
        for pose in agent_plan.paths[0].path.poses :
            human_poses.append([pose.pose.position.x , pose.pose.position.y , pose.pose.orientation.z , pose.pose.orientation.w])

        human_crossing_point = human_poses[crossing_index]
        human_post_crossing_point = human_poses[crossing_index + 1] 

        robot_poses = []
        for pose in robot_plan.poses :
            robot_poses.append([pose.pose.position.x , pose.pose.position.y , pose.pose.orientation.z , pose.pose.orientation.w])
        robot_crossing_point = robot_poses[crossing_info.indices[0]]

        human_heading_angle = quat_to_euler(human_crossing_point[3] , human_crossing_point[2])
        robot_heading_angle = quat_to_euler(robot_crossing_point[3] , robot_crossing_point[2])
    
        human_dx = human_post_crossing_point[0] - human_crossing_point[0]
        human_dy = human_post_crossing_point[1] - human_crossing_point[1]

        angle_of_robot_wrt_human = rad_to_deg2(robot_heading_angle - human_heading_angle)

        distance_of_passby = np.linalg.norm(np.array(robot_crossing_point[:2]) - np.array(human_crossing_point[:2]))

        direction = self.get_direction(robot_crossing_point[:2] , human_crossing_point[:2] , human_dx , human_dy , angle_of_robot_wrt_human , suggest_human= suggest_human )

        if from_llm : 
            pass
        else :
            if suggest_human:
                self.speak(f"Could you please {direction}")
            else:
                self.speak(f"I will {direction} of you")



    def get_direction(self, robot_position , human_position , human_heading_dx , human_heading_dy , angle_of_robot_wrt_human , suggest_human = False):
        d1 = (robot_position[0] - human_position[0]) * (human_heading_dy) - (robot_position[1] - human_position[1]) * (human_heading_dx)
        d1_orthogonal = (robot_position[0] - human_position[0]) * (-human_heading_dx) - (robot_position[1] - human_position[1]) * (human_heading_dy)
        if d1 > 0:
            left_or_right = 'right'
        else:
            left_or_right = 'left'
        if d1_orthogonal > 0:
            front_or_behind = 'behind'
        else:
            front_or_behind = 'front'
        angle_dict = position_dict[(left_or_right ,front_or_behind)]
        angle_dict_keys = list(angle_dict.keys())
        angle_dict_keys = np.array(angle_dict_keys)
        angle_difference = np.abs(angle_dict_keys - angle_of_robot_wrt_human)
        min_index = np.argmin(angle_difference)
        crossing_direction = angle_dict[angle_dict_keys[min_index]]

        return crossing_direction.split(',')[0] if not suggest_human else crossing_direction.split(',')[1]
        
        # return crossing_direction

    def speak_direction_of_passby_static_human(self , tracked_agent_data , robot_plan):
        pass







