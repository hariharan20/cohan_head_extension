import math
import socket
import tf
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi



def get_agent_id(cohan_agent_plans):
    # To look at a specific agent.
    pass


def is_internet_connected(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False
    


def quat_to_euler(w , z):
    euler_angles = tf.transformations.euler_from_quaternion([0 , 0  , z , w])
    return euler_angles[2]




def rad_to_deg2(self, rad) : 
    if rad < 0 : 
        return 360 + (rad *180 / math.pi)
    return rad * 180 / math.pi