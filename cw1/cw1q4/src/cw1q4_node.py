#!/usr/bin/env python3

import rospy
import numpy as np


# TODO: Include all the required service classes
# your code starts here -----------------------------

from cw1q4_srv.srv import quat2zyx
from cw1q4_srv.srv import quat2zyxRequest
from cw1q4_srv.srv import quat2zyxResponse
from cw1q4_srv.srv import quat2rodrigues
from cw1q4_srv.srv import quat2rodriguesRequest
from cw1q4_srv.srv import quat2rodriguesResponse

# your code ends here -------------------------------




def convert_quat2zyx(request):
    # TODO complete the function
    """Callback ROS service function to convert quaternion to Euler z-y-x representation

    Args:
        request (quat2zyxRequest): cw1q4_srv service message, containing
        the quaternion you need to convert.

    Returns:
        quat2zyxResponse: cw1q4_srv service response, in which 
        you store the requested euler angles 
    """
    assert isinstance(request, quat2zyxRequest)

    # Your code starts here ----------------------------
    
    # Get quaternion from quat2zyxRequest
    # Quaternions have 4 components: q_x, q_y, q_z, q_w
    qx = request.q.x
    qy = request.q.y
    qz = request.q.z
    qw = request.q.w
    

    # Find Euler angles using the quaternion to Euler formula
    # X-axis rotation
    roll = np.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)) 
    
    # Y-axis rotation
    pitch = np.arcsin(2.0 * (qw * qy - qz * qx))
    
    # Z-axis rotation
    yaw = np.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)) 


    # Create a response object
    response = quat2zyxResponse()

    # Fill in the response with the Euler angles
    response.x.data = roll
    response.y.data = pitch
    response.z.data = yaw

    # Your code ends here ------------------------------

    assert isinstance(response, quat2zyxResponse)
    return response


def convert_quat2rodrigues(request):
    # TODO complete the function

    """Callback ROS service function to convert quaternion to rodrigues representation
    
    Args:
        request (quat2rodriguesRequest): cw1q4_srv service message, containing
        the quaternion you need to convert

    Returns:
        quat2rodriguesResponse: cw1q4_srv service response, in which 
        you store the requested rodrigues representation 
    """
    assert isinstance(request, quat2rodriguesRequest)

    # Your code starts here ----------------------------
    
    # Get the quaternion from quat2rodriguesRequest
    qx = request.q.x
    qy = request.q.y
    qz = request.q.z
    qw = request.q.w

    # Normalise quaternion
    norm_quat = np.sqrt(qw* qw + qx * qx + qy * qy + qz * qz)
    
    # If qw > 1, normalise to avoid errors 
    if qw > 1.0:
        qw /= norm_quat
        qx /= norm_quat
        qy /= norm_quat
        qz /= norm_quat


    # Calculate the rotation angle theta
    theta = 2 * np.arccos(qw)

    # Calculate the magnitude of the vector of the quaternion i.e. axis of rotation
    # Assumes w is less than 1, so it is always positive
    rot_axis = np.sqrt(1 - qw * qw)

    # Check if the axis of rotation approaches 0 to avoid dividing by zero
    # If rot_axis is close to 0, return the quaternion components
    # Otherwise, normalise the rotation axis to get the direction 
    # and scale by the angle of rotation
    if rot_axis < 1e-6:
        r_x = qx
        r_y = qy
        r_z = qz
    else:
        r_x = (qx/rot_axis) * theta
        r_y = (qy/rot_axis) * theta
        r_z = (qz/rot_axis) * theta 

    # Create a response object and store the Rodrigues vector components
    response = quat2rodriguesResponse()
    response.x.data = r_x
    response.y.data = r_y
    response.z.data = r_z

    # Your code ends here ------------------------------

    assert isinstance(response, quat2rodriguesResponse)
    return response

def rotation_converter():
    rospy.init_node('rotation_converter')

    #Initialise the services
    rospy.Service('quat2rodrigues', quat2rodrigues, convert_quat2rodrigues)
    rospy.Service('quat2zyx', quat2zyx, convert_quat2zyx)

    rospy.spin()


if __name__ == "__main__":
    rotation_converter()
