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
    response.roll = roll
    response.pitch = pitch
    response.yaw = yaw


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
    
    qx = request.qx
    qy = request.qy
    qz = request.qz
    qw = request.qw

    # Calculate axis of rotation 
    # Axis of rotation is the normalized vector [qx, qy, qz]
    
    axis_mag = np.sqrt(qx * qx + qy * qy + qz * qz)
    
    # If the magnitude is zero, the quaternion represents no rotation, so return a zero Rodrigues vector
    
    if axis_mag == 0:
        response = quat2rodriguesResponse()
        response.rx = 0.0
        response.ry = 0.0
        response.rz = 0.0
        return response

    # Normalize the vector to get the direction of the axis of rotation
    
    x = qx / axis_mag
    y = qy / axis_mag
    z = qz / axis_mag

    # Calculate the rotation angle theta
    
    theta = 2 * np.arctan2(axis_mag, qw)

    # The Rodrigues vector is the angle times the unit axis of rotation
    
    r_x = theta * x
    r_y = theta * y
    r_z = theta * z

    # Create a response object and store the Rodrigues vector components
    
    response = quat2rodriguesResponse()
    response.r_x = r_x
    response.r_y = r_y
    response.r_z = r_z

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
