#!/usr/bin/env python3

import numpy as np
from cw2q4.youbotKineBase import YoubotKinematicBase


class YoubotKinematicStudent(YoubotKinematicBase):
    def __init__(self):
        super(YoubotKinematicStudent, self).__init__(tf_suffix='student')

        # Set the offset for theta --> This was updated on 03/12/2024. 
        youbot_joint_offsets = [170.0 * np.pi / 180.0,
                                -65.0 * np.pi / 180.0,
                                146 * np.pi / 180,
                                -102.5 * np.pi / 180,
                                -167.5 * np.pi / 180]

        # Apply joint offsets to dh parameters
        self.dh_params['theta'] = [theta + offset for theta, offset in
                                   zip(self.dh_params['theta'], youbot_joint_offsets)]

        # Joint reading polarity signs
        self.youbot_joint_readings_polarity = [-1, 1, 1, 1, 1]

    def forward_kinematics(self, joints_readings, up_to_joint=5):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        frame number. The frame transformation used in the computation are derived from dh parameters and
        joint_readings.
        Args:
            joints_readings (list): the state of the robot joints. In a youbot those are revolute
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint}
                w.r.t the base of the robot.
        """
        assert isinstance(self.dh_params, dict)
        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)
        assert up_to_joint >= 0
        assert up_to_joint <= len(self.dh_params['a'])

        T = np.identity(4)

	# --> This was updated on 23/11/2023. Feel free to use your own code.

        # Apply offset and polarity to joint readings (found in URDF file)
        joints_readings = [sign * angle for sign, angle in zip(self.youbot_joint_readings_polarity, joints_readings)]

        for i in range(up_to_joint):
            A = self.standard_dh(self.dh_params['a'][i],
                                 self.dh_params['alpha'][i],
                                 self.dh_params['d'][i],
                                 self.dh_params['theta'][i] + joints_readings[i])
            T = T.dot(A)
            
        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"
        return T

    def get_jacobian(self, joint):
        """Given the joint values of the robot, compute the Jacobian matrix. Coursework 2 Question 4a.
        Reference - Lecture 5 slide 24.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            Jacobian (numpy.ndarray): NumPy matrix of size 6x5 which is the Jacobian matrix.
        """
        assert isinstance(joint, list)
        assert len(joint) == 5

        # Your code starts here ----------------------------

        # For your solution to match the KDL Jacobian, z0 needs to be set [0, 0, -1] instead of [0, 0, 1], since that is how its defined in the URDF.
        # Both are correct.

        # Initialise the Jacobian matrix which is 6x5
        jacobian = np.zeros((6, 5))
        
        # Compute forward kinematics of the end-effector position
        T_0_n = self.forward_kinematics(joint, up_to_joint=5)

        # Extract position of end-effector
        p_ee = T_0_n[:3, 3]

        # Initialise variables for the previous transformation matrix
        T_0_i = np.identity(4) # Indentity matrix for base frame

        for i in range(5):
            A_i = self.standard_dh(self.dh_params['a'][i],
                                   self.dh_params['alpha'][i],
                                   self.dh_params['d'][i],
                                   self.dh_params['theta'][i] + joint[i])
            
            # Update the transformation matrix from base to joint i
            T_0_i = T_0_i.dot(A_i) 

            # Extract rotation axis (z-axis) 
            z_i = T_0_i[:3, 2]

            # Compute the position of the i-th joint
            p_i = T_0_i[:3, 3]

            # Compute the linear velocity of the Jacobian by the cross product
            jacobian[:3, i] = np.cross(z_i, (p_ee - p_i))

            # Compute the angular velocity of the Jacobian from the rotation axis
            jacobian[3:, i] = z_i


        # Your code ends here ------------------------------

        assert jacobian.shape == (6, 5)
        return jacobian

    def check_singularity(self, joint):
        """Check for singularity condition given robot joints. Coursework 2 Question 4c.
        Reference Lecture 5 slide 30.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            singularity (bool): True if in singularity and False if not in singularity.

        """
        assert isinstance(joint, list)
        assert len(joint) == 5
        
        # Your code starts here ----------------------------

        # Compute the Jacobian matrix for the given joint configuration 
        jacobian = self.get_jacobian(joint)

        # Perform Singular Value Decomposition to check rank
        u, s, vh = np.linalg.svd(jacobian)

        # Check the smallest singular value 
        # If the smallest singular value is close to zero, the robot is in singularity
        threshold = 1e-6
        if s[-1] < threshold:
            singularity = True
        else: 
            singularity = False

        # Your code ends here ------------------------------

        assert isinstance(singularity, bool)
        return singularity
