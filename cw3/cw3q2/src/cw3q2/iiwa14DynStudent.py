#!/usr/bin/env python3

import numpy as np
from cw3q2.iiwa14DynBase import Iiwa14DynamicBase


class Iiwa14DynamicRef(Iiwa14DynamicBase):
    def __init__(self):
        super(Iiwa14DynamicRef, self).__init__(tf_suffix='ref')

    def forward_kinematics(self, joints_readings, up_to_joint=7):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        joint. Reference Lecture 9 slide 13.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 7.
        Returns:
            np.ndarray The output is a numpy 4*4 matrix describing the transformation from the 'iiwa_link_0' frame to
            the selected joint frame.
        """

        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)

        T = np.identity(4)
        # iiwa base offset
        T[2, 3] = 0.1575

        # 1. Recall the order from lectures. T_rot_z * T_trans * T_rot_x * T_rot_y. You are given the location of each
        # joint with translation_vec, X_alpha, Y_alpha, Z_alpha. Also available are function T_rotationX, T_rotation_Y,
        # T_rotation_Z, T_translation for rotation and translation matrices.
        # 2. Use a for loop to compute the final transformation.
        for i in range(0, up_to_joint):
            T = T.dot(self.T_rotationZ(joints_readings[i]))
            T = T.dot(self.T_translation(self.translation_vec[i, :]))
            T = T.dot(self.T_rotationX(self.X_alpha[i]))
            T = T.dot(self.T_rotationY(self.Y_alpha[i]))

        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"

        return T

    def get_jacobian_centre_of_mass(self, joint_readings, up_to_joint=7):
        """Given the joint values of the robot, compute the Jacobian matrix at the centre of mass of the link.
        Reference - Lecture 9 slide 14.

        Args:
            joint_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute the Jacobian.
            Defaults to 7.

        Returns:
            jacobian (numpy.ndarray): The output is a numpy 6*7 matrix describing the Jacobian matrix defining at the
            centre of mass of a link.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7

        # Your code starts here ----------------------------

        # Initialise the Jacobian matrix
        jacobian = np.zeros((6, 7))

        # Compute transformation matrices for the center of mass
        T_base_com = [self.forward_kinematics_centre_of_mass(joint_readings, j + 1) for j in range(up_to_joint)]

        # Compute the position of each center of mass
        p_com = [T[:3, 3] for T in T_base_com]

        # z-axis of base frame
        z_axes = [np.array([0, 0, 1])]

        # Compute z-axis of each joint frame in the base frame
        for j in range(up_to_joint - 1):
            T = self.forward_kinematics(joint_readings, j + 1)
            z_axes.append(T[:3, 2])

        # Compute Jacobian columns for each joint    
        for i in range(up_to_joint): 
            # Linear velocity contribution
            p = p_com[up_to_joint - 1] - p_com[i]

            # Vector from joint to end-effection
            jacobian[:3, i] = np.cross(z_axes[i], p)

            # Angular velocity contribution
            jacobian[3:, i] = z_axes[i]

        # Your code ends here ------------------------------

        assert jacobian.shape == (6, 7)
        return jacobian

    def forward_kinematics_centre_of_mass(self, joints_readings, up_to_joint=7):
        """This function computes the forward kinematics up to the centre of mass for the given joint frame.
        Reference - Lecture 9 slide 14.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematicks.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint} for the
            centre of mass w.r.t the base of the robot.
        """
        T= np.identity(4)
        T[2, 3] = 0.1575

        T = self.forward_kinematics(joints_readings, up_to_joint-1)
        T = T.dot(self.T_rotationZ(joints_readings[up_to_joint-1]))
        T = T.dot(self.T_translation(self.link_cm[up_to_joint-1, :]))

        return T

    def get_B(self, joint_readings):
        """Given the joint positions of the robot, compute inertia matrix B.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            B (numpy.ndarray): The output is a numpy 7*7 matrix describing the inertia matrix B.
        """
        B = np.zeros((7, 7))
        
        # Your code starts here ------------------------------

        # Calculate contributions of each joint to the intertia matrix
        for i in range(7):
            # Compute Jacobian for the centre of mass of link i
            J_com = self.get_jacobian_centre_of_mass(joint_readings, up_to_joint=i + 1)

            # Extract linear and anglular components of the Jacobian 
            J_v = J_com[:3, :]
            J_w = J_com[3:, :]

            # Compute inertia contribution for link i
            # Mass of link i
            m_i = self.mass[i] 
            # Inertia tensor of link i in its local frame
            I_i = np.diag(self.Ixyz[i])

            # Contribution to B from linear and angular terms
            B += m_i * (J_v.T @ J_v) + (J_w.T @ I_i @ J_w)


        # Your code ends here ------------------------------
        
        return B
    
    def get_B_derivative(self, i, j, k, joint_readings):
        """
        Compute derivative of inertia matrix element B_ij with respect to q_k.
        """

        delta_qk = np.zeros(7)
        delta_qk[k] = 1e-6
        B_plus_delta_qk = self.get_B((np.array(joint_readings) + delta_qk).tolist())
        B_minus_delta_qk = self.get_B((np.array(joint_readings) - delta_qk).tolist())

        return (B_plus_delta_qk[i, j] - B_minus_delta_qk[i, j]) / (2 * 1e-6)

    def get_C_times_qdot(self, joint_readings, joint_velocities):
        """Given the joint positions and velocities of the robot, compute Coriolis terms C.
        Args:
            joint_readings (list): The positions of the robot joints.
            joint_velocities (list): The velocities of the robot joints.

        Returns:
            C (numpy.ndarray): The output is a numpy 7*1 matrix describing the Coriolis terms C times joint velocities.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        assert isinstance(joint_velocities, list)
        assert len(joint_velocities) == 7

        # Your code starts here ------------------------------

        # Initialise Coriolis terms
        C = np.zeros(7)

        # Compute partial derivatives of the inertia matrix B
        B = self.get_B(joint_readings)

        for k in range(7):
            for i in range(7):
                for j in range(7):
                    # Christoffel symbols of the first kind
                    c_ijk = 0.5 * (self.get_B_derivative(i, j, k, joint_readings) + self.get_B_derivative(i, j, k, joint_readings) - self.get_B_derivative(j, k, i, joint_readings))
                
                # Compute Coriolis term
                C[k] += c_ijk * joint_velocities[j] * joint_velocities[k]

        # Your code ends here ------------------------------

        assert isinstance(C, np.ndarray)
        assert C.shape == (7,)
        return C

    def get_G(self, joint_readings):
        """Given the joint positions of the robot, compute the gravity matrix g.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            G (numpy.ndarray): The output is a numpy 7*1 numpy array describing the gravity matrix g.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7

        # Your code starts here ------------------------------

        # Initialise gravity vector
        g = np.zeros(7)

        # Gravity acceleration vector in base frame
        gravity = np.array([0, 0, -self.g])

        # Compute the transformation matrix to the center of mass of link i
        for i in range(7):
            T_com = self.forward_kinematics_centre_of_mass(joint_readings, up_to_joint=i + 1)

            # Extract position of the centre of mass in the base frame
            p_com = T_com[:3, 3]

            # Compute the gravity force activing on the centre of mass 
            F_g = self.mass[i] * gravity 

            # Compute the torque due to gravity at joint i
            z_axis = self.forward_kinematics(joint_readings, up_to_joint=i)[:3, 2]
            torque = np.cross(p_com, F_g)
            g[i] = np.dot(z_axis, torque)

        # Your code ends here ------------------------------

        assert isinstance(g, np.ndarray)
        assert g.shape == (7,)
        return g
