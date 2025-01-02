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

        # Initialise Jacobian matrix
        jacobian = np.zeros((6, 7)) 

        # Transformation matrix to the centre of mass
        T0_com = self.forward_kinematics_centre_of_mass(joint_readings, up_to_joint)
        p_com = T0_com[:3, 3]

        # Compute Jacobian columns for each joint
        T = []
        for i in range(up_to_joint):
            T.append(self.forward_kinematics(joint_readings, i))
            T_prev = T[i]
            # z-axis of previous join
            z_axis = T_prev[:3, 2]
            p_prev = T_prev[:3, 3]

            jacobian[:3,i] = np.cross(z_axis, (p_com - p_prev))
            jacobian[3:6, i] = z_axis
        
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
       
        for i in range(1, 8): 
            # Compute rotation matrix to the centre of mass
            R0_com = self.forward_kinematics_centre_of_mass(joint_readings, i)[:3, :3]

            # Compute link inertia in global frame
            B_loc = np.diag(self.Ixyz[i - 1, :])
            B_glob = R0_com @ B_loc @ R0_com.T

            # Compute Jacobian at the centre of mass 
            J_com = self.get_jacobian_centre_of_mass(joint_readings, i)

            # Linear component of Jacobian
            J_v = J_com[:3, :]

            # Angular component of Jacobian
            J_w = J_com[3:6, :]

            # Add link's contribution to inertia matrix B
            mass_link = self.mass[i - 1]
            B += (mass_link * (J_v.T @ J_v) + (J_w.T @ B_glob @ J_w))
            
        # Your code ends here ------------------------------
        
        return B
    
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

        # Retrieve inertia matrix B for the given joint positions
        B = self.get_B(joint_readings)
        C = np.zeros((7,7))

        # Small step for numerical derivatives
        delta = 1e-3

        # Compute the Coriolis terms
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    # Perturb the joint positions for the kth joint
                    p_k = np.copy(joint_readings)
                    p_k[k] += delta
                    B_ij = self.get_B(p_k.tolist() )

                    # Perturb the joint positions for the ith joint
                    p_i = np.copy(joint_readings)
                    p_i[i] += delta
                    B_jk = self.get_B((p_i).tolist())
                    
                    # Approximate partial derivatives
                    d_B_ij_dq_k = (B_ij[i,j] - B[i,j]) / delta
                    d_B_jk_dq_i = (B_jk[j,k] - B[j,k]) / delta
                    
                    # Compute the Coriolis term for the current indices
                    C[i,j] += (d_B_ij_dq_k - 0.5 * d_B_jk_dq_i) * joint_velocities[k]
                                
        C = np.matmul(C, np.array(joint_velocities))
        
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
        
        # Gravity matrix of zeros
        G = np.zeros((7,))

        # Gravity vector in the robot's base frame
        g = np.array([0, 0, self.g])

        # Compute the gravity terms of each joint
        for i in range(7):
            G_i = 0

            # Sum contributions from all links up to the current joint
            for j in range(7):
                mass_link = self.mass[j]
                
                # Compute the Jacobian of the centre of mass for the current link
                J_com = self.get_jacobian_centre_of_mass(joint_readings, j + 1)[:3,i]
                J_com = J_com.reshape(3,1)

                # Compute the contribution of the link to the gravity term
                G_i += mass_link * np.dot(g, J_com) 
                
            G[i] = G_i

        # Your code ends here ------------------------------

        assert isinstance(g, np.ndarray)
        assert G.shape == (7,)
        return G
