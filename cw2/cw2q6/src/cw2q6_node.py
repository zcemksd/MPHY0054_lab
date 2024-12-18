#!/usr/bin/env python3
import numpy as np
import rospy
import rosbag
import rospkg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cw2q4.youbotKineKDL import YoubotKinematicKDL
from itertools import permutations

import PyKDL
from visualization_msgs.msg import Marker


class YoubotTrajectoryPlanning(object):
    def __init__(self):
        # Initialize node
        rospy.init_node('youbot_traj_cw2', anonymous=True)

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicKDL()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = rospy.Publisher('/EffortJointInterface_trajectory_controller/command', JointTrajectory,
                                        queue_size=5)
        self.checkpoint_pub = rospy.Publisher("checkpoint_positions", Marker, queue_size=100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        rospy.loginfo("Waiting 5 seconds for everything to load up.")
        rospy.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        self.traj_pub.publish(traj)

    def q6(self):
        """ This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # Steps to solving Q6.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------

        # Load in targets from bagfile
        target_cart_tf, target_joint_positions = self.load_targets()

        # Compute shorterst path visting eah checkpoint position
        sorted_order, _ = self.get_shortest_path(target_cart_tf)

        # Determine intermediate checkpoints to achieve a linear path between each checkpoint
        # Generate intermediate transformations alonge the sorted path with 10 points per segment
        intermediate_tfs = self.intermediate_tfs(sorted_order, target_cart_tf, num_points=10)

        # Publish checkpoints for visualisation
        self.publish_traj_tfs(intermediate_tfs)

        # Convert all checkpoints into joint values using IK solver
        init_joint_position = target_joint_positions[:, 0]
        joint_positions = self.full_checkpoints_to_joints(intermediate_tfs, init_joint_position)

        traj = JointTrajectory()
        for i in range(joint_positions.shape[1]):
            point = JointTrajectory()

            # Joint positions for current step
            point.positions = joint_positions[:, i]

            # Increment time step by 0.1 sec for each point
            point.time_from_start = rospy.Duration(i * 0.1)
            traj.points.append(point)

        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the 'data.bag' file. In the bag file, you will find messages
        relating to the target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        # Defining ros package path
        rospack = rospkg.RosPack()
        path = rospack.get_path('cw2q6')

        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, 5))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(np.identity(4), 5, axis=1).reshape((4, 4, 5))

        # Load path for selected question
        bag = rosbag.Bag(path + '/bags/data.bag')
        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.kdl_jnt_array_to_list(self.kdl_youbot.current_joint_position)
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(target_joint_positions[:, 0])

        # Your code starts here ------------------------------

        i = 1
        for topic, msg, t in bag.read_messages(topics=['joint_data']):

            # Extract joint positions from current message 
            # Populate all joint values for the i-th checkpoint
            target_joint_positions[i, :] = msg.position

            # Pass joint positions to compute T matrix for end-effector position
            target_cart_tf[:, :, i] = self.kdl_youbot.forward_kinematics(target_joint_positions[i, :])

            # Increments index to process next checkpoint 
            i += 1

            # Check if 5 checkpoints have been processed
            # Exit loop if 5 or more checkpoints are processed
            if i >= 5:
                break

        # Your code ends here ------------------------------

        # Close the bag
        bag.close()

        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, 5)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, 5)

        return target_cart_tf, target_joint_positions

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray.
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """

        # Your code starts here ------------------------------

        # Extract each checkpoint position from the transformation matrices
        # There are 5 checkpoints
        num_checkpoints = checkpoints_tf.shape[2]

        # Create a list to store checkpoint Cartesian positions
        positions = []

        # i takes values 0 to 4 (5 checkpoints)
        # Extracted positions for each checkpoint are added to positions list
        for i in range(num_checkpoints):
            position = checkpoints_tf[:3, 3, i]
            positions.append(position)
        positions = np.array(positions)

        # Track shortest path by starting with an infinitely large distance
        min_dist = float('inf')

        # Store best checkpoint order
        sorted_order = None

        # Generate all permutations or possible orders of checkpoint indices
        all_orders = permutations(range(1, num_checkpoints))

        # Evaluate each order to find the shortest path
        # Starting point 0 added at the beginning
        # Track total distance for the current path
        for order in all_orders:
            current_order = [0] + list(order)
            current_distance = 0

            # Calculate total distance for the current order
            # Use sum of Euclidean distances between consecutive positions
            for j in range(len(current_order) - 1):
                start = positions[current_order[j]]
                end = positions[current_order[j + 1]]
                distance = np.linalg.norm(start - end)
                current_distance += distance

            # Update minimum distance and best order if current path is better
            # Convert the best order to a numpy array 
            if current_distance < min_dist:
                min_dist = current_distance
                sorted_order = np.array(current_order)


        # Your code ends here ------------------------------

        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (5,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = 'base_link'
            marker.header.stamp = rospy.Time.now()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.checkpoint_pub.publish(marker)

    def intermediate_tfs(self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points):
        """This function takes the target checkpoint transforms and the desired order based on the shortest path sorting, 
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """

        # Your code starts here ------------------------------

        # List to store all intermediate transformation matrices
        all_transforms = []
        
        # Loop through each pair of consecutive checkpoints to:
        # 1. get the index of the current and next checkpoints from the list
        # 2. extract the transformation matrices for the two checkpoints
        # 3. generate intermediate transformations between the checkpoints
        # 4. add the intermediate matrices to the list
        for i in range(len(sorted_checkpoint_idx) - 1):
            start_idx = sorted_checkpoint_idx[i]
            end_idx = sorted_checkpoint_idx[i + 1]
            start_tf = target_checkpoint_tfs[:, :, start_idx]
            end_tf = target_checkpoint_tfs[:, :, end_idx]

            # Create intermediate transformations
            intermediate_transforms = self.decoupled_rot_and_trans(start_tf, end_tf, num_points)

            # Add intermediate transformations to the list
            all_transforms.append(intermediate_transforms)

        # Combine all intermediate transformations in an array
        full_checkpoint_tfs = np.concatenate(all_transforms, axis=2)
        
        # Your code ends here ------------------------------
       
        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:
            tfs: 4x4x(num_points) homogeneous transformations matrices describing the full desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """

        # Your code starts here ------------------------------

        # Initialise an array to store intermediate transformation matrices
        # 4x4
        tfs = np.zeros((4,4, num_points))

        # Extract rotation and translation components from the 4x4 matrices
        # Rotation matrix of the starting pose
        start_rot = PyKDL.Rotation(checkpoint_a_tf[:3, :3])

        # Rotation matrix of the ending pose
        end_rot = PyKDL.Rotation(checkpoint_b_tf[:3, :3])

        # Translation vector of the starting pose
        start_position = checkpoint_a_tf[:3, 3]

        # Translation vector of the ending pose
        end_position = checkpoint_b_tf[:3, 3]

        # Calculate intermediate transformations by:
        # 1. finding the interpolation factor alpha between 0 and 1 (starting and ending pose)
        # 2. computing the position fo the current intermediate step
        # 3. calculating the intermediate rotation
        # 4. building the homogenous 4x4 transformation matrix
        for step in range(num_points):
            alpha = step / (num_points -1)

            # Position of current intermediate step
            interpolated_position = (1 - alpha) * start_position + alpha * end_position

            # Calculate intermediate rotation
            interpolated_rotation = start_rot.Interpolate(end_rot, alpha)

            # Top-left 3x3 of the matrix: interpolated rotation
            tfs[:3, :3, step] = interpolated_rotation

            # Last column of the 4x4 matrix: translation vector 3x1
            tfs[:3, 3, step] = interpolated_position

            # Bottom row of the 4x4 matrix: [0, 0, 0, 1] 
            tfs[3, :, step] = [0, 0, 0, 1]

        # Your code ends here ------------------------------

        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints, 
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        
        # Your code starts here ------------------------------

        # Number of checkpoints
        num_checkpoints = full_checkpoint_tfs.shape[2]

        # Matrix to store joint positions
        q_checkpoints = np.zeros((5, num_checkpoints))

        # Initial joint position
        current_joint_positions = init_joint_position.copy()

        for i in range(num_checkpoints):
            current_checkpoint_pose = full_checkpoint_tfs[:, :, i]

            # Solve inverse kinematics for the current pose using position-only IK
            q, error = self.ik_position_only(current_checkpoint_pose, current_joint_positions)

            # Store solution in joint positions matrix
            q_checkpoints[:, i] = q.flatten()

            # Update initial guess for next checkpoint to current solution
            current_joint_positions = q.copy() 

        # Your code ends here ------------------------------

        return q_checkpoints 

    def ik_position_only(self, pose, q0):
        """This function implements position only inverse kinematics.
        Args:
            pose (np.ndarray, 4x4): 4x4 transformations describing the pose of the end-effector position.
            q0 (np.ndarray, 5x1):A 5x1 array for the initial starting point of the algorithm.
        Returns:
            q (np.ndarray, 5x1): The IK solution for the given pose.
            error (float): The Cartesian error of the solution.
        """
        # Some useful notes:
        # We are only interested in position control - take only the position part of the pose as well as elements of the
        # Jacobian that will affect the position of the error.

        # Your code starts here ------------------------------

        # Maximum number of iterations
        max_iterations = 100

        # Positioni error threshold to stop iteration
        error_tolerance = 1e-3

        # Initialise joint positions with the initial guess
        q = q0.copy()

        # Get target position from the pose matrix
        target_position = pose[:3, 3]

        for i in range(max_iterations):
            # FK to get current end-effector position
            current_pose = self.kdl_youbot.forward_kinematics(q)
            current_position = current_pose[:3, 3]

            # Find position error
            position_difference = target_position - current_position
            error = np.linalg.norm(position_difference)

            # Error should be above set threshold
            if error < error_tolerance:
                break
            
            # Calculate Jacobian matrix 
            full_jacobian = self.kdl_youbot.get_jacobian(q)
            position_jacobian = full_jacobian[:3, :]

            # Compute pseudo-inverse of the position Jacobian
            jacobian_pseudo_inverse = np.linalg.pinv(position_jacobian)

            # Newton-Raphson method for updating the joint positions 
            delta_joint_positions = jacobian_pseudo_inverse @ position_difference
            q += delta_joint_positions

        # Your code ends here ------------------------------

        return q, error

    @staticmethod
    def list_to_kdl_jnt_array(joints):
        """This converts a list to a KDL jnt array.
        Args:
            joints (joints): A list of the joint values.
        Returns:
            kdl_array (PyKDL.JntArray): JntArray object describing the joint position of the robot.
        """
        kdl_array = PyKDL.JntArray(5)
        for i in range(0, 5):
            kdl_array[i] = joints[i]
        return kdl_array


if __name__ == '__main__':
    try:
        youbot_planner = YoubotTrajectoryPlanning()
        youbot_planner.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
