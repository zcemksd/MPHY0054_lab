#!/usr/bin/env python3

import rospy
import rosbag
import rospkg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL
import numpy as np
import matplotlib.pyplot as plt


class JointAccelerationCalculator:
    """ 
    Class to manage trajectory planning and acceleration calculation for the iiwa robot.
    """
    def __init__(self):
        """
        Initialise the JointAccelerationCalculator class.
        """
        # List to store timestamps i.e. time for each joint state
        self.time_stamps = []   

        # Store acceleration data for each joint
        self.joint_accelerations = np.zeros((7,))  


    def load_trajectory(self):
        """
        Load trajectory from the bagfile and prepare it for publishing.
        
        Returns:
            JointTrajectory: The trajectory extracted from the bagfile.
        """

        rospy.loginfo("Loading trajectory from bagfile...")
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()

        # Resolve path to the bagfile
        rospack = rospkg.RosPack()
        bagfile_path = rospack.get_path('cw3q5') + '/bag/cw3q5.bag'

        try:
            with rosbag.Bag(bagfile_path, 'r') as bag:
                # Initialise message count
                message_count = 0

                for topic, msg, t in bag.read_messages(topics=['/iiwa/EffortJointInterface_trajectory_controller/command']):
                    # Increment message count
                    message_count += 1
                    joint_traj.joint_names = msg.joint_names

                    for point in msg.points:
                        point_obj = JointTrajectoryPoint(
                        positions=point.positions,
                        velocities=point.velocities,
                        accelerations=point.accelerations,
                        time_from_start=point.time_from_start
                        )
                        joint_traj.points.append(point_obj)

                # Print message type, number of message, and content of the messages
                print(f"Loaded {message_count} messages with joint names: {joint_traj.joint_names}")
                    
            rospy.loginfo("Trajectory successfully loaded from bagfile.")
            print(f"Loaded trajectory: {joint_traj}")

            return joint_traj

        except Exception as e:
            rospy.logerr(f"Error loading trajectory from bagfile: {e}")
            return None
        
        
    def calculate_acceleration(self, joint_state):
        """
        Calculate joint accelerations using dynamics.
        
        Args:
            joint_state (JointState): The joint state message containing positions, velocity, and effort.
        """

        rospy.loginfo("Calculating joint accelerations...")

        # Extract joint positions, velocities, and efforts.
        q = np.array(joint_state.position)
        q_dot = np.array(joint_state.velocity)
        tau = np.array(joint_state.effort)

        print(f"q shape: {q.shape}, q:{q}")
        print(f"q_dot shape: {q_dot.shape}, q_dot:{q_dot}")
        print(f"tau shape: {tau.shape}, tau:{tau}")
        
        try:
            # Calculate dynamic components 
            B = Iiwa14DynamicKDL.get_B(Iiwa14DynamicKDL(), q)
            C_qdot = np.array(Iiwa14DynamicKDL.get_C_times_qdot(Iiwa14DynamicKDL(), q, q_dot))
            G = np.array(Iiwa14DynamicKDL.get_G(Iiwa14DynamicKDL(), q))

            print(f"B shape: {B.shape}, B:{B}")
            print(f"C_qdot shape: {C_qdot.shape}, C_qdot:{C_qdot}")
            print(f"G shape: {G.shape}, B:{G}")
            print(f"tau - C_qdot - G shape: {(tau - C_qdot - G).shape}")

            # Compute joint accelerations
            q_ddot = np.linalg.inv(B).dot(tau - C_qdot - G)
            q_ddot = q_ddot.reshape((7,))

            print(f"q_ddot shape: {q_ddot.shape}")

            self.plot_acceleration(joint_state.header.stamp, q_ddot)

        except Exception as e:
            rospy.logerr(f"Error during matrix inversion: {e}")

        
    def plot_acceleration(self, stamp, q_ddot):
        """
        Plot the joint accelerations as a function of time.

        Args:
            stamp (rospy.Time): The timestamp of the current joint state.
            q_ddot (numpy.ndarray): The calculated joint accelerations.
        """

        # Convert ROS time to seconds
        time = stamp.secs + stamp.nsecs * 1e-9
        self.time_stamps.append(time)
        print(f"Time in seconds: {time}")

        # Plot each joint and add label to the legend
        plt.plot(time, q_ddot[:,0], 'k*', label='Joint 1')
        plt.plot(time, q_ddot[:,1], 'r*', label='Joint 2')
        plt.plot(time, q_ddot[:,2], 'b*', label='Joint 3')
        plt.plot(time, q_ddot[:,3], 'g*', label='Joint 4')
        plt.plot(time, q_ddot[:,4], 'm*', label='Joint 5')
        plt.plot(time, q_ddot[:,5], 'c*', label='Joint 6')
        plt.plot(time, q_ddot[:,6], 'y*', label='Joint 7')

        # Legend displays each joint once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # Add title and axis titles to the graph
        plt.title('Joint Accelerations vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Acceleration (rad/s^2)')

        plt.draw
        plt.pause(1e-5)


if __name__ == "__main__":

    try:
        rospy.init_node("joint_acceleration_calculator", anonymous=True)
        calculator = JointAccelerationCalculator()

        # Load and publish trajectory
        joint_traj = calculator.load_trajectory()

        if joint_traj:
            traj_publisher = rospy.Publisher('/iiwa/EffortJointInterface_trajectory_controller/command', JointTrajectory, queue_size=5)
            rospy.sleep(1)
            traj_publisher.publish(joint_traj)
            rospy.loginfo("Trajectory published to topic.")

        # Subscribe to joint states and calculate acceleration
        rospy.Subscriber('/iiwa/joint_states', JointState, calculator.calculate_acceleration)
        
        # Initialise plot
        plt.ion()
        plt.show()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node terminated.")
