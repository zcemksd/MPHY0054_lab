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
        # List to store timestamps
        self.time_stamps = []   

        # Store acceleration data for each joint
        self.joint_accelerations = [[] for _ in range(7)]    

    def load_trajectory(self):
        """
        Load trajectory from the bagfile and prepare it for publishing.
        
        Returns:
            JointTrajectory: The trajectory extracted from the bagfile.
        """

        rospy.loginfo("Loading trajectory from bagfile...")
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()


        rospack = rospkg.RosPack()
        bagfile_path = rospack.get_path('cw3q5') + '/bag/cw3q5.bag'

        try:
            with rosbag.Bag(bagfile_path, 'r') as bag:
                for topic, msg, t in bag.read_messages(topics=['/iiwa/EffortJointInterface_trajectory_controller/command']):
                    joint_traj.joint_names = msg.joint_names
                    for point in msg.points:
                        point_obj = JointTrajectoryPoint(
                        positions=point.positions,
                        velocities=point.velocities,
                        accelerations=point.accelerations,
                        time_from_start=point.time_from_start
                        )
                        joint_traj.points.append(point_obj)
                    
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

        q = np.array(joint_state.position)
        q_dot = np.array(joint_state.velocity)
        tau = np.array(joint_state.effort)

        print(f"q shape: {q.shape}, q:{q}")
        print(f"q_dot shape: {q_dot.shape}, q_dot:{q_dot}")
        print(f"tau shape: {tau.shape}, tau:{tau}")
            

        
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

        print(f"np.linalg.inv(B) shape: {np.linalg.inv(B).shape}")

        q_ddot = q_ddot.reshape((7,))

        print(f"q_ddot shape: {q_ddot.shape}")


    def plot_acceleration(self, q_ddot, time):
        """Plot joint accelerations as a function of time."""

        rospy.loginfo("Plotting joint accelerations...")

        for i in range(7):
            plt.plot(
                time,
                q_ddot[i],
                label=f"Joint {i+1}"
            )
        
        plt.title("Joint Acceleration Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (rad/s^2)")
        plt.legend(loc="upper right")

        plt.draw()
        plt.pause(1e-5)
        

if __name__ == "__main__":

    try:
        rospy.init_node("joint_acceleration_calculator", anonymous=True)
        calculator = JointAccelerationCalculator()

        joint_traj = calculator.load_trajectory()
        if joint_traj:
            traj_publisher = rospy.Publisher('/iiwa/EffortJointInterface_trajectory_controller/command', JointTrajectory, queue_size=5)
            rospy.sleep(1)
            traj_publisher.publish(joint_traj)
            rospy.loginfo("Trajectory published to topic.")

        rospy.Subscriber('/iiwa/joint_states', JointState, calculator.calculate_acceleration)

        plt.ion()
        plt.show()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node terminated.")
